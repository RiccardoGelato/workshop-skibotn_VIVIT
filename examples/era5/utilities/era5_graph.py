"""ERA5 graph generator: radius neighbourhoods over the global grid, plus node features.

``nofe.graph.PointGraph`` owns assembly; this subclass supplies the two rules it cannot know: the
radius neighbourhood (banded by latitude on a full raster, so a stride-1 grid stays buildable) and
the radius itself. Coordinates are normalised Cartesian directions on a sphere of radius one, so a
radius is a chord length and the poles and the antimeridian need no special casing.

``normalize=False`` is deliberate: ``load`` already standardises every variable per field, and
z-scoring the unit-sphere coordinates would turn the documented chord into a radius in a stretched
space, changing which grid points are neighbours.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

from nofe import PointGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graph_utils import radius_edges, raster_radius_edges  # noqa: E402

EARTH_RADIUS = 6_371_000.0  # metres; positions are coordinates / EARTH_RADIUS
VARIABLES = ("w", "v", "z", "q", "r", "t", "d", "o3", "pv", "u", "vo")
DEFAULT_DATA_DIR = "/Volumes/T7-storage/ERA5/data"


class Era5Graph(PointGraph):
    """Radius-neighbourhood graphs over the ERA5 grid, plus node features."""

    def __init__(
        self,
        *,
        radius: float,
        stride: int = 2,
        data_dir: Path | None = None,
        **kwargs,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer")
        super().__init__(normalize=False, **kwargs)
        self.radius = float(radius)
        self.stride = int(stride)
        self.data_dir = Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
        self._raster: tuple[int, int] | None = None

    def positions(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """Coordinates are in earth radii, so ``|p| = 1`` for every node; a radius r is therefore a
        chord length, equal to ``2*sin(theta/2)`` for an angular separation theta."""
        self._raster = (int(latitudes.size), int(longitudes.size))
        phi = np.deg2rad(np.repeat(latitudes, longitudes.size))
        lam = np.deg2rad(np.tile(longitudes, latitudes.size))
        unit = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=1)
        # Keeping the metres round trip visible is the point: the unit is earth radii, not metres.
        return (unit * EARTH_RADIUS / EARTH_RADIUS).astype(np.float32)

    def load(self, date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardised ``[lat, lon, variable]`` block over the globe plus its coordinates."""
        import netCDF4

        path = self.data_dir / f"{date}.nc"
        if not path.exists():
            raise SystemExit(
                f"no ERA5 file at {path}; set NOFE_ERA5_DATA to the directory that holds "
                f"<date>.nc (default {DEFAULT_DATA_DIR}) or pass --data"
            )
        stride = self.stride
        dataset = netCDF4.Dataset(path)
        try:
            latitudes = np.asarray(dataset["latitude"][:], dtype=np.float64)
            longitudes = np.asarray(dataset["longitude"][:], dtype=np.float64)
            # ERA5 stores longitude in [0, 360) and latitude north to south; both axes are
            # reordered to increasing order so the block reshapes straight into a raster.
            longitude_order = np.argsort((longitudes + 180.0) % 360.0 - 180.0)
            longitudes = ((longitudes + 180.0) % 360.0 - 180.0)[longitude_order]
            latitude_order = np.argsort(latitudes)
            latitudes = latitudes[latitude_order]

            # The two pole rows repeat one value along every longitude: keeping them would add
            # 1440 coincident nodes with zero-length edges. Sample the rest evenly, aligned on
            # the equator so the two hemispheres are covered symmetrically.
            usable = np.flatnonzero(np.abs(latitudes) < 90.0)
            lat_rows = usable[usable.size // 2 % stride :: stride]
            lon_cols = np.arange(0, longitudes.size, stride)

            channels = []
            for name in VARIABLES:
                if name not in dataset.variables:
                    raise SystemExit(f"{path} has no variable {name!r}")
                field = np.asarray(dataset[name][0, 0], dtype=np.float32)
                block = field[np.ix_(latitude_order[lat_rows], longitude_order)][:, lon_cols]
                channels.append((block - block.mean()) / (block.std() + 1e-8))
        finally:
            dataset.close()
        return np.stack(channels, axis=-1), latitudes[lat_rows], longitudes[lon_cols]

    def edge_parameters(self) -> tuple[float, int]:
        """The chord on the unit sphere; the radius graph ignores ``k``."""
        return self.radius, 0

    def neighbour_pairs(self, points: np.ndarray, radius: float, k: int) -> tuple:
        """Radius pairs; a complete raster takes the latitude-banded search.

        The banded search is exact only over the whole raster, so it is used when the point set is
        exactly the grid the last :meth:`positions` call described and nothing was sampled;
        otherwise the straight index search applies the same rule.
        """
        if self._raster is not None and self.sample is None:
            rows, cols = self._raster
            if rows * cols == points.shape[0]:
                return raster_radius_edges(points, radius, rows=rows, cols=cols)
        return radius_edges(points, radius)

    def report(self, data, *, radius: float | None = None) -> str:
        """Print and return the neighbour statistics of ``data``."""
        radius = self.radius if radius is None else float(radius)
        counts = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
        mean = float(counts.float().mean())
        isolated = int((counts == 0).sum())
        arc = math.degrees(2.0 * math.asin(min(1.0, radius / 2.0)))
        text = (
            f"radius {radius:.5f} (chord on the unit sphere, {arc:.2f} deg arc), "
            f"{data.num_nodes} nodes, {data.edge_index.shape[1]} edges\n"
            f"mean neighbours per receiver: {mean:.2f}  "
            f"max: {int(counts.max()) if counts.numel() else 0}  "
            f"receivers with no sender: {isolated}"
        )
        print(text)
        return text

    def build_raster(self, features: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray):
        """Raster adapter: one node per grid point, at its location on the unit sphere.

        ``features`` is ``[lat, lon, variable]``; it is flattened to one row per grid point in
        row-major (lat-major) order, which is the order the plotting helpers reshape back into.
        """
        flat = features.reshape(-1, features.shape[2])
        data = self.build(flat, self.positions(latitudes, longitudes))
        self.report(data)
        return data
