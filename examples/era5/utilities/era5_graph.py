"""ERA5 graph generator: radius neighbourhoods over the global grid, plus node features.

``nofe`` consumes a precomputed graph, so the ERA5 examples build theirs here. Coordinates are
normalised Cartesian directions on a sphere of radius one, and the neighbourhood is a radius graph
with observed rows as the only senders.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graph_utils import merge_queries as _merge_queries  # noqa: E402
from graph_utils import radius_edges  # noqa: E402

EARTH_RADIUS = 6_371_000.0  # metres; positions are coordinates / EARTH_RADIUS
VARIABLES = ("w", "v", "z", "q", "r", "t", "d", "o3", "pv", "u", "vo")
DEFAULT_DATA_DIR = "/Volumes/T7-storage/ERA5/data"
FILL = 10  # neighbours used by the inverse-distance substitute on withheld rows


class Era5Graph:
    """Build radius-neighbourhood graphs over the ERA5 grid, plus node features."""

    def __init__(self, *, radius: float, stride: int = 2, data_dir: Path | None = None) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.radius = float(radius)
        self.stride = int(stride)
        self.data_dir = Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)

    def positions(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """Coordinates are in earth radii, so ``|p| = 1`` for every node; a radius r is therefore a
        chord length, equal to ``2*sin(theta/2)`` for an angular separation theta."""
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

    def _radius_edges(
        self, pos: Tensor, sources: np.ndarray, targets: np.ndarray
    ) -> tuple[Tensor, Tensor]:
        """Radius neighbourhood with ``sources`` sending and ``targets`` receiving."""
        return radius_edges(pos, self.radius, sources=sources, targets=targets)

    def build(
        self,
        features: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        *,
        query_mask: np.ndarray | None = None,
        query_features: np.ndarray | None = None,
        fill: str = "interpolate",
    ) -> Data:
        """Raster adapter: positions and features plus ``query_mask`` to a ``Data``.

        This is the mask-shaped door for callers that hold a full raster (the ERA5 example and the
        notebooks); it splits the raster into known and withheld rows and delegates to
        ``graph_utils.merge_queries``, the block form. Passing the grid ids through keeps
        ``data.observed_index``/``data.withheld_index`` in grid order, so a caller can scatter graph
        rows back into its own raster. Queried locations off the raster go through
        ``merge_queries(graph.positions(lat, lon), ...)`` directly.

        Locations, not features, are unioned: withheld rows are placed in the graph after the
        known ones, receive their neighbours' messages and send none, keep a ``fill`` substitute in
        ``x`` and their truth in ``y``.
        """
        pos = self.positions(latitudes, longitudes)
        flat = features.reshape(-1, features.shape[2])
        truth = None
        if query_features is not None:
            if query_features.shape[2:] != features.shape[2:]:
                raise ValueError(
                    f"query_features has {query_features.shape[-1]} channels but features has "
                    f"{features.shape[2]}"
                )
            truth = query_features.reshape(flat.shape)
        if query_mask is None:
            data = _merge_queries(pos, flat, fill=fill, k=FILL, edge_builder=self._radius_edges)
            self.report(data)
            return data

        flags = np.asarray(query_mask).reshape(-1).astype(bool)
        if flags.size != flat.shape[0]:
            raise ValueError(f"query_mask must have {flat.shape[0]} entries, got {flags.size}")
        observed = np.flatnonzero(~flags)
        withheld = np.flatnonzero(flags)
        if observed.size == 0:
            raise ValueError("query_mask withholds every row; nothing would be observed")
        data = _merge_queries(
            pos[observed],
            flat[observed],
            pos[withheld],
            None if truth is None else truth[withheld],
            ids=observed,
            query_ids=withheld,
            fill=fill,
            k=FILL,
            edge_builder=self._radius_edges,
        )
        self.report(data)
        return data

    def report(self, data: Data) -> str:
        """Print and return the neighbour statistics of ``data``."""
        counts = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
        mean = float(counts.float().mean())
        isolated = int((counts == 0).sum())
        percent = 100.0 * isolated / max(1, data.num_nodes)
        arc = math.degrees(2.0 * math.asin(min(1.0, self.radius / 2.0)))
        text = (
            f"radius {self.radius:.5f} (chord on the unit sphere, {arc:.2f} deg arc), "
            f"{data.num_nodes} nodes, {data.edge_index.shape[1]} edges\n"
            f"mean neighbours per receiver: {mean:.2f}  "
            f"max: {int(counts.max()) if counts.numel() else 0}\n"
            f"nodes without a neighbour:    {isolated} ({percent:.2f}%)"
        )
        print(text)
        return text
