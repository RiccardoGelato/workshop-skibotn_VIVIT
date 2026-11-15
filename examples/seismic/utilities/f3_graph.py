"""F3 cube: one graph node per horizontal location, its trace as the feature vector.

The NOFE demonstration on the Netherlands F3 benchmark treats the **horizontal plane as the
domain** and the **vertical axis as the feature space**: every (inline, crossline) location carries
the 253 amplitude samples of its seismic trace, and the operator learns a 3-D embedding of those
traces that preserves how laterally neighbouring traces relate.

Two consequences drive the choices here:

- **The domain is a regular raster**, so a radius neighbourhood is exactly a fixed set of
  (row, column) offsets. Deriving the offsets from the bin size is exact, needs no index search
  even for 541,501 nodes, and gives the same physical operator support at any grid stride.
- **The vertical axis is a feature axis, not a node axis.** Locations are connected laterally only;
  a node is one whole trace, so nothing about the layering is baked into the graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graph_utils import geometry_edge_attr  # noqa: E402

# 25 m inline/crossline bin spacing of the F3 acquisition grid (Alaudah et al. 2019, sec. 3.2.2).
BIN_KM = 0.025
# Vertical planes 0 and 254 are padding: constant amplitude with class-0 labels throughout. They are
# flagged in the shipped metadata (`suspect_vertical_planes`), and a trace built from them would
# describe the padding rather than the geology.
PADDING_PLANES = (0, 254)
DATA_SUFFIX = Path("data") / "f3-full-ready" / "f3-full" / "data" / "processed" / "f3" / "full"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / DATA_SUFFIX


class F3Cube:
    """Load the F3 volume and turn horizontal rasters of traces into ``Data`` graphs."""

    def __init__(
        self,
        *,
        radius_km: float = 0.1,
        data_dir: Path | None = None,
        stride: int = 1,
    ) -> None:
        if radius_km <= 0:
            raise ValueError("radius_km must be positive")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.radius_km = float(radius_km)
        self.stride = int(stride)
        self.data_dir = Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
        self._scale: tuple[np.ndarray, np.ndarray] | None = None
        self._shape: tuple[int, int, int] | None = None

    # -- data -----------------------------------------------------------------

    def offsets(self) -> list[tuple[int, int]]:
        """``(row, column)`` lattice offsets whose centres lie inside ``radius_km``.

        The radius is a physical distance, so the step count is ``round(radius / bin)``: 50 m is two
        25 m bins, giving the 12 offset rings that define the operator's support. Exact because the
        domain is a regular raster — no index search, and the same neighbour set for every interior
        node.
        """
        steps = max(1, round(self.radius_km / BIN_KM))
        return [
            (d_row, d_col)
            for d_row in range(-steps, steps + 1)
            for d_col in range(-steps, steps + 1)
            if 0 < d_row * d_row + d_col * d_col <= steps * steps
        ]

    def geometry(self) -> tuple[int, int, int]:
        """``(rows, cols, depth_samples)`` of the source volume, with padding planes removed."""
        if self._shape is None:
            volume = self._open()
            # Count the padding planes that actually exist: the constants index the 255-sample F3
            # axis, so a shallower volume must not have them subtracted blindly.
            padding = sum(1 for z in PADDING_PLANES if z < volume.shape[2])
            self._shape = (volume.shape[0], volume.shape[1], volume.shape[2] - padding)
        return self._shape

    def _open(self):
        path = self.data_dir / "seismic.npy"
        if not path.exists():
            raise SystemExit(
                f"no F3 volume at {path}; put the f3-full package under examples/seismic/data/ "
                "or pass a data directory"
            )
        volume = np.load(path, mmap_mode="r")
        if volume.ndim != 3:
            raise ValueError(f"seismic.npy must be 3-D, got shape {volume.shape}")
        return volume

    def load(self, stride: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sub-sampled traces ``[R, C, depth]`` plus the retained source row and column indices."""
        stride = self.stride if stride is None else int(stride)
        volume = self._open()
        depth = [z for z in range(volume.shape[2]) if z not in PADDING_PLANES]
        rows = np.arange(0, volume.shape[0], stride)
        cols = np.arange(0, volume.shape[1], stride)
        # Two gathers rather than one: the row gather already shrinks the block the column gather
        # then reads, so a stride-1 load never materialises the full 527 MB twice.
        features = np.asarray(volume[rows][:, cols][:, :, depth], dtype=np.float32)
        return features, rows, cols

    def positions(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Node coordinates in km, divided by the cube's longer side so the domain spans ``[0, 1]``.

        The divisor is the source cube's extent, **not** the last retained index: normalising by the
        sampled span would rescale with the stride and silently change what a physical radius means.
        The two axes keep their true metric ratio (15 km x 22.5 km), so a radius in the graph is a
        physical distance rather than a square stretched over a rectangle.
        """
        source_rows, source_cols, _ = self.geometry()
        span = max(source_rows, source_cols) * BIN_KM
        row_km = np.asarray(rows, dtype=np.float32) * BIN_KM / span
        col_km = np.asarray(cols, dtype=np.float32) * BIN_KM / span
        pos = np.empty((row_km.size * col_km.size, 2), dtype=np.float32)
        pos[:, 0] = np.repeat(row_km, col_km.size)
        pos[:, 1] = np.tile(col_km, row_km.size)
        return pos

    def normalise(self, features: np.ndarray) -> np.ndarray:
        """Standardise every vertical sample by global constants, fitted once.

        One global mean/std for the whole depth axis keeps the relative amplitude of shallow and
        deep samples intact — a per-sample z-score would erase exactly the vertical structure the
        embedding is asked to encode. The constants are cached, so the training raster and the
        finer inference raster are scaled by the same numbers.
        """
        flattened = features.reshape(-1, features.shape[-1])
        if self._scale is None:
            probe = flattened[:: max(1, flattened.shape[0] // 20_000)]
            self._scale = (probe.mean(axis=0), probe.std(axis=0) + 1e-6)
        return self.apply_scale(flattened)

    def apply_scale(self, features: np.ndarray) -> np.ndarray:
        """Scale ``features`` with this cube's fitted constants, without refitting them."""
        if self._scale is None:
            raise RuntimeError("no normalisation constants yet; call normalise() first")
        mean, std = self._scale
        return ((features.reshape(-1, features.shape[-1]) - mean) / std).astype(np.float32)

    def adopt_scale(self, other: F3Cube) -> None:
        """Use ``other``'s normalisation constants, so two rasters share one feature scaling."""
        if other._scale is None:
            raise RuntimeError("the source cube has no normalisation constants yet")
        self._scale = other._scale

    def reset_scale(self) -> None:
        """Forget the cached normalisation constants (fit them again on the next build)."""
        self._scale = None

    # -- graph ----------------------------------------------------------------

    def _lattice_edges(
        self, num_rows: int, num_cols: int, offsets: list[tuple[int, int]]
    ) -> torch.Tensor:
        """``[2, E]`` sender/receiver pairs for a raster, one block per lattice offset."""
        ids = np.arange(num_rows * num_cols, dtype=np.int64).reshape(num_rows, num_cols)
        senders: list[np.ndarray] = []
        receivers: list[np.ndarray] = []
        for d_row, d_col in offsets:
            receiver = ids[
                max(0, -d_row) : num_rows - max(0, d_row),
                max(0, -d_col) : num_cols - max(0, d_col),
            ]
            sender = ids[
                max(0, d_row) : num_rows - max(0, -d_row) or None,
                max(0, d_col) : num_cols - max(0, -d_col) or None,
            ]
            senders.append(sender.ravel())
            receivers.append(receiver.ravel())
        src = np.concatenate(senders)
        dst = np.concatenate(receivers)
        # Sorted by (receiver, sender): the order `radius_edges` produces, so the offset loop
        # leaves no trace in edge order.
        order = np.lexsort((src, dst))
        edge_index = torch.empty((2, src.size), dtype=torch.long)
        edge_index[0] = torch.from_numpy(src[order])
        edge_index[1] = torch.from_numpy(dst[order])
        return edge_index

    def build(self, features: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> Data:
        """One node per retained trace, radius edges over the horizontal plane."""
        num_rows, num_cols = features.shape[:2]
        pos = self.positions(rows, cols)
        edge_index = self._lattice_edges(num_rows, num_cols, self.offsets())
        pos_tensor = torch.from_numpy(pos)
        data = Data(
            x=torch.from_numpy(self.normalise(features).copy()),
            pos=pos_tensor,
            edge_index=edge_index,
            edge_attr=geometry_edge_attr(pos_tensor, edge_index),
        )
        data.grid_rows = int(num_rows)
        data.grid_cols = int(num_cols)
        data.source_rows = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        data.source_cols = torch.from_numpy(np.asarray(cols, dtype=np.int64))
        self.report(data)
        return data

    def report(self, data: Data) -> str:
        """Print and return the neighbourhood statistics of ``data``."""
        degrees = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
        isolated = int((degrees == 0).sum())
        text = (
            f"radius {self.radius_km * 1000:.0f} m on a {BIN_KM * 1000:.0f} m grid, "
            f"{data.num_nodes} traces, {data.edge_index.shape[1]} edges, "
            f"{data.x.shape[1]} vertical samples per trace\n"
            f"mean neighbours per trace: {float(degrees.float().mean()):.2f}  "
            f"max: {int(degrees.max())}  isolated: {isolated} "
            f"({100.0 * isolated / max(1, data.num_nodes):.2f}%)"
        )
        print(text)
        return text
