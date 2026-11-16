"""Graph helpers shared by the examples.

``nofe.graph.PointGraph`` owns the graph assembly — features, locations, edge features and the
affinity target. What it cannot know is the neighbourhood rule, so the searches live here: an exact
L2 radius and k-nearest search (on faiss, or cKDTree on macOS), plus the banded raster variant the
ERA5 grid uses to stay buildable at stride 1.

The index itself comes from :mod:`nofe.search`, so the examples and the package share one backend
decision and one edge-ordering rule.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from nofe.search import NearestIndex, RadiusIndex, sorted_pairs


def _as_points(pos: Tensor | np.ndarray, name: str = "pos") -> np.ndarray:
    points = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"{name} must be 2-D [num_nodes, position_dims], got shape {points.shape}")
    return points


def tiled_radius_edges(
    points: Tensor | np.ndarray,
    radius: float,
    *,
    rows: int,
    cols: int,
    band: int = 32,
    halo: int = 6,
) -> tuple[Tensor, Tensor]:
    """Radius pairs for a ``rows x cols`` raster, searched one latitude band at a time.

    ``points`` is row-major over the raster (``index = lat_row * cols + lon_col``). Band ``b``'s
    queries are matched only against rows ``b`` +/- ``halo``, which is exact for any radius that
    cannot span more than ``halo`` rows — true for every ERA5 setting here (the widest, 2.5 deg at
    0.25 deg spacing, spans 10 rows). Longitude needs no halo: every row keeps all ``cols`` columns.

    Returns ``(sender, receiver)`` int64 tensors sorted by ``(receiver, sender)``.
    """
    values = _as_points(points)
    if rows * cols != values.shape[0]:
        raise ValueError(
            f"rows * cols = {rows * cols} must match the {values.shape[0]} points of a "
            f"{rows} x {cols} raster"
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    sender_parts: list[np.ndarray] = []
    receiver_parts: list[np.ndarray] = []
    for start in range(0, rows, band):
        stop = min(start + band, rows)
        lo = max(0, start - halo)
        hi = min(rows, stop + halo)
        index = RadiusIndex(values, np.arange(lo * cols, hi * cols))
        sender, receiver = index.within(np.arange(start * cols, stop * cols), radius)
        sender_parts.append(sender)
        receiver_parts.append(receiver)

    sender = np.concatenate(sender_parts)
    receiver = np.concatenate(receiver_parts)
    if sender.size == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty.clone()
    # Sorted by (receiver, sender) across all bands, so the banding leaves no trace in the output.
    sender, receiver = sorted_pairs(sender, receiver)
    return torch.from_numpy(sender), torch.from_numpy(receiver)


def _pairs(
    points: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    radius: float | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sender, receiver)`` global node id pairs, excluding self hits."""
    if sources.size == 0 or targets.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    if radius is not None:
        sender, receiver = RadiusIndex(points, sources).within(targets, float(radius))
        # Sorted by (receiver, sender): deterministic edge order, so graphs are reproducible.
        return sorted_pairs(sender, receiver)

    assert k is not None
    return NearestIndex(points, sources).nearest(targets, int(k))


def _build(
    pos: Tensor | np.ndarray,
    *,
    sources: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    radius: float | None = None,
    k: int | None = None,
) -> tuple[Tensor, Tensor]:
    """``(sender, receiver)`` int64 tensors sorted by ``(receiver, sender)``."""
    points = _as_points(pos)
    n = points.shape[0]
    src_ids = np.arange(n) if sources is None else np.asarray(sources, dtype=np.int64)
    dst_ids = np.arange(n) if targets is None else np.asarray(targets, dtype=np.int64)
    sender, receiver = _pairs(points, src_ids, dst_ids, radius=radius, k=k)
    sender, receiver = sorted_pairs(sender, receiver)
    return torch.from_numpy(sender), torch.from_numpy(receiver)


def radius_edges(
    pos: Tensor | np.ndarray,
    radius: float,
    *,
    sources: Tensor | np.ndarray | None = None,
    targets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Radius neighbourhood: ``(sender, receiver)`` pairs, senders from ``sources``."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    src = None if sources is None else np.asarray(sources, dtype=np.int64)
    dst = None if targets is None else np.asarray(targets, dtype=np.int64)
    return _build(pos, sources=src, targets=dst, radius=float(radius))


def raster_radius_edges(
    pos: Tensor | np.ndarray,
    radius: float,
    *,
    rows: int,
    cols: int,
) -> tuple[Tensor, Tensor]:
    """``(sender, receiver)`` radius pairs for a full ``rows x cols`` raster, banded by latitude."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    return tiled_radius_edges(pos, radius, rows=rows, cols=cols)


def knn_edges(
    pos: Tensor | np.ndarray,
    k: int,
    *,
    sources: Tensor | np.ndarray | None = None,
    targets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """k-nearest-neighbour graph: ``(sender, receiver)`` pairs, senders from ``sources``."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    src = None if sources is None else np.asarray(sources, dtype=np.int64)
    dst = None if targets is None else np.asarray(targets, dtype=np.int64)
    return _build(pos, sources=src, targets=dst, k=int(k))
