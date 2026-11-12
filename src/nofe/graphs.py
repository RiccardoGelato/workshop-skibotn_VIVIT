"""In-memory graph construction from point positions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor
from torch_geometric.data import Data

DEFAULT_EPS = 1e-8
_FILLS = ("interpolate", "zeros", "mean")
_NEIGHBORS = ("radius", "knn")

_FAISS_WARNED = False


def _as_points(pos: Tensor | np.ndarray, name: str = "pos") -> np.ndarray:
    points = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"{name} must be 2-D [num_nodes, position_dims], got shape {points.shape}")
    return points


def geometry_edge_attr(
    pos: Tensor,
    edge_index: Tensor,
    edge_dist: Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Edge features ``[pos[dst], direction, relative distance]``, shape ``[E, 2P + 1]``."""
    if pos.dim() != 2:
        raise ValueError(
            f"pos must be 2-D [num_nodes, position_dims], got shape {tuple(pos.shape)}"
        )
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
        )
    if edge_index.numel() == 0:
        return pos.new_empty((0, 2 * pos.shape[1] + 1))

    src, dst = edge_index[0], edge_index[1]
    delta = pos[src] - pos[dst]
    norm = torch.linalg.vector_norm(delta, dim=1)
    if edge_dist is None:
        edge_dist = norm
    direction = delta / (norm + eps).unsqueeze(1)
    max_dist = pos.new_full((pos.shape[0],), -torch.inf)
    max_dist.scatter_reduce_(0, dst, edge_dist, reduce="amax", include_self=True)
    relative = (edge_dist / (max_dist[dst] + eps)).unsqueeze(1)
    return torch.cat((pos[dst], direction, relative), dim=1)


def _scipy_index(points: np.ndarray, reason: str) -> cKDTree:
    """scipy fallback index, warning once why faiss is not in use."""
    global _FAISS_WARNED
    if not _FAISS_WARNED:
        warnings.warn(
            f"{reason}; falling back to scipy cKDTree, which is much slower for kNN search "
            "on large point sets. Install nofe's faiss-cpu dependency to silence this warning.",
            RuntimeWarning,
            stacklevel=4,
        )
        _FAISS_WARNED = True
    return cKDTree(points)


def _duplicate_openmp() -> bool:
    """True when two OpenMP runtimes are already mapped into the process.

    faiss wheels bundle their own ``libomp``; when the process also loaded torch's copy, the
    first parallel faiss call aborts with ``OMP: Error #15`` (a hard process exit, not an
    exception), so the index must not be used at all.
    """
    import ctypes
    import os

    try:
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        mapped = {
            os.path.realpath(libc._dyld_get_image_name(index).decode())
            for index in range(libc._dyld_image_count())
            if os.path.basename(libc._dyld_get_image_name(index).decode())
            .lower()
            .startswith(("libomp", "libiomp5", "libgomp"))
        }
    except (AttributeError, OSError, ValueError):  # not macOS: no dyld image table
        return False
    return len(mapped) > 1


_MISSING_FAISS = "faiss-cpu is not installed"
_CONFLICTING_OPENMP = (
    "faiss-cpu is installed but the process maps two OpenMP runtimes, so its parallel "
    "search would abort"
)


def _search_index(points: np.ndarray) -> Any:
    """Build a flat L2 index over ``points``; scipy when faiss is unavailable."""
    try:
        import faiss
    except ImportError:
        return _scipy_index(points, _MISSING_FAISS)
    if _duplicate_openmp():
        return _scipy_index(points, _CONFLICTING_OPENMP)
    try:
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(np.ascontiguousarray(points, dtype=np.float32))
    except Exception:  # faiss imported but unusable: same fallback as a missing install
        return _scipy_index(points, _MISSING_FAISS)
    return index


def _pairs(
    points: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    radius: float | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(neighbor, query)`` global node id pairs, excluding self hits."""
    if sources.size == 0 or targets.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    tree = _search_index(points[sources])
    query_points = np.ascontiguousarray(points[targets], dtype=np.float32)

    if radius is not None:
        neighbor: list[int] = []
        query: list[int] = []
        if isinstance(tree, cKDTree):
            # Sorted per target so both backends emit the same edge order.
            for i, row in enumerate(tree.query_ball_point(query_points, r=radius)):
                target = int(targets[i])
                for j in sorted(row):
                    source = int(sources[j])
                    if source == target:
                        continue
                    neighbor.append(source)
                    query.append(target)
        else:
            # faiss range_search takes a squared radius and returns ascending node ids.
            lims, _, hits = tree.range_search(query_points, radius * radius)
            counts = np.diff(lims).astype(np.int64)
            hit_list = np.asarray(hits, dtype=np.int64)
            neighbor_pairs = sources[hit_list]
            query_pairs = np.repeat(np.asarray(targets, dtype=np.int64), counts)
            keep = neighbor_pairs != query_pairs
            neighbor = neighbor_pairs[keep].tolist()
            query = query_pairs[keep].tolist()
        if not neighbor:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.asarray(neighbor, dtype=np.int64), np.asarray(query, dtype=np.int64)

    assert k is not None
    kk = min(k + 1, len(sources))
    if isinstance(tree, cKDTree):
        # faiss's flat search returns `-1` padding and scipy's `query` does not, so the
        # backends are mapped onto one shape here.
        _, index = tree.query(query_points, k=kk)
    else:
        _, index = tree.search(query_points, kk)
    index = np.asarray(index, dtype=np.int64)
    if index.ndim == 1:
        index = index.reshape(-1, 1) if kk == 1 else index.reshape(1, -1)
    flat = index.reshape(-1)
    valid = flat >= 0  # faiss pads unfilled slots with -1
    query = np.repeat(np.asarray(targets, dtype=np.int64), index.shape[1])[valid]
    neighbor = sources[flat[valid]]
    keep = neighbor != query
    return neighbor[keep].astype(np.int64), query[keep].astype(np.int64)


def _build(
    pos: Tensor | np.ndarray,
    *,
    sources: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    radius: float | None = None,
    k: int | None = None,
) -> tuple[Tensor, Tensor]:
    points = _as_points(pos)
    n = points.shape[0]
    src_ids = np.arange(n) if sources is None else np.asarray(sources, dtype=np.int64)
    dst_ids = np.arange(n) if targets is None else np.asarray(targets, dtype=np.int64)
    neighbor, query = _pairs(points, src_ids, dst_ids, radius=radius, k=k)

    edge_index = torch.empty((2, neighbor.shape[0]), dtype=torch.long)
    if neighbor.shape[0] > 0:
        edge_index[0] = torch.from_numpy(neighbor)
        edge_index[1] = torch.from_numpy(query)
    point_tensor = torch.as_tensor(points, dtype=torch.float32)
    return edge_index, geometry_edge_attr(point_tensor, edge_index)


def radius_graph(
    pos: Tensor | np.ndarray,
    radius: float,
    *,
    sources: Tensor | np.ndarray | None = None,
    targets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Radius neighbourhood graph; senders are ``sources``, receivers ``targets``."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    _as_points(pos)
    src = None if sources is None else np.asarray(sources, dtype=np.int64)
    dst = None if targets is None else np.asarray(targets, dtype=np.int64)
    return _build(pos, sources=src, targets=dst, radius=float(radius))


def knn_graph(
    pos: Tensor | np.ndarray,
    k: int,
    *,
    sources: Tensor | np.ndarray | None = None,
    targets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """k-nearest-neighbour graph; senders are ``sources``, receivers ``targets``."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    _as_points(pos)
    src = None if sources is None else np.asarray(sources, dtype=np.int64)
    dst = None if targets is None else np.asarray(targets, dtype=np.int64)
    return _build(pos, sources=src, targets=dst, k=int(k))


def _as_features(
    features: Tensor | np.ndarray | None,
    num_nodes: int | None,
    name: str = "features",
    container: str = "positions",
) -> np.ndarray | None:
    if features is None:
        return None
    values = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D [num_nodes, features], got shape {values.shape}")
    if num_nodes is not None and values.shape[0] != num_nodes:
        raise ValueError(f"{name} has {values.shape[0]} rows but {container} has {num_nodes}")
    return values


def _query_mask(positions: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: ``True`` for query rows whose exact location is new."""
    known = {row.tobytes() for row in np.ascontiguousarray(positions).view(np.uint32)}
    return np.asarray(
        [row.tobytes() not in known for row in np.ascontiguousarray(queries).view(np.uint32)],
        dtype=bool,
    )


def _fill_rows(
    x: Tensor,
    pos: Tensor,
    observed: np.ndarray,
    added: np.ndarray,
    *,
    fill: str,
    k: int,
) -> None:
    """Write the ``fill`` substitute for the ``added`` rows of ``x``, in place."""
    if added.size == 0 or observed.size == 0:
        return
    if fill == "zeros":
        x[torch.as_tensor(added)] = 0.0
        return
    if fill == "mean":
        x[torch.as_tensor(added)] = x[torch.as_tensor(observed)].mean(dim=0)
        return

    edge_index, _ = knn_graph(pos, k, sources=observed, targets=added)
    if edge_index.numel() == 0:
        return
    neighbor, query = edge_index[0], edge_index[1]
    lookup = np.full(pos.shape[0], -1, dtype=np.int64)
    lookup[added] = np.arange(added.size)
    local = torch.as_tensor(lookup[query.numpy()])
    weights = 1.0 / (torch.linalg.vector_norm(pos[neighbor] - pos[query], dim=1) + DEFAULT_EPS)
    total = torch.zeros(added.size, dtype=x.dtype)
    total.index_add_(0, local, weights)
    accumulated = torch.zeros((added.size, x.shape[1]), dtype=x.dtype)
    accumulated.index_add_(0, local, weights.unsqueeze(1) * x[neighbor])
    x[torch.as_tensor(added)] = accumulated / total.clamp_min(DEFAULT_EPS).unsqueeze(1)


def build_graph(
    positions: Tensor | np.ndarray,
    features: Tensor | np.ndarray | None,
    *,
    query_positions: Tensor | np.ndarray | None = None,
    query_features: Tensor | np.ndarray | None = None,
    feature_dim: int | None = None,
    neighbor: str | None = None,
    radius: float | None = None,
    k: int | None = None,
    fill: str = "interpolate",
) -> Data:
    """Positions and features to a NOFE ``Data``, optionally with added out-of-sample locations.

    ``positions``/``features`` are the observed rows. ``query_positions`` adds locations whose
    features are unknown; they are initialized by ``fill`` and stay ``input_mask=False``, so they
    receive messages and never send — the same graph at training and at inference.
    ``query_features`` is those locations' ground truth and goes to ``y`` for the loss only: it
    never enters ``x`` or the edges.
    """
    neighbor = "radius" if neighbor is None else neighbor
    if neighbor not in _NEIGHBORS:
        raise ValueError(f"neighbor must be one of {_NEIGHBORS}, got {neighbor!r}")
    fill = "interpolate" if fill is None else fill
    if fill not in _FILLS:
        raise ValueError(f"fill must be one of {_FILLS}, got {fill!r}")
    radius = 1.0 if radius is None else float(radius)
    k = 10 if k is None else int(k)

    points = _as_points(positions, name="positions")
    num_inputs = points.shape[0]
    if num_inputs == 0:
        raise ValueError("positions has no rows")

    observed = _as_features(features, num_inputs)
    if observed is None and (feature_dim is None or query_positions is None):
        # No observed features to copy from, so the width and the added rows must be given.
        raise ValueError("feature_dim is required when features is None")
    feature_width = observed.shape[1] if observed is not None else int(feature_dim)

    added_points = np.zeros((0, points.shape[1]), dtype=np.float32)
    added_features: np.ndarray | None = None
    if query_positions is not None:
        queries = _as_points(query_positions, name="query_positions")
        if query_features is not None:
            added_features = _as_features(query_features, None, name="query_features")
            assert added_features is not None
            if added_features.shape[0] != queries.shape[0]:
                raise ValueError(
                    f"query_features has {added_features.shape[0]} rows but query_positions has "
                    f"{queries.shape[0]}"
                )
            feature_width = added_features.shape[1]
        keep = _query_mask(points, queries)
        dropped = int((~keep).sum())
        if dropped:
            warnings.warn(
                f"dropped {dropped} query points already present in positions",
                UserWarning,
                stacklevel=2,
            )
        added_points = queries[keep]
        if added_points.shape[0] == 0:
            raise ValueError("all query_positions were already present in positions")
        if added_features is not None:
            added_features = added_features[keep]

    merged = np.concatenate((points, added_points), axis=0)
    pos_tensor = torch.as_tensor(merged, dtype=torch.float32)
    x = torch.zeros((merged.shape[0], feature_width), dtype=torch.float32)
    if observed is not None:
        x[:num_inputs] = torch.as_tensor(observed, dtype=torch.float32)

    input_ids = np.arange(num_inputs, dtype=np.int64)
    added_ids = np.arange(num_inputs, merged.shape[0], dtype=np.int64)
    if added_ids.size:
        _fill_rows(x, pos_tensor, input_ids, added_ids, fill=fill, k=k)

    every = np.arange(merged.shape[0], dtype=np.int64)
    if neighbor == "knn":
        edge_index, edge_attr = knn_graph(pos_tensor, k, sources=input_ids, targets=every)
    else:
        edge_index, edge_attr = radius_graph(pos_tensor, radius, sources=input_ids, targets=every)

    data = Data(x=x, pos=pos_tensor, edge_index=edge_index, edge_attr=edge_attr)
    if query_positions is not None:
        data.input_mask = torch.as_tensor(
            np.concatenate((np.ones(num_inputs, dtype=bool), np.zeros(added_ids.size, dtype=bool)))
        )
        data.output_mask = torch.ones(merged.shape[0], dtype=torch.bool)
        if added_features is not None:
            truth = torch.zeros((merged.shape[0], feature_width), dtype=torch.float32)
            if observed is not None:
                truth[:num_inputs] = torch.as_tensor(observed, dtype=torch.float32)
            truth[num_inputs:] = torch.as_tensor(added_features, dtype=torch.float32)
            data.y = truth
    return data
