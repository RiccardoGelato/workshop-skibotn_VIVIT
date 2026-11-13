"""Graph helpers shared by the examples and their tests.

``nofe`` consumes a precomputed graph, so neighbourhood construction lives here: a FAISS exact L2
index for both builders, ``[pos[dst], direction, relative distance]`` edge features, and the
location-union merge that turns known locations plus queried locations into one graph.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import faiss
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

DEFAULT_EPS = 1e-8
FILLS = ("interpolate", "zeros", "mean")


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


def _flat_index(points: np.ndarray, sources: np.ndarray) -> faiss.IndexFlatL2:
    """Exact L2 index over ``points[sources]``; row ``i`` of the index is ``sources[i]``."""
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(np.ascontiguousarray(points[sources], dtype=np.float32))
    return index


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

    Returns ``(neighbor, query)`` int64 tensors sorted by ``(query, neighbor)``, the same order and
    dtypes :func:`radius_edges` returns.
    """
    values = _as_points(points)
    if rows * cols != values.shape[0]:
        raise ValueError(
            f"rows * cols = {rows * cols} must match the {values.shape[0]} points of a "
            f"{rows} x {cols} raster"
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    neighbor_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for start in range(0, rows, band):
        stop = min(start + band, rows)
        lo = max(0, start - halo)
        hi = min(rows, stop + halo)
        index = _flat_index(values, np.arange(lo * cols, hi * cols))
        query_points = np.ascontiguousarray(values[start * cols : stop * cols], dtype=np.float32)
        # L2 search reports squared distances, so the search radius is squared.
        limits, _, found = index.range_search(query_points, float(radius) ** 2)
        counts = np.diff(limits.astype(np.int64))
        recv = np.repeat(np.arange(start * cols, stop * cols, dtype=np.int64), counts)
        # The index holds rows ``lo * cols`` onwards, so shift the hits back to global ids.
        src = found.astype(np.int64) + lo * cols
        keep = src != recv
        neighbor_parts.append(src[keep])
        query_parts.append(recv[keep])

    neighbor = np.concatenate(neighbor_parts)
    query = np.concatenate(query_parts)
    if neighbor.size == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty.clone()
    # Sorted by (target, source) across all bands, so the banding leaves no trace in the output.
    order = np.lexsort((neighbor, query))
    return torch.from_numpy(neighbor[order]), torch.from_numpy(query[order])


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

    index = _flat_index(points, sources)
    query_points = np.ascontiguousarray(points[targets], dtype=np.float32)

    if radius is not None:
        # L2 search reports squared distances, so the search radius is squared.
        limits, _, found = index.range_search(query_points, float(radius) ** 2)
        # ``limits`` is uint64 and ``np.repeat`` refuses it against an int64 index otherwise.
        counts = np.diff(limits.astype(np.int64))
        query = np.repeat(np.asarray(targets, dtype=np.int64), counts)
        # ``found`` holds index rows, whose ids are ``sources``; map them back to node ids.
        neighbor = np.asarray(sources, dtype=np.int64)[found]
        keep = neighbor != query
        query, neighbor = query[keep], neighbor[keep]
        if neighbor.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        # Sorted by (target, source): deterministic edge order, so graphs are reproducible.
        # ``lexsort``'s primary key is the last argument, matching ``sorted(zip(query, neighbor))``.
        order = np.lexsort((neighbor, query))
        return neighbor[order].astype(np.int64), query[order].astype(np.int64)

    assert k is not None
    kk = min(k + 1, len(sources))
    found = index.search(query_points, kk)[1]
    if found.ndim == 1:
        found = found.reshape(-1, 1) if kk == 1 else found.reshape(1, -1)
    # ``kk <= index.ntotal`` by construction, so FAISS never pads a row with its -1 sentinel.
    query = np.repeat(np.asarray(targets, dtype=np.int64), found.shape[1])
    neighbor = sources[found.reshape(-1)]
    keep = neighbor != query
    return neighbor[keep].astype(np.int64), query[keep].astype(np.int64)


def knn_indices(points: Tensor | np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` nearest other rows, shape ``[num_rows, k]``, exact L2.

    Self hits are dropped: each row gets ``k`` neighbours, never itself.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    values = _as_points(points)
    num_rows = values.shape[0]
    if num_rows <= k:
        raise ValueError(f"knn_indices needs at least {k + 1} rows, got {num_rows}")
    index = _flat_index(values, np.arange(num_rows, dtype=np.int64))
    found = index.search(np.ascontiguousarray(values), k + 1)[1]
    # ``num_rows > k`` keeps kk = k + 1 inside the index, so FAISS never pads with -1 here.
    keep = found != np.arange(num_rows, dtype=np.int64)[:, None]
    # Stable-sorting the survivors leftmost (their FAISS order, nearest first) and slicing k
    # columns drops the self hit without a Python loop.
    order = np.argsort(~keep, axis=1, kind="stable")
    return np.take_along_axis(found, order, axis=1)[:, :k].astype(np.int64)


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


def radius_edges(
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


def raster_radius_edges(
    pos: Tensor | np.ndarray,
    radius: float,
    *,
    rows: int,
    cols: int,
    sources: Tensor | np.ndarray | None = None,
    targets: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """``(neighbor, query)`` radius pairs for a full raster, via :func:`tiled_radius_edges`.

    The banded search is exact only when every row both sends and receives, so anything narrower
    falls back to the untiled ``radius_edges``: a restricted sender/receiver set is the merged
    workflow, whose query set is small enough for the full search. Both paths return the same
    ``(neighbor, query)`` int64 tensors.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    points = _as_points(pos)
    every_row = np.arange(points.shape[0], dtype=np.int64)

    def _is_every_row(ids: Tensor | np.ndarray | None) -> bool:
        return ids is None or np.array_equal(np.asarray(ids, dtype=np.int64), every_row)

    if _is_every_row(sources) and _is_every_row(targets):
        return tiled_radius_edges(pos, radius, rows=rows, cols=cols)
    neighbor, query = _pairs(
        points,
        every_row if sources is None else np.asarray(sources, dtype=np.int64),
        every_row if targets is None else np.asarray(targets, dtype=np.int64),
        radius=float(radius),
    )
    return torch.from_numpy(neighbor), torch.from_numpy(query)


def knn_edges(
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
    feats: Tensor | np.ndarray | None,
    num_nodes: int | None,
    name: str = "features",
) -> np.ndarray | None:
    if feats is None:
        return None
    values = feats.detach().cpu().numpy() if isinstance(feats, torch.Tensor) else feats
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D [num_nodes, features], got shape {values.shape}")
    if num_nodes is not None and values.shape[0] != num_nodes:
        raise ValueError(f"{name} has {values.shape[0]} rows but pos has {num_nodes}")
    return values


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

    edge_index, _ = knn_edges(pos, k, sources=observed, targets=added)
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


def _new_locations(known: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: ``True`` for query rows whose exact location is not already known."""
    seen = {row.tobytes() for row in np.ascontiguousarray(known).view(np.uint32)}
    return np.asarray(
        [row.tobytes() not in seen for row in np.ascontiguousarray(queries).view(np.uint32)],
        dtype=bool,
    )


def _graph(pos: Tensor, x: Tensor, edges: tuple[Tensor, Tensor]) -> Data:
    """Assemble the ``Data`` every merge path returns, ``edge_dist`` included."""
    edge_index, edge_attr = edges
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_dist = torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    return data


def merge_queries(
    pos: Tensor | np.ndarray,
    feats: Tensor | np.ndarray,
    query_pos: Tensor | np.ndarray | None = None,
    query_feats: Tensor | np.ndarray | None = None,
    *,
    ids: Tensor | np.ndarray | None = None,
    query_ids: Tensor | np.ndarray | None = None,
    fill: str = "interpolate",
    k: int = 10,
    edge_builder: Callable[[Tensor, np.ndarray, np.ndarray], tuple[Tensor, Tensor]] | None = None,
) -> Data:
    """Location-union merge: known locations first, then the queried locations, in one ``Data``.

    ``pos`` and ``feats`` are the **known** rows: their input vectors are known, so they send
    messages. ``query_pos`` are the locations to query, in their own array and anywhere in space,
    including outside the known rows' bounding box; their truth is optional (``query_feats``) and
    only ever reaches ``y``. Queried rows keep a ``fill`` substitute in ``x``, receive messages and
    never send them, so one graph serves training and inference.

    ``query_pos=None`` withholds nothing: no ``input_mask``/``output_mask``/``y`` and no index
    arrays, just the known rows' graph. A query block that repeats a known location is dropped with
    a ``UserWarning`` (the float32 coordinates are compared exactly).

    ``edge_builder(pos, sources, targets)`` returns the edges to use; it defaults to
    ``knn_edges(..., k)`` and the ERA5 generator passes its radius builder instead, so
    known-only senders hold for either neighbourhood.

    ``data.observed_index``/``data.withheld_index`` are the caller's row ids for each block: the
    ``ids``/``query_ids`` it passed, else the concatenated input order.

    Caveat: a fixed-radius neighbourhood (the ERA5 path) leaves a queried point with no known row
    inside its radius, and such a row's embedding is only the projection bias. Check it with
    ``torch.bincount(data.edge_index[1], minlength=data.num_nodes) == 0`` (``Era5Graph.report``
    prints the count) before trusting those rows. The default k-NN builder always attaches ``k``
    known senders to every row, however far away.
    """
    points = _as_points(pos)
    num_known = points.shape[0]
    if num_known == 0:
        raise ValueError("pos has no rows")
    values = _as_features(feats, num_known)
    if values is None:
        raise ValueError("features is required: the observed rows need their input vectors")
    fill = "interpolate" if fill is None else fill
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
    build_edges = knn_edges_k(k) if edge_builder is None else edge_builder

    queries = None if query_pos is None else _as_points(query_pos, name="query_pos")
    if queries is not None and queries.shape[0] == 0:
        queries = None
    if queries is not None and queries.shape[1] != points.shape[1]:
        raise ValueError(f"query_pos has {queries.shape[1]} columns but pos has {points.shape[1]}")

    if queries is None:
        if query_feats is not None:
            raise ValueError("query_feats requires query_pos")
        pos_tensor = torch.as_tensor(points, dtype=torch.float32)
        x = torch.zeros((num_known, values.shape[1]), dtype=torch.float32)
        x[:] = torch.as_tensor(values, dtype=torch.float32)
        rebuild = np.arange(num_known, dtype=np.int64)
        return _graph(pos_tensor, x, build_edges(pos_tensor, rebuild, rebuild))

    num_queries = queries.shape[0]
    truth = None if query_feats is None else _as_features(query_feats, None, name="query_feats")
    if truth is not None:
        if truth.shape[0] != num_queries:
            raise ValueError(
                f"query_feats has {truth.shape[0]} rows but query_pos has {num_queries}"
            )
        if truth.shape[1] != values.shape[1]:
            raise ValueError(
                f"query_feats has {truth.shape[1]} columns but feats has {values.shape[1]}"
            )

    keep = _new_locations(points, queries)
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"dropped {dropped} query points already present in positions",
            UserWarning,
            stacklevel=2,
        )
    queries = queries[keep]
    if queries.shape[0] == 0:
        raise ValueError("all query_pos were already present in pos")
    if truth is not None:
        truth = truth[keep]

    pos_tensor = torch.as_tensor(np.concatenate((points, queries)), dtype=torch.float32)
    x = torch.zeros((pos_tensor.shape[0], values.shape[1]), dtype=torch.float32)
    x[:num_known] = torch.as_tensor(values, dtype=torch.float32)
    known_ids = np.arange(num_known, dtype=np.int64)
    query_row_ids = np.arange(num_known, pos_tensor.shape[0], dtype=np.int64)
    _fill_rows(x, pos_tensor, known_ids, query_row_ids, fill=fill, k=k)
    data = _graph(pos_tensor, x, build_edges(pos_tensor, known_ids, np.arange(pos_tensor.shape[0])))

    observed_index = (
        torch.arange(num_known, dtype=torch.long)
        if ids is None
        else torch.as_tensor(ids, dtype=torch.long)
    )
    if observed_index.numel() != num_known:
        raise ValueError(f"ids must have {num_known} entries, got {observed_index.numel()}")
    withheld_index = (
        torch.as_tensor(query_row_ids, dtype=torch.long)
        if query_ids is None
        else torch.as_tensor(query_ids, dtype=torch.long)
    )
    if withheld_index.numel() != queries.shape[0]:
        raise ValueError(
            f"query_ids must have {queries.shape[0]} entries, got {withheld_index.numel()}"
        )

    data.input_mask = torch.arange(pos_tensor.shape[0]) < num_known
    data.output_mask = torch.ones(pos_tensor.shape[0], dtype=torch.bool)
    data.observed_index = observed_index
    data.withheld_index = withheld_index
    if truth is not None:
        data.y = torch.cat((x[:num_known], torch.as_tensor(truth, dtype=torch.float32)))
    return data


def knn_edges_k(k: int) -> Callable[[Tensor, np.ndarray, np.ndarray], tuple[Tensor, Tensor]]:
    """Bind ``k`` into a ``merge_queries`` edge builder."""

    def build(pos: Tensor, sources: np.ndarray, targets: np.ndarray) -> tuple[Tensor, Tensor]:
        return knn_edges(pos, k, sources=sources, targets=targets)

    return build
