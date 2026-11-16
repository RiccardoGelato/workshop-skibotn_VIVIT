"""Neighbour searches: radius and k-nearest, on faiss or ``scipy.spatial.cKDTree``.

Both backends answer the same two questions — which point ids lie within ``radius`` of each query,
and which ``k`` lie nearest — and return **exactly the same pairs**, which is what lets a macOS
checkout and a Linux CI run build identical graphs:

- **Radius is compared as faiss compares it**: squared float32 distances with a strict ``<``.
  cKDTree returns candidates by its own inclusive ``<=`` on float64, so each candidate is re-scored
  in float32 and kept only when ``d² < r²``. That reproduces faiss's boundary rule (a point at
  exactly ``radius`` is excluded) instead of approximating it.
- **k-NN ties resolve by index**: candidates are ranked by a stable ``lexsort`` on
  ``(distance, index)``, the order faiss reports for equal distances.

Neither backend is imported at module import time. The macOS faiss wheel aborts the process on
import (``OMP: Error #15``), so a fallback placed after the import would never run; the import
happens inside the constructor, only once :func:`nofe.backend.backend` has chosen it.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# The function, not the module: `nofe/__init__` rebinds the name `backend` to it on import.
from .backend import backend as select_backend


def _as_points(pos, name: str = "pos") -> np.ndarray:
    """``pos`` as a contiguous float32 2-D array, ``ValueError`` if it is not 2-D."""
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    points = np.asarray(pos, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"{name} must be 2-D [num_nodes, dims], got shape {points.shape}")
    return np.ascontiguousarray(points)


def _subset(points: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Contiguous float32 copy of ``points[sources]``: the rows a backend indexes."""
    return np.ascontiguousarray(points[sources])


class _Index:
    """Shared construction: the backend choice and the exact subset both index types search."""

    def __init__(self, points, sources, *, backend: str | None = None) -> None:
        self.points = _as_points(points)
        self.sources = np.asarray(sources, dtype=np.int64)
        self.backend = select_backend(backend)
        subset = _subset(self.points, self.sources)
        self._index = None
        self._tree = None
        if self.backend == "faiss":
            import faiss  # noqa: PLC0415  (only imported when this backend was chosen)

            self._index = faiss.IndexFlatL2(subset.shape[1])
            self._index.add(subset)
        else:
            # cKDTree wants float64 for its own arithmetic; the rescoring below returns to float32.
            self._tree = cKDTree(subset.astype(np.float64, copy=False))


class RadiusIndex(_Index):
    """Exact L2 radius search over ``points[sources]``; hits come back as ``sources`` ids."""

    def within(self, targets, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """``(sender, receiver)`` pairs strictly inside ``radius``, self hits excluded, unsorted."""
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        targets = np.asarray(targets, dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64)
        if targets.size == 0 or self.sources.size == 0:
            return empty, empty.copy()
        queries = np.ascontiguousarray(self.points[targets])

        if self.backend == "faiss":
            # faiss reports squared distances, so the search radius is squared; it already
            # applies the strict ``<`` this function documents.
            limits, _, found = self._index.range_search(queries, float(radius) ** 2)
            counts = np.diff(limits.astype(np.int64))
            receiver = np.repeat(targets, counts)
            sender = self.sources[found]
        else:
            hits = self._tree.query_ball_point(
                queries.astype(np.float64, copy=False), float(radius)
            )
            counts = np.fromiter((len(hit) for hit in hits), dtype=np.int64, count=len(hits))
            receiver = np.repeat(targets, counts)
            flat = np.fromiter(
                (index for hit in hits for index in hit), dtype=np.int64, count=int(counts.sum())
            )
            sender = self.sources[flat]
            # Re-score in float32 against the squared radius: cKDTree's float64 ``<=`` would keep
            # points sitting exactly on the boundary, which faiss drops.
            delta = self.points[sender] - self.points[receiver]
            squared = np.einsum("ij,ij->i", delta, delta)
            keep = squared < np.float32(float(radius) ** 2)
            sender, receiver = sender[keep], receiver[keep]

        keep = sender != receiver
        return sender[keep], receiver[keep]


class NearestIndex(_Index):
    """Exact L2 k-nearest search over ``points[sources]``; hits come back as ``sources`` ids."""

    def _query(self, queries: np.ndarray, kk: int):
        """Raw ``(distances, local row ids)`` from the configured backend, shape ``(Q, kk)``."""
        rows = len(queries)
        if self.backend == "faiss":
            distances, found = self._index.search(queries, kk)
        else:
            distances, found = self._tree.query(queries.astype(np.float64, copy=False), kk)
        distances = np.asarray(distances, dtype=np.float64).reshape(rows, kk)
        return distances, found.reshape(rows, kk)

    def nearest(self, targets, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``(sender, receiver)`` for each target's ``k`` nearest rows, self hits excluded.

        Both backends are asked for ``kk = min(k + 1, n)`` candidates — one extra, so dropping the
        self hit never leaves a target short — and the ranking is then done here, on float32 squared
        distances, instead of trusting either library's order. Ties are the reason: faiss and
        cKDTree break an equal-distance tie differently, so a lattice (many exactly equal distances)
        would otherwise yield different edges on macOS and Linux.

        A tie *at* the cut-off makes the candidate set itself ambiguous, so ``kk`` doubles until no
        tie straddles it. At ``kk == n`` the set is the whole source array, which settles it.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        targets = np.asarray(targets, dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64)
        if targets.size == 0 or self.sources.size == 0:
            return empty, empty.copy()
        queries = np.ascontiguousarray(self.points[targets])
        total = self.sources.size
        kk = min(k + 1, total)

        while True:
            _, found = self._query(queries, kk)
            sender = self.sources[found]
            receiver = np.repeat(targets, kk)
            squared = self._squared_between(receiver, sender.reshape(-1)).reshape(len(targets), kk)
            if kk >= total or not self._tie_at_cutoff(squared):
                break
            kk = min(total, kk * 2)

        # Drop self hits before ranking: they are the nearest candidate by definition and would
        # otherwise occupy one of each target's ``k`` slots.
        keep = sender.reshape(-1) != receiver.reshape(-1)
        sender = sender.reshape(-1)[keep]
        receiver = receiver.reshape(-1)[keep]
        squared = squared.reshape(-1)[keep]
        # Rank each receiver's candidates by ``(squared distance, sender id)``: fully determined by
        # the data, so both backends produce the same order once they see the same candidate set.
        order = np.lexsort((sender, squared, receiver))
        sender, receiver = sender[order], receiver[order]
        rank = np.arange(sender.size) - np.searchsorted(receiver, receiver, side="left")
        return sender[rank < k], receiver[rank < k]

    def _squared_between(self, receiver: np.ndarray, sender: np.ndarray) -> np.ndarray:
        """float32 squared distances, accumulated per candidate in bounded chunks."""
        out = np.empty(receiver.size, dtype=np.float32)
        for start in range(0, receiver.size, 1_048_576):
            stop = min(start + 1_048_576, receiver.size)
            delta = self.points[sender[start:stop]] - self.points[receiver[start:stop]]
            out[start:stop] = np.einsum("ij,ij->i", delta, delta)
        return out

    @staticmethod
    def _tie_at_cutoff(squared: np.ndarray) -> bool:
        """Whether any row's last distance equals its second-to-last."""
        if squared.shape[1] < 2:
            return False
        return bool((squared[:, -1] == squared[:, -2]).any())


def sorted_pairs(sender: np.ndarray, receiver: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sender, receiver)`` sorted by ``(receiver, sender)``: deterministic edge order."""
    order = np.lexsort((sender, receiver))
    return sender[order].astype(np.int64), receiver[order].astype(np.int64)


__all__ = ["NearestIndex", "RadiusIndex", "sorted_pairs"]
