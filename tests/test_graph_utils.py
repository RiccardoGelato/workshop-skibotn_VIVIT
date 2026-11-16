"""The two neighbourhood backends must produce identical edge sets.

faiss is the fast path and cKDTree is the macOS default, so any disagreement would mean a macOS
graph and a Linux graph of the same data are different graphs — silently changing what the operator
trains on. These tests pin the equivalence, including the two places where a naive port drifts:

- the radius boundary (faiss excludes a point at exactly ``radius``, cKDTree's own ``<=`` keeps it);
- k-NN ties, where equal distances must resolve by the same index order.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from graph_utils import (  # noqa: E402
    knn_edges,
    radius_edges,
    raster_radius_edges,
    tiled_radius_edges,
)

from nofe.backend import backend  # noqa: E402
from nofe.search import NearestIndex, RadiusIndex  # noqa: E402


def _faiss_usable() -> bool:
    """Installed is not enough on macOS: only a repaired wheel imports without aborting."""
    try:
        return backend("faiss") == "faiss"
    except (ImportError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _faiss_usable(), reason="faiss cannot be used on this platform")


def _points(rows: int, dim: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((rows, dim)).astype(np.float32)


def _both(index_class, points, sources, targets, *args):
    """The same query on both backends, as sorted ``(sender, receiver)`` pairs."""
    left = index_class(points, sources, backend="faiss")
    right = index_class(points, sources, backend="ckdtree")
    method = "within" if index_class is RadiusIndex else "nearest"
    first = getattr(left, method)(targets, *args)
    second = getattr(right, method)(targets, *args)
    order_a = np.lexsort((first[0], first[1]))
    order_b = np.lexsort((second[0], second[1]))
    return (first[0][order_a], first[1][order_a]), (second[0][order_b], second[1][order_b])


def test_radius_pairs_match_on_random_points() -> None:
    points = _points(300, seed=1)
    every = np.arange(300)
    faiss_pairs, tree_pairs = _both(RadiusIndex, points, every, every, 0.25)
    assert faiss_pairs[0].size > 0
    assert np.array_equal(faiss_pairs[0], tree_pairs[0])
    assert np.array_equal(faiss_pairs[1], tree_pairs[1])


def test_radius_boundary_and_self_hits_agree() -> None:
    # Points exactly one unit apart: a radius of 1.0 sits on the boundary of every neighbour.
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    every = np.arange(4)
    faiss_pairs, tree_pairs = _both(RadiusIndex, points, every, every, 1.0)
    assert np.array_equal(faiss_pairs[0], tree_pairs[0])
    assert np.array_equal(faiss_pairs[1], tree_pairs[1])


def test_radius_pairs_match_on_a_raster_band() -> None:
    # The ERA5 banded path: a coarse raster, searched band by band.
    rows, cols = 12, 16
    grid = (
        np.stack(
            np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij"), axis=-1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )
    faiss_edges = raster_radius_edges(grid, 0.15, rows=rows, cols=cols)
    tree_edges = tiled_radius_edges(grid, 0.15, rows=rows, cols=cols)
    assert faiss_edges[0].numel() > 0
    assert torch.equal(faiss_edges[0], tree_edges[0])
    assert torch.equal(faiss_edges[1], tree_edges[1])


def test_knn_pairs_match_including_ties() -> None:
    # A lattice has many exactly-equal distances, which is where tie-breaking shows up.
    steps = np.arange(6, dtype=np.float32)
    lattice = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 2)
    every = np.arange(lattice.shape[0])
    faiss_pairs, tree_pairs = _both(NearestIndex, lattice, every, every, 4)
    assert faiss_pairs[0].size == tree_pairs[0].size == every.size * 4
    assert np.array_equal(faiss_pairs[0], tree_pairs[0])
    assert np.array_equal(faiss_pairs[1], tree_pairs[1])


def test_knn_pairs_match_with_a_restricted_sender_set() -> None:
    points = _points(200, dim=3, seed=2)
    sources = np.arange(0, 200, 3)
    targets = np.arange(1, 200, 2)
    faiss_pairs, tree_pairs = _both(NearestIndex, points, sources, targets, 5)
    assert np.array_equal(faiss_pairs[0], tree_pairs[0])
    assert np.array_equal(faiss_pairs[1], tree_pairs[1])


def test_public_helpers_agree_across_backends(monkeypatch) -> None:
    # The helpers pick their backend from the environment, so the same call can be run both ways.
    points = _points(120, seed=3)
    monkeypatch.setenv("NOFE_BACKEND", "faiss")
    faiss_radius = radius_edges(points, 0.3)
    faiss_knn = knn_edges(points, 5)
    monkeypatch.setenv("NOFE_BACKEND", "ckdtree")
    tree_radius = radius_edges(points, 0.3)
    tree_knn = knn_edges(points, 5)

    assert faiss_radius[0].numel() > 0
    for left, right in zip(faiss_radius, tree_radius, strict=True):
        assert torch.equal(left, right)
    for left, right in zip(faiss_knn, tree_knn, strict=True):
        assert torch.equal(left, right)
