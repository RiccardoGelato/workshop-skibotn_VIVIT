import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from graph_utils import (  # noqa: E402
    _pairs,
    geometry_edge_attr,
    knn_edges,
    merge_queries,
    radius_edges,
    tiled_radius_edges,
)


def test_block_merge_orders_known_before_queried_and_masks_them() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = merge_queries(pos, feats, torch.tensor([[0.2, 0.1], [9.9, 9.9]]))

    assert data.num_nodes == 5
    assert data.input_mask.tolist() == [True, True, True, False, False]
    assert data.output_mask.all()
    assert data.y is None
    assert set(data.edge_index[0].tolist()) <= {0, 1, 2}
    assert torch.equal(data.x[:3], feats)
    assert torch.isfinite(data.x[3:]).all()
    assert not torch.equal(data.x[3:], torch.zeros(2, 2))


def test_block_merge_keeps_truth_out_of_the_substitute() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    truth = torch.tensor([[7.0, 7.0], [9.0, 9.0]])
    data = merge_queries(pos, feats, torch.tensor([[0.2, 0.1], [9.9, 9.9]]), truth)

    assert data.y.shape == (5, 2)
    assert torch.equal(data.y[:3], feats)
    assert torch.equal(data.y[3:], truth)
    assert not torch.equal(data.x[3:], data.y[3:])
    assert data.observed_index.tolist() == [0, 1, 2]
    assert data.withheld_index.tolist() == [3, 4]


def test_block_merge_round_trips_caller_ids() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = merge_queries(
        pos,
        feats,
        torch.tensor([[0.2, 0.1], [9.9, 9.9]]),
        ids=torch.tensor([5, 9, 12]),
        query_ids=torch.tensor([40, 41]),
    )
    assert data.observed_index.tolist() == [5, 9, 12]
    assert data.withheld_index.tolist() == [40, 41]


def test_coincident_query_locations_are_dropped() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    queries = torch.tensor([[0.0, 0.0], [0.2, 0.1]])
    truth = torch.tensor([[5.0, 5.0], [7.0, 7.0]])

    with pytest.warns(UserWarning, match="dropped 1 query points") as recorded:
        data = merge_queries(pos, feats, queries, truth)
    assert len(recorded) == 1
    assert data.num_nodes == 4
    # The surviving query row keeps its own truth, so the drop mask stays aligned with ``y``.
    assert torch.equal(data.y[3], truth[1])
    assert data.withheld_index.tolist() == [3]

    with pytest.raises(ValueError, match="all query_pos were already present"):
        merge_queries(pos, feats, torch.tensor([[0.0, 0.0], [10.0, 10.0]]))


def test_empty_query_block_is_the_in_sample_graph() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    empty = merge_queries(pos, feats, torch.empty((0, 2)))
    in_sample = merge_queries(pos, feats)

    for name in ("input_mask", "output_mask", "y", "observed_index", "withheld_index"):
        assert getattr(empty, name, None) is None
    assert torch.equal(empty.x, in_sample.x)
    assert torch.equal(empty.edge_index, in_sample.edge_index)
    assert torch.equal(empty.edge_attr, in_sample.edge_attr)


def test_block_merge_rejects_mismatched_blocks() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    queries = torch.tensor([[0.2, 0.1], [9.9, 9.9]])

    for call, match in (
        (lambda: merge_queries(pos, feats, query_feats=feats), "query_feats requires query_pos"),
        (lambda: merge_queries(pos, feats, queries, feats), "rows but query_pos has"),
        (
            lambda: merge_queries(pos, feats, queries, torch.zeros(2, 3)),
            "columns but feats has",
        ),
        (lambda: merge_queries(pos, feats, queries, ids=torch.tensor([1, 2])), "ids must have 3"),
        (
            lambda: merge_queries(pos, feats, queries, query_ids=torch.tensor([4])),
            "query_ids must have 2",
        ),
        (lambda: merge_queries(pos, feats, torch.zeros(2, 3)), "columns but pos has"),
        (lambda: merge_queries(torch.empty((0, 2)), torch.empty((0, 2))), "pos has no rows"),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_radius_neighbourhood_leaves_a_far_query_row_without_edges() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = merge_queries(
        pos,
        feats,
        torch.tensor([[500.0, 500.0]]),
        edge_builder=lambda p, s, t: radius_edges(p, 1.0, sources=s, targets=t),
    )
    degrees = torch.bincount(data.edge_index[1], minlength=4)
    assert int(degrees[3]) == 0
    assert data.x[3].abs().sum() > 0  # the fill substitute is what the model sees


def test_knn_edges_have_k_incoming_edges_and_no_self_loops() -> None:
    pos = torch.rand(40, 2) * 10.0
    edge_index, edge_attr = knn_edges(pos, 5)

    assert edge_index.dtype == torch.long
    assert edge_index.shape == (2, 40 * 5)
    assert edge_attr.shape == (edge_index.shape[1], 2 * pos.shape[1] + 1)
    assert (edge_index[0] != edge_index[1]).all()
    degrees = torch.bincount(edge_index[1], minlength=pos.shape[0])
    assert degrees.tolist() == [5] * pos.shape[0]


def test_sources_and_targets_restrict_edges() -> None:
    pos = torch.rand(30, 2) * 10.0
    sources = np.arange(10)
    targets = np.arange(20, 30)
    edge_index, _ = knn_edges(pos, 3, sources=sources, targets=targets)
    assert set(edge_index[0].tolist()) <= set(sources.tolist())
    assert set(edge_index[1].tolist()) <= set(targets.tolist())
    degrees = torch.bincount(edge_index[1], minlength=pos.shape[0])
    # Disjoint sets: no self hit is dropped, so each target keeps min(k + 1, |sources|).
    assert degrees[20:].tolist() == [4] * 10
    assert int(degrees[:20].sum()) == 0

    all_index, _ = knn_edges(pos, 3, sources=np.arange(30), targets=targets)
    all_degrees = torch.bincount(all_index[1], minlength=pos.shape[0])
    assert all_degrees[20:].tolist() == [3] * 10

    radius_index, _ = radius_edges(pos, 2.0, sources=sources, targets=targets)
    assert set(radius_index[0].tolist()) <= set(sources.tolist())
    assert set(radius_index[1].tolist()) <= set(targets.tolist())


def test_radius_edges_respect_radius_and_normalise_distance() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
    edge_index, edge_attr = radius_edges(pos, 1.5)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert edge_attr.shape == (4, 5)

    # Relative-distance column: every edge is at most its query's farthest neighbour.
    relative = edge_attr[:, -1]
    assert (relative <= 1.0 + 1e-6).all()
    for node in range(pos.shape[0]):
        rows = edge_index[1] == node
        assert torch.allclose(relative[rows].max(), torch.tensor(1.0), atol=1e-6)

    points = torch.rand(20, 3) * 5.0
    knn_index, knn_attr = knn_edges(points, 4)
    assert knn_attr.shape == (knn_index.shape[1], 2 * 3 + 1)
    assert (knn_attr[:, -1] <= 1.0 + 1e-6).all()
    assert (knn_attr[:, -1] > 0).all()
    assert torch.allclose(geometry_edge_attr(points, knn_index), knn_attr, atol=1e-6)


def test_empty_results_and_invalid_arguments() -> None:
    pos = torch.rand(5, 2) * 10.0
    edge_index, edge_attr = radius_edges(pos, 1e-6)
    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long
    assert edge_attr.shape == (0, 2 * pos.shape[1] + 1)

    empty_index = torch.empty((2, 0), dtype=torch.long)
    assert geometry_edge_attr(pos, empty_index).shape == (0, 5)

    for call, match in (
        (lambda: radius_edges(pos, 0.0), "radius must be positive"),
        (lambda: radius_edges(pos, -1.0), "radius must be positive"),
        (lambda: knn_edges(pos, 0), "k must be at least 1"),
        (lambda: knn_edges(torch.rand(5), 3), "pos must be 2-D"),
        (lambda: radius_edges(torch.rand(5), 1.0), "pos must be 2-D"),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_tiled_radius_edges_match_the_untiled_builder() -> None:
    """A raster band search must reproduce the full search exactly, ordering included."""
    rows, cols = 24, 40
    lat = np.linspace(-60.0, 60.0, rows)
    lon = np.linspace(-180.0, 180.0, cols, endpoint=False)
    phi = np.deg2rad(np.repeat(lat, cols))
    lam = np.deg2rad(np.tile(lon, rows))
    pos = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=1)
    pos = pos.astype(np.float32)

    tiled = tiled_radius_edges(pos, 0.3, rows=rows, cols=cols)
    untiled = _pairs(pos, np.arange(rows * cols), np.arange(rows * cols), radius=0.3)
    assert np.array_equal(tiled[0].numpy(), untiled[0])
    assert np.array_equal(tiled[1].numpy(), untiled[1])
    assert tiled[0].dtype == torch.long and tiled[1].dtype == torch.long
