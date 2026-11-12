import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from nofe import build_graph
from nofe.graphs import geometry_edge_attr, knn_graph, radius_graph


def test_knn_graph_has_k_incoming_edges_and_no_self_loops() -> None:
    pos = torch.rand(40, 2) * 10.0
    edge_index, edge_attr = knn_graph(pos, 5)

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
    edge_index, _ = knn_graph(pos, 3, sources=sources, targets=targets)
    assert set(edge_index[0].tolist()) <= set(sources.tolist())
    assert set(edge_index[1].tolist()) <= set(targets.tolist())
    degrees = torch.bincount(edge_index[1], minlength=pos.shape[0])
    # Disjoint sets: no self hit is dropped, so each target keeps min(k + 1, |sources|).
    assert degrees[20:].tolist() == [4] * 10
    assert int(degrees[:20].sum()) == 0

    all_index, _ = knn_graph(pos, 3, sources=np.arange(30), targets=targets)
    all_degrees = torch.bincount(all_index[1], minlength=pos.shape[0])
    assert all_degrees[20:].tolist() == [3] * 10

    radius_index, _ = radius_graph(pos, 2.0, sources=sources, targets=targets)
    assert set(radius_index[0].tolist()) <= set(sources.tolist())
    assert set(radius_index[1].tolist()) <= set(targets.tolist())


def test_radius_graph_respects_radius_and_normalises_distance() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
    edge_index, edge_attr = radius_graph(pos, 1.5)
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
    knn_index, knn_attr = knn_graph(points, 4)
    assert knn_attr.shape == (knn_index.shape[1], 2 * 3 + 1)
    assert (knn_attr[:, -1] <= 1.0 + 1e-6).all()
    assert (knn_attr[:, -1] > 0).all()
    assert torch.allclose(geometry_edge_attr(points, knn_index), knn_attr, atol=1e-6)


def test_empty_results_and_invalid_arguments() -> None:
    pos = torch.rand(5, 2) * 10.0
    edge_index, edge_attr = radius_graph(pos, 1e-6)
    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long
    assert edge_attr.shape == (0, 2 * pos.shape[1] + 1)

    empty_index = torch.empty((2, 0), dtype=torch.long)
    assert geometry_edge_attr(pos, empty_index).shape == (0, 5)

    for call, match in (
        (lambda: radius_graph(pos, 0.0), "radius must be positive"),
        (lambda: radius_graph(pos, -1.0), "radius must be positive"),
        (lambda: knn_graph(pos, 0), "k must be at least 1"),
        (lambda: knn_graph(torch.rand(5), 3), "pos must be 2-D"),
        (lambda: radius_graph(torch.rand(5), 1.0), "pos must be 2-D"),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_scipy_fallback_matches_faiss_and_warns_once(monkeypatch) -> None:
    """The neighbourhood is method-independent: both backends emit the same edges."""
    torch.manual_seed(0)
    pos = torch.rand(200, 2)
    reference_knn = knn_graph(pos, 8)
    reference_radius = radius_graph(pos, 0.5)

    monkeypatch.setitem(sys.modules, "faiss", None)
    monkeypatch.setattr("nofe.graphs._FAISS_WARNED", False)
    with pytest.warns(RuntimeWarning, match="faiss-cpu is not installed"):
        fallback_knn = knn_graph(pos, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback_radius = radius_graph(pos, 0.5)
    assert caught == []

    assert torch.equal(fallback_knn[0], reference_knn[0])
    assert torch.equal(fallback_knn[1], reference_knn[1])
    assert torch.equal(fallback_radius[0], reference_radius[0])
    assert torch.equal(fallback_radius[1], reference_radius[1])


def test_build_graph_without_queries_is_an_in_sample_graph() -> None:
    torch.manual_seed(0)
    pos = torch.rand(120, 2) * 4.0
    features = torch.rand(120, 5)
    data = build_graph(pos, features, neighbor="radius", radius=2.0)

    assert isinstance(data, Data)
    assert data.x.shape == (120, 5)
    assert data.pos.shape == (120, 2)
    assert data.edge_attr.shape == (data.edge_index.shape[1], 2 * 2 + 1)
    assert getattr(data, "input_mask", None) is None
    assert getattr(data, "output_mask", None) is None
    assert getattr(data, "y", None) is None
    assert (data.edge_index[0] != data.edge_index[1]).all()
    assert torch.equal(data.x, features)


def _merged(*, with_truth: bool = True, fill: str = "interpolate") -> Data:
    torch.manual_seed(1)
    pos = torch.rand(120, 2) * 4.0
    features = torch.rand(120, 5)
    queries = torch.rand(30, 2) * 4.0
    truth = torch.rand(30, 5)
    kwargs = {"query_features": truth} if with_truth else {}
    data = build_graph(
        pos,
        features,
        query_positions=queries,
        neighbor="knn",
        k=10,
        fill=fill,
        **kwargs,
    )
    return data


def test_merge_keeps_truth_out_of_the_input_and_scores_every_row() -> None:
    data = _merged()
    num_inputs = 120
    features_truth = data.y[:num_inputs]
    added_truth = data.y[num_inputs:]

    assert data.num_nodes == 150
    assert data.pos.shape == (150, 2)
    assert data.x.shape == (150, 5)
    assert torch.equal(data.x[:num_inputs], features_truth)
    # The added rows hold the substitute, never their own truth.
    assert not torch.equal(data.x[num_inputs:], added_truth)
    assert torch.equal(data.y[:num_inputs], features_truth)
    assert int(data.input_mask.sum()) == num_inputs
    assert not data.input_mask[num_inputs:].any()
    assert data.output_mask.all()
    # Added locations receive messages and never send them.
    assert set(data.edge_index[0].tolist()) == set(range(num_inputs))


def test_merged_graph_is_bit_identical_without_truth() -> None:
    with_truth = _merged()
    without_truth = _merged(with_truth=False)

    assert without_truth.num_nodes == with_truth.num_nodes
    assert torch.equal(without_truth.pos, with_truth.pos)
    assert torch.equal(without_truth.x, with_truth.x)
    assert torch.equal(without_truth.input_mask, with_truth.input_mask)
    assert torch.equal(without_truth.edge_index, with_truth.edge_index)
    assert without_truth.y is None


def test_duplicate_query_positions_are_dropped_with_one_warning() -> None:
    torch.manual_seed(2)
    pos = torch.rand(120, 2) * 4.0
    features = torch.rand(120, 5)
    queries = torch.cat((pos[:3], torch.rand(27, 2) * 4.0))
    truth = torch.rand(30, 5)

    with pytest.warns(UserWarning, match="dropped 3 query points"):
        data = build_graph(
            pos, features, query_positions=queries, query_features=truth, neighbor="knn", k=10
        )

    assert data.num_nodes == 147
    assert torch.equal(data.y[120:], truth[3:])
    added = data.pos[120:]
    for row in added:
        assert not bool((pos == row).all(dim=1).any())

    with pytest.raises(ValueError, match="all query_positions were already present"):
        build_graph(pos, features, query_positions=pos[:5], neighbor="knn", k=10)


def test_no_holdout_knob_exists() -> None:
    pos = torch.rand(20, 2)
    features = torch.rand(20, 3)
    with pytest.raises(TypeError):
        build_graph(pos, features, holdout=0.2)  # type: ignore[call-arg]


def test_fill_modes_and_argument_guards() -> None:
    torch.manual_seed(3)
    pos = torch.rand(60, 2) * 4.0
    features = torch.rand(60, 5)
    queries = torch.rand(10, 2) * 4.0

    zeros = build_graph(pos, features, query_positions=queries, fill="zeros")
    assert (zeros.x[60:] == 0).all()

    mean = build_graph(pos, features, query_positions=queries, fill="mean")
    assert torch.allclose(mean.x[60:], features.mean(dim=0).expand(10, 5))

    interpolated = build_graph(pos, features, query_positions=queries)
    assert torch.isfinite(interpolated.x).all()
    # Inverse-distance interpolation produces working features, not a collapse onto one point.
    assert not torch.allclose(interpolated.x[60:], zeros.x[60:])

    blind = build_graph(pos, None, query_positions=queries, feature_dim=5)
    assert blind.x.shape == (70, 5)
    assert blind.y is None

    with pytest.raises(ValueError, match="feature_dim is required when features is None"):
        build_graph(pos, None, query_positions=queries)
    with pytest.raises(ValueError, match="features must be 2-D"):
        build_graph(pos, features.flatten())
    with pytest.raises(ValueError, match="features has 5 rows but positions has 60"):
        build_graph(pos, features[:5])
    with pytest.raises(ValueError, match="rows but query_positions"):
        build_graph(pos, features, query_positions=queries, query_features=features)
    with pytest.raises(ValueError, match="neighbor must be one of"):
        build_graph(pos, features, neighbor="grid")
    with pytest.raises(ValueError, match="fill must be one of"):
        build_graph(pos, features, fill="median")
    with pytest.raises(ValueError, match="positions has no rows"):
        build_graph(torch.empty((0, 2)), torch.empty((0, 3)))


_FAISS_PROBE = """
import numpy as np
import faiss
from scipy.spatial import cKDTree

source = open("src/nofe/graphs.py").read()
body = source[source.index("def _pairs(") : source.index("def _build(")]
namespace = {"np": np, "cKDTree": cKDTree}


def faiss_index(points):
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(np.ascontiguousarray(points, dtype=np.float32))
    return index


rng = np.random.default_rng(0)
points = rng.random((80, 3)).astype(np.float32)
senders = rng.choice(80, 25, replace=False)
receivers = rng.choice(80, 30, replace=False)
results = {}
for name, factory in (("faiss", faiss_index), ("scipy", lambda p: cKDTree(p))):
    namespace["_search_index"] = factory
    exec(body, namespace)
    pairs = namespace["_pairs"]
    for label, kwargs in (("knn", {"k": 5}), ("radius", {"radius": 0.3})):
        results[(name, label)] = np.stack(pairs(points, senders, receivers, **kwargs))
        assert set(results[(name, label)][0].tolist()) <= set(senders.tolist()), (name, label)
        assert set(results[(name, label)][1].tolist()) <= set(receivers.tolist()), (name, label)
    assert pairs(points, senders[:2], receivers, radius=1e-9)[0].size == 0

for label in ("knn", "radius"):
    assert np.array_equal(results[("faiss", label)], results[("scipy", label)]), label
print("faiss and scipy agree")
"""


def test_faiss_backend_matches_scipy_when_faiss_can_run() -> None:
    """Both search backends must emit identical edges.

    faiss bundles its own ``libomp``, so a process that also imported torch aborts on the first
    parallel search; the faiss branch is therefore exercised in a faiss-only subprocess.
    """
    pytest.importorskip("faiss")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", _FAISS_PROBE], cwd=root, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("faiss and scipy agree")
