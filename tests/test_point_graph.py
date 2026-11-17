import numpy as np
import pytest
import torch

from nofe import PointGraph  # noqa: E402
from nofe.search import radius_edges


class NnGraph(PointGraph):
    """A radius graph over plain points: the fixture both test modules build graphs with."""

    def __init__(self, radius: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radius = radius

    def edge_parameters(self) -> tuple[float, int]:
        return self.radius, 0

    def neighbour_pairs(self, points: np.ndarray, radius: float, k: int):
        return radius_edges(points, radius)


def _points(num: int, dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((num, dim)).astype(np.float32)


def test_builds_at_any_location_dimension() -> None:
    for dim in (1, 2, 3, 5):
        features = _points(200, 4, seed=dim)
        locations = _points(200, dim, seed=dim + 100)
        data = NnGraph(radius=0.6).build(features, locations)

        assert data.edge_index.shape[1] > 0, dim
        assert tuple(data.x.shape) == (200, 4)
        assert tuple(data.pos.shape) == (200, dim)
        assert tuple(data.pos_raw.shape) == (200, dim)
        assert data.edge_attr.shape[1] == 2 * dim + 1
        assert tuple(data.affinity.shape) == (data.edge_index.shape[1],)
        assert torch.isfinite(data.affinity).all()
        assert float(data.affinity.sum()) > 0.0


def test_normalization_is_per_column_and_keeps_raw_locations() -> None:
    features = _points(120, 3, seed=1) * 5.0 + 7.0
    locations = _points(120, 2, seed=2) * 3.0 - 1.0
    data = NnGraph(radius=0.6).build(features, locations)
    # ``correction=0`` is the population std numpy's ``.std()`` computes, i.e. the one the builder
    # divided by; torch's default is the sample std and would be off by sqrt(n / (n - 1)).
    assert torch.allclose(data.x.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(data.x.std(0, correction=0), torch.ones(3), atol=1e-3)
    assert torch.allclose(data.pos.mean(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(data.pos.std(0, correction=0), torch.ones(2), atol=1e-3)
    assert not torch.equal(data.pos, data.pos_raw)

    raw = NnGraph(radius=0.6, normalize=False).build(features, locations)
    assert torch.equal(raw.x, torch.as_tensor(features))
    assert torch.equal(raw.pos, torch.as_tensor(locations))


def test_sampling_keeps_order_and_is_reproducible() -> None:
    features = _points(60, 3, seed=3)
    locations = _points(60, 2, seed=4)

    full = NnGraph(radius=0.4).build(features, locations)
    assert full.num_nodes == 60

    sampled = NnGraph(radius=0.4, sample=25).build(features, locations)
    assert sampled.num_nodes == 25
    assert bool((sampled.edge_index[1].diff() >= 0).all())
    again = NnGraph(radius=0.4, sample=25).build(features, locations)
    assert torch.equal(again.edge_index, sampled.edge_index)
    assert torch.equal(again.x, sampled.x)

    other = NnGraph(radius=0.4, sample=25, seed=1).build(features, locations)
    assert not torch.equal(other.pos_raw, sampled.pos_raw)


def test_sampling_does_not_refit_the_scaling() -> None:
    features = _points(60, 3, seed=5)
    locations = _points(60, 2, seed=6)
    full = NnGraph(radius=0.4).build(features, locations)
    sampled = NnGraph(radius=0.4, sample=25).build(features, locations)

    # Locate each sampled row in the full graph by its raw location, then compare features.
    lookup = {tuple(row.tolist()): index for index, row in enumerate(full.pos_raw)}
    for row in range(sampled.num_nodes):
        index = lookup[tuple(sampled.pos_raw[row].tolist())]
        assert torch.equal(sampled.x[row], full.x[index])


def test_errors_and_tensor_inputs() -> None:
    features = _points(20, 3, seed=7)
    locations = _points(19, 2, seed=8)
    with pytest.raises(ValueError, match="features has 20 rows"):
        NnGraph(radius=0.4).build(features, locations)
    with pytest.raises(ValueError, match="must be 2-D"):
        NnGraph(radius=0.4).build(features[:, 0], _points(20, 2, seed=9))
    with pytest.raises(ValueError, match="sample"):
        NnGraph(sample=0)

    tensor_pos = NnGraph(radius=0.4).build(
        torch.from_numpy(features), torch.from_numpy(_points(20, 2, seed=10))
    )
    numpy_pos = NnGraph(radius=0.4).build(features, _points(20, 2, seed=10))
    assert torch.equal(tensor_pos.pos, numpy_pos.pos)


def test_edges_are_sorted_and_free_of_self_loops() -> None:
    data = NnGraph(radius=0.4).build(_points(80, 3, seed=11), _points(80, 2, seed=12))
    assert bool((data.edge_index[1].diff() >= 0).all())
    assert bool((data.edge_index[0] != data.edge_index[1]).all())
