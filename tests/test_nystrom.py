import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from nofe.nystrom import NystromSampler, node_indices


def _data(num_nodes: int = 60) -> Data:
    """Masked graph: observed rows 0..39 send to their 3 nearest, every row receives."""
    torch.manual_seed(0)
    pos = torch.rand(num_nodes, 2) * 8.0
    x = torch.rand(num_nodes, 3)
    senders, receivers = [], []
    for node in range(num_nodes):
        if node >= 40:
            continue
        distance = (pos - pos[node]).norm(dim=1)
        for peer in distance.argsort()[1:4].tolist():
            senders.append(node)
            receivers.append(peer)
    edge_index = torch.tensor([senders, receivers])
    edge_attr = torch.rand(edge_index.shape[1], 2 * pos.shape[1] + 1)
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.input_mask = torch.arange(num_nodes) < 40
    data.output_mask = torch.arange(num_nodes) >= 20
    return data


def test_node_indices_defaults_and_masks() -> None:
    masked = _data()
    assert np.array_equal(node_indices(masked, "input_mask"), np.arange(40))
    assert np.array_equal(node_indices(masked, "output_mask"), np.arange(20, 60))

    unmasked = Data(x=torch.rand(9, 3), pos=torch.rand(9, 2))
    assert np.array_equal(node_indices(unmasked, "input_mask"), np.arange(9))
    assert np.array_equal(node_indices(unmasked, "output_mask"), np.arange(9))


def test_active_graph_structure_inherits_the_supplied_neighbourhood() -> None:
    data = _data()
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    generator = torch.Generator().manual_seed(3)
    queries = sampler.sample_nodes(node_indices(data, "output_mask"), 8, generator)
    landmarks = sampler.sample_nodes(node_indices(data, "input_mask"), 12, generator)
    active = sampler.active_graph(data, queries, landmarks)

    assert torch.equal(active.global_indices[: len(queries)], torch.as_tensor(queries))
    assert active.query_mask.sum() == len(queries)
    assert active.query_mask[: len(queries)].all()
    assert active.x.shape[0] == len(set(queries.tolist()) | set(landmarks.tolist()))
    assert int(active.edge_index.max()) < active.x.shape[0]
    assert (active.edge_index[0] != active.edge_index[1]).all()
    assert active.edge_attr.shape == (active.edge_index.shape[1], 2 * data.pos.shape[1] + 1)

    # Every active edge is sent by a landmark, because the source graph only lets inputs send.
    senders = active.global_indices[active.edge_index[0]].tolist()
    assert set(senders) <= {int(node) for node in landmarks}


def test_active_graph_uses_the_supplied_edges() -> None:
    """The subgraph keeps the caller's edges: the neighbourhood is not re-derived here."""
    data = _data()
    data.output_mask = torch.ones(60, dtype=torch.bool)
    sampler = NystromSampler(transition=0.5, chunk_size=8)

    queries = np.arange(50, 60)
    landmarks = np.arange(0, 10)
    active = sampler.active_graph(data, queries, landmarks)
    # The subgraph keeps exactly the original edges with a landmark sender and an active
    # receiver; queries first, then the landmarks, so the count is comparable directly.
    active_nodes = set(queries.tolist()) | set(landmarks.tolist())
    kept = sum(
        1
        for sender, receiver in zip(*data.edge_index.tolist(), strict=True)
        if sender in set(landmarks.tolist()) and receiver in active_nodes
    )
    assert active.edge_index.shape[1] == kept
    assert kept > 0


def test_generator_determinism_and_capping() -> None:
    data = _data()
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    candidates = node_indices(data, "output_mask")

    def build() -> tuple[np.ndarray, torch.Tensor]:
        generator = torch.Generator().manual_seed(11)
        queries = sampler.sample_nodes(candidates, 5, generator)
        landmarks = sampler.sample_nodes(node_indices(data, "input_mask"), 7, generator)
        active = sampler.active_graph(data, queries, landmarks)
        return active.global_indices.numpy(), active.edge_index

    first_indices, first_edges = build()
    second_indices, second_edges = build()
    assert np.array_equal(first_indices, second_indices)
    assert torch.equal(first_edges, second_edges)

    drawn = sampler.sample_nodes(np.arange(4), 100, torch.Generator().manual_seed(0))
    assert drawn.shape == (4,)
    assert sorted(drawn.tolist()) == [0, 1, 2, 3]


def test_fraction_becomes_a_landmark_count() -> None:
    sampler = NystromSampler(transition=0.25, chunk_size=8)
    assert sampler.transition == 0.25
    assert sampler.landmark_fraction == 0.25
    assert sampler.landmark_count(40) == 10
    # A fraction of a small pool still yields one landmark, capped at the pool size.
    assert sampler.landmark_count(2) == 1
    assert sampler.landmark_count(1000) == 250
    assert NystromSampler(transition=1.0, chunk_size=1).landmark_count(7) == 7
    assert sampler.query_count(13) == 13


def test_invalid_settings_and_empty_candidates_raise() -> None:
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    with pytest.raises(ValueError, match="cannot sample from an empty node set"):
        sampler.sample_nodes(np.zeros(0, dtype=np.int64), 3, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="landmark fraction must be in"):
        NystromSampler(transition=0.0, chunk_size=8)
    with pytest.raises(ValueError, match="landmark fraction must be in"):
        NystromSampler(transition=1.5, chunk_size=8)
    with pytest.raises(ValueError, match="landmark fraction must be in"):
        NystromSampler(transition=True, chunk_size=8)
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        NystromSampler(transition=0.5, chunk_size=0)


def test_mask_membership_is_enforced() -> None:
    data = _data()
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    with pytest.raises(ValueError, match="queries must be selected by output_mask"):
        sampler.active_graph(data, np.array([0, 1]), np.array([20, 21]))
    with pytest.raises(ValueError, match="landmarks must be selected by input_mask"):
        sampler.active_graph(data, np.array([20, 21]), np.array([50, 51]))


def test_active_graph_copies_the_supplied_edge_features() -> None:
    """Edge features are data, not recomputed: the edge kernel was trained on these."""
    data = _data()
    data.edge_dist = torch.arange(data.edge_index.shape[1], dtype=torch.float32)
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    queries = np.arange(50, 60)
    landmarks = np.arange(0, 10)
    active = sampler.active_graph(data, queries, landmarks)

    senders = data.edge_index[0].tolist()
    receivers = data.edge_index[1].tolist()
    active_nodes = set(queries.tolist()) | set(landmarks.tolist())
    kept = [
        index
        for index, (sender, receiver) in enumerate(zip(senders, receivers, strict=True))
        if sender in set(landmarks.tolist()) and receiver in active_nodes
    ]
    assert active.edge_index.shape[1] == len(kept)
    assert torch.equal(active.edge_attr, data.edge_attr[kept].float())
    assert torch.equal(active.edge_dist, data.edge_dist[kept])


def test_graph_without_edges_is_refused() -> None:
    data = _data()
    data.edge_index = None  # a graph without edges cannot back the Nyström path
    sampler = NystromSampler(transition=0.5, chunk_size=8)
    with pytest.raises(ValueError, match="needs data.edge_index"):
        sampler.active_graph(data, np.array([20, 21]), np.array([0, 1]))
