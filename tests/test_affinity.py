import pytest
import torch

from nofe.affinity import affinities, affinity_kl_loss, per_query_kl


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor([[0.0], [1.0], [2.5], [0.5]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return target, edge_index


def test_affinities_sum_to_one_per_query() -> None:
    values, edge_index = _graph()
    weights, connected = affinities(values, edge_index)
    assert weights.shape == (edge_index.shape[1],)
    assert connected.tolist() == [True, True, True, True]
    for node in range(values.shape[0]):
        total = weights[edge_index[1] == node].sum()
        assert torch.allclose(total, torch.tensor(1.0), atol=1e-5)

    # A node with no incoming edge is unconnected and carries no weight entry.
    sparse_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sparse_weights, sparse_connected = affinities(values, sparse_index)
    assert sparse_connected.tolist() == [False, True, True, False]
    assert sparse_weights.shape == (2,)


def test_identical_distributions_have_zero_kl() -> None:
    values, edge_index = _graph()
    assert float(affinity_kl_loss(values, values, edge_index)) == pytest.approx(0.0, abs=1e-6)
    per_node, valid = per_query_kl(values, values, edge_index)
    assert valid.all()
    assert per_node.abs().max() < 1e-6


def test_query_mask_equals_filtered_edges() -> None:
    target = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float32)
    prediction = torch.tensor([[0.1], [0.9], [2.0], [5.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)
    query_mask = torch.tensor([True, False, True, False])

    masked = affinity_kl_loss(target, prediction, edge_index, query_mask=query_mask)
    keep = query_mask[edge_index[1]]
    filtered = affinity_kl_loss(target, prediction, edge_index[:, keep])
    assert torch.allclose(masked, filtered, atol=1e-6)
    assert masked > 0.0

    per_node, valid = per_query_kl(target, prediction, edge_index, query_mask=query_mask)
    assert valid.tolist() == [True, False, True, False]
    assert per_node[1] == 0.0 and per_node[3] == 0.0
    assert torch.allclose(per_node[valid].mean(), masked, atol=1e-6)


def test_empty_edges_give_zero_loss_with_gradient() -> None:
    target = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    prediction = torch.zeros(2, 2, requires_grad=True)
    empty = torch.empty((2, 0), dtype=torch.long)

    weights, connected = affinities(target, empty)
    assert weights.shape == (0,)
    assert not connected.any()

    loss = affinity_kl_loss(target, prediction, empty)
    assert loss.detach().item() == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() == 0.0

    values, valid = per_query_kl(target, prediction, empty)
    assert values.shape == (2,)
    assert not valid.any()
    assert values.abs().sum() == 0.0


def test_precomputed_weights_match_the_target_path() -> None:
    values, edge_index = _graph()
    prediction = torch.tensor([[0.2], [1.4], [2.1], [0.9]], dtype=torch.float32)
    weights = affinities(values, edge_index)[0]

    recomputed = affinity_kl_loss(values, prediction, edge_index)
    supplied = affinity_kl_loss(values, prediction, edge_index, weights=weights)
    targetless = affinity_kl_loss(None, prediction, edge_index, weights=weights)
    assert torch.allclose(supplied, recomputed, atol=1e-7)
    assert torch.allclose(targetless, recomputed, atol=1e-7)

    with pytest.raises(ValueError, match="target is required"):
        affinity_kl_loss(None, prediction, edge_index)
