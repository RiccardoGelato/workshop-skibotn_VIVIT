import copy

import torch
from torch import nn
from torch_geometric.nn import NNConv

from nofe.model import EdgeConv, KernelNN


def _ring_edges(num_nodes: int, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic neighbour ring: node i receives from i+1 .. i+k."""
    rows = torch.arange(num_nodes).repeat_interleave(k)
    cols = (rows + torch.arange(1, k + 1).repeat(num_nodes)) % num_nodes
    edge_index = torch.stack([cols, rows])
    return edge_index, torch.rand(edge_index.shape[1], 2 * 2 + 1)


def _fixture():
    torch.manual_seed(0)
    edge_index, edge_attr = _ring_edges(40, 5)
    # Node 3 loses all incoming edges: an isolated receiver must not break parity.
    keep = edge_index[1] != 3
    edge_index = edge_index[:, keep]
    edge_attr = edge_attr[keep]
    x = torch.rand(41, 4)
    kernel = nn.Sequential(nn.Linear(edge_attr.shape[1], 8), nn.ReLU(), nn.Linear(8, 16))
    return x, edge_index, edge_attr, kernel


def test_chunked_message_matches_stock_conv() -> None:
    x, edge_index, edge_attr, kernel = _fixture()
    stock = NNConv(4, 4, kernel, aggr="mean")
    chunked = EdgeConv(4, copy.deepcopy(kernel), edge_batch_size=1)
    chunked.load_state_dict(stock.state_dict())

    expected = stock(x, edge_index, edge_attr)
    actual = chunked(x, edge_index, edge_attr)
    assert torch.allclose(actual, expected, atol=1e-6)

    grad_expected = torch.autograd.grad(expected.pow(2).sum(), list(stock.parameters()))
    grad_actual = torch.autograd.grad(actual.pow(2).sum(), list(chunked.parameters()))
    for left, right in zip(grad_expected, grad_actual, strict=True):
        assert torch.allclose(right, left, atol=1e-5, rtol=1e-4)


def test_edge_batch_size_does_not_change_output() -> None:
    x, edge_index, edge_attr, _ = _fixture()
    _, _, _, shared = _fixture()
    source = EdgeConv(4, copy.deepcopy(shared), edge_batch_size=None)
    torch.manual_seed(7)
    source.lin.weight.data.normal_()

    reference = source(x, edge_index, edge_attr)
    for limit in (1, 5, 100):
        layer = EdgeConv(4, copy.deepcopy(shared), edge_batch_size=limit)
        layer.load_state_dict(source.state_dict())
        assert torch.allclose(layer(x, edge_index, edge_attr), reference, atol=1e-6)


def test_empty_edges_return_one_row_per_node() -> None:
    model = KernelNN(2, in_channels=3, edge_channels=5, width=8, ker_width=8, depth=2)
    empty_index = torch.empty((2, 0), dtype=torch.long)
    empty_attr = torch.empty((0, 5))
    out = model(torch.rand(6, 3), empty_index, empty_attr)
    assert out.shape == (6, 2)


def test_state_dict_round_trip_rebuilds_the_operator() -> None:
    torch.manual_seed(1)
    model = KernelNN(2, in_channels=3, edge_channels=5, width=8, ker_width=8, depth=2)
    rebuilt = KernelNN(2, in_channels=3, edge_channels=5, width=8, ker_width=8, depth=2)
    rebuilt.load_state_dict(model.state_dict())

    x = torch.rand(12, 3)
    edge_index, _ = _ring_edges(12, 3)
    edge_attr = torch.rand(edge_index.shape[1], 5)
    assert torch.equal(rebuilt(x, edge_index, edge_attr), model(x, edge_index, edge_attr))
    assert rebuilt.depth == model.depth == 2
