"""``KernelNN`` as an operator with lazy widths, and ``NOFE`` as the trainer around it."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_point_graph import NnGraph  # noqa: E402

from nofe import NOFE, KernelNN  # noqa: E402
from nofe.affinity import affinity_kl_loss  # noqa: E402


def _graph(num_nodes: int = 40, dim: int = 2, features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    return NnGraph(radius=0.6).build(
        rng.random((num_nodes, features)).astype(np.float32),
        rng.random((num_nodes, dim)).astype(np.float32),
    )


def test_lazy_widths_resolve_on_the_first_forward() -> None:
    net = KernelNN(2, width=8, ker_width=8)
    with pytest.raises(RuntimeError, match="call forward"):
        _ = net.in_channels
    with pytest.raises(RuntimeError, match="call forward"):
        _ = net.edge_channels

    x = torch.rand(10, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.rand(3, 7)
    out = net(x, edge_index, edge_attr)

    assert tuple(out.shape) == (10, 2)
    assert torch.isfinite(out).all()
    assert net.in_channels == 4
    assert net.edge_channels == 7


def test_explicit_widths_match_lazy_ones() -> None:
    torch.manual_seed(0)
    lazy = KernelNN(2, width=8, ker_width=8)
    explicit = KernelNN(2, in_channels=4, edge_channels=7, width=8, ker_width=8)

    x = torch.rand(12, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.rand(4, 7)
    reference = lazy(x, edge_index, edge_attr)  # materialises the lazy parameters
    explicit.load_state_dict(lazy.state_dict())
    assert torch.equal(explicit(x, edge_index, edge_attr), reference)


def test_trainer_holds_an_inspectable_module(tmp_path) -> None:
    # Explicit widths, so the parameters exist before any forward pass: an uninitialised lazy
    # parameter cannot be loaded back with ``weights_only=True``.
    model = NOFE(2, in_channels=4, edge_channels=7, width=8, ker_width=8)
    assert isinstance(model.net, torch.nn.Module)
    assert list(model.net.parameters())

    path = tmp_path / "net.pt"
    torch.save(model.net.state_dict(), path)
    rebuilt = NOFE(2, in_channels=4, edge_channels=7, width=8, ker_width=8)
    rebuilt.net.load_state_dict(torch.load(path, weights_only=True))


def test_fit_trains_then_returns_the_embedding() -> None:
    graph = _graph()
    model = NOFE(2, width=8, ker_width=8)
    embedding = model.fit(graph, epochs=10)

    assert tuple(embedding.shape) == (graph.num_nodes, 2)
    assert torch.isfinite(embedding).all()
    assert torch.allclose(model.embed(graph), embedding, atol=1e-5)


def test_training_embed_and_loss_dispatch_on_the_argument() -> None:
    g1, g2 = _graph(30, seed=1), _graph(50, seed=2)
    model = NOFE(2, width=8, ker_width=8)

    history = model.training(g1, epochs=3)
    assert isinstance(history, list)
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "best_val_loss"}

    histories = model.training([g1, g2], epochs=3)
    assert len(histories) == 2
    assert all(len(part) == 3 for part in histories)

    embeddings = model.embed([g1, g2])
    assert len(embeddings) == 2
    assert embeddings[0].shape[0] == g1.num_nodes
    assert embeddings[1].shape[0] == g2.num_nodes

    assert isinstance(model.loss(g1), float)
    losses = model.loss([g1, g2])
    assert len(losses) == 2 and all(isinstance(value, float) for value in losses)


def test_loss_uses_the_stored_affinities() -> None:
    graph = _graph()
    model = NOFE(2, width=8, ker_width=8, device="cpu")
    with torch.no_grad():
        prediction = model.net(graph.x, graph.edge_index, graph.edge_attr)
        expected = affinity_kl_loss(graph.x, prediction, graph.edge_index, weights=graph.affinity)
    assert model.loss(graph) == pytest.approx(float(expected), abs=1e-6)


def test_construction_seeds_the_initialisation() -> None:
    # The weights are drawn at construction, before `training`'s set_seed runs; two trainers built
    # back to back must therefore agree regardless of the global RNG state in between.
    torch.manual_seed(1234)
    first = NOFE(2, in_channels=4, edge_channels=7, width=8, ker_width=8)
    torch.manual_seed(4321)
    second = NOFE(2, in_channels=4, edge_channels=7, width=8, ker_width=8)
    for left, right in zip(first.net.parameters(), second.net.parameters(), strict=True):
        assert torch.equal(left, right)


def test_loss_scores_the_configured_operator() -> None:
    # On the Nyström path a dense pass reports a KL the trained operator does not achieve, so the
    # score must come from the same forward path training used.
    graph = _graph(200, seed=5)
    model = NOFE(2, nystrom=0.25, chunk_size=32, width=8, ker_width=8, device="cpu")
    model.training(graph, epochs=3)
    dense = float(model._graph_loss(graph, model.net(graph.x, graph.edge_index, graph.edge_attr)))
    assert model.loss(graph) != pytest.approx(dense, abs=1e-6)


def test_nystrom_path_trains_and_embeds() -> None:
    graph = _graph(400, seed=3)
    model = NOFE(3, nystrom=0.25, chunk_size=16, width=8, ker_width=8)
    model.training(graph, epochs=3)

    embedding = model.embed(graph)
    assert tuple(embedding.shape) == (graph.num_nodes, 3)
    assert torch.isfinite(embedding).all()
    assert np.isfinite(model.loss(graph))


def test_invalid_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="nystrom"):
        NOFE(3, nystrom=0.0)
    with pytest.raises(ValueError, match="chunk_size"):
        NOFE(3, chunk_size=0)
    with pytest.raises(ValueError, match="share both widths"):
        NOFE(3, width=8, ker_width=8).training([_graph(20, seed=4), _graph(20, features=5)])
