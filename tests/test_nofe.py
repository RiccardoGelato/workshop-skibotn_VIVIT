import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from nofe import NOFE, build_graph
from nofe.graphs import knn_graph


def _graph(num_nodes: int = 200) -> Data:
    rng = np.random.default_rng(0)
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, num_nodes)
    height = rng.uniform(0.0, 10.0, num_nodes)
    pos = np.stack([angle * np.cos(angle), height], axis=1)
    pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0)) * (2 * np.pi)
    features = np.stack([np.sin(angle), np.cos(angle), height / 10.0], axis=1)
    pos = pos.astype(np.float32)
    features = features.astype(np.float32)
    edge_index, edge_attr = knn_graph(pos, 10)
    return Data(
        x=torch.from_numpy(features),
        pos=torch.from_numpy(pos),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def _config(**overrides) -> dict:
    base = {"device": "cpu", "seed": 0, "width": 16, "ker_width": 16, "depth": 2}
    base.update(overrides)
    return base


def _split_graph(num_inputs: int = 120, num_added: int = 30) -> tuple[Data, Data, Data]:
    """Swiss-roll graph split into inputs and added locations; both truth states."""
    rng = np.random.default_rng(7)
    total = num_inputs + num_added
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, total)
    height = rng.uniform(0.0, 10.0, total)
    pos = np.stack([angle * np.cos(angle), height], axis=1)
    pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0)) * (2 * np.pi)
    features = np.stack([np.sin(angle), np.cos(angle), height / 10.0], axis=1)
    pos = pos.astype(np.float32)
    features = features.astype(np.float32)

    in_sample = build_graph(pos[:num_inputs], features[:num_inputs], neighbor="knn", k=10)
    merged = build_graph(
        pos[:num_inputs],
        features[:num_inputs],
        query_positions=pos[num_inputs:],
        query_features=features[num_inputs:],
        neighbor="knn",
        k=10,
    )
    blind = build_graph(
        pos[:num_inputs],
        features[:num_inputs],
        query_positions=pos[num_inputs:],
        neighbor="knn",
        k=10,
    )
    return in_sample, merged, blind


def _small(**overrides) -> dict:
    base = {"epochs": 40}
    base.update(overrides)
    return _config(**base)


def test_graph_training_reports_history_and_finite_loss() -> None:
    data = _graph()
    original_x = data.x.clone()
    model = NOFE(dim=3, config=_config(epochs=6, early_stopping=False))
    history = model.train(data)

    assert len(history) == 6
    assert [entry["epoch"] for entry in history] == list(range(1, 7))
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert all(np.isfinite(entry["train_loss"]) for entry in history)
    assert history[-1]["best_val_loss"] == min(entry["val_loss"] for entry in history)
    assert model.best_val_loss == history[-1]["best_val_loss"]

    assert np.isfinite(model.loss(data))
    assert model.loss(data) < 0.2

    embedding = model.embed(data)
    assert embedding.shape == (data.num_nodes, 3)
    assert embedding.dtype == torch.float32
    assert embedding.device.type == "cpu"
    # to_device clones: the caller's graph keeps device and dtype.
    assert data.x.device.type == "cpu"
    assert data.edge_index.device.type == "cpu"
    assert torch.equal(data.x, original_x)


def test_early_stopping_toggle_controls_epoch_count() -> None:
    data = _graph(120)
    patient = NOFE(dim=2, config=_config(epochs=9, early_stopping=True, patience=1))
    assert len(patient.train(data)) < 9

    exact = NOFE(dim=2, config=_config(epochs=9, early_stopping=False, patience=1))
    assert len(exact.train(data)) == 9


def test_verbose_logging_respects_log_every_and_final_epoch(capsys) -> None:
    data = _graph(60)
    model = NOFE(
        dim=2,
        config=_config(epochs=5, early_stopping=False, verbose=True, log_every=2),
    )
    model.train(data)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3
    assert printed[0].startswith("epoch    2/5")
    assert printed[-1].startswith("epoch    5/5")


def test_save_load_round_trip(tmp_path) -> None:
    data = _graph(80)
    model = NOFE(dim=2, config=_config(epochs=4, early_stopping=False))
    model.train(data)
    expected = model.embed(data)

    path = model.save(tmp_path / "model.pt")
    assert path.exists()

    restored = NOFE.load(path)
    assert restored.dim == 2
    assert restored.best_val_loss == model.best_val_loss
    assert torch.allclose(restored.embed(data), expected, atol=1e-6)

    on_device = NOFE.load(path, device="cpu")
    assert on_device.device.type == "cpu"
    assert torch.allclose(on_device.embed(data), expected, atol=1e-6)


def test_method_checkpoints_are_rejected_with_a_migration_hint(tmp_path) -> None:
    data = _graph(80)
    model = NOFE(dim=2, config=_config(epochs=2, early_stopping=False))
    model.train(data)
    path = model.save(tmp_path / "model.pt")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format"] = "nofe-checkpoint-1"
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    with pytest.raises(ValueError, match="nofe-checkpoint-1 file"):
        NOFE.load(legacy)


def test_untrained_and_foreign_checkpoints_raise(tmp_path) -> None:
    model = NOFE(dim=2, config=_config())
    with pytest.raises(RuntimeError, match="not built yet"):
        model.save(tmp_path / "never.pt")

    foreign = tmp_path / "foreign.pt"
    torch.save({"hello": "world"}, foreign)
    with pytest.raises(ValueError, match="is not a NOFE checkpoint"):
        NOFE.load(foreign)


def test_merged_graph_reads_truth_from_y_and_the_substitute_from_x() -> None:
    _, merged, _ = _split_graph()
    untrained = NOFE(dim=2, config=_small(early_stopping=False))
    before = untrained.loss(merged)
    model = NOFE(dim=2, config=_small(early_stopping=False))
    history = model.train(merged)
    assert len(history) == 40
    assert all(np.isfinite(entry["train_loss"]) for entry in history)

    embedding = model.embed(merged)
    assert embedding.shape == (150, 2)
    trained = model.loss(merged)
    assert np.isfinite(trained)
    assert trained < before

    # Truth reaches the objective only: zeroing y changes the loss, not the embedding.
    without_truth = merged.clone()
    without_truth.y = merged.y.clone()
    without_truth.y[120:] = 0.0
    assert model.loss(without_truth) != trained
    assert torch.equal(model.embed(without_truth), embedding)

    # The substitute is what the model consumes: overwriting x changes the embedding.
    other_input = merged.clone()
    other_input.x = merged.x.clone()
    other_input.x[120:] = 0.5
    assert not torch.equal(model.embed(other_input), embedding)


def test_added_locations_enter_the_objective() -> None:
    in_sample, merged, _ = _split_graph()
    on_inputs = NOFE(dim=2, config=_small()).train(in_sample)
    on_merged = NOFE(dim=2, config=_small()).train(merged)
    assert on_merged[-1]["train_loss"] != on_inputs[-1]["train_loss"]


def test_unscorable_graph_trains_and_scores_nowhere_but_embeds() -> None:
    _, _, blind = _split_graph()
    model = NOFE(dim=2, config=_small())
    with pytest.raises(ValueError, match="no ground-truth features"):
        model.train(blind)
    with pytest.raises(ValueError, match="no ground-truth features"):
        model.loss(blind)
    assert model.embed(blind).shape == (150, 2)


def test_nystrom_is_fixed_at_construction_and_used_by_every_entry_point() -> None:
    _, merged, _ = _split_graph()
    config = _small(nystrom=0.3, chunk_size=16, epochs=5)
    model = NOFE(dim=2, config=config)

    history = model.train(merged)
    assert len(history) == 5
    assert all(np.isfinite(entry["train_loss"]) for entry in history)

    embedding = model.embed(merged)
    assert embedding.shape == (150, 2)
    assert embedding.device.type == "cpu"

    first = model.loss(merged)
    assert np.isfinite(first)
    assert model.loss(merged) == pytest.approx(first, abs=1e-5)

    dense = NOFE(dim=2, config=_small(nystrom=None, epochs=5))
    dense.train(merged)
    assert dense.embed(merged).shape == (150, 2)
    assert not torch.allclose(dense.embed(merged), embedding, atol=1e-3)


def test_sampler_requires_nystrom_to_be_enabled() -> None:
    model = NOFE(dim=2, config=_config(nystrom=None))
    with pytest.raises(RuntimeError, match="nystrom is not enabled"):
        model._sampler()


def test_config_and_graph_validation_errors() -> None:
    model = NOFE(dim=2, config=_config(epochs=1))
    assert repr(model) == "NOFE(dim=2, width=16, depth=2, device='cpu')"

    for kwargs, match in (
        ({"config": {"nope": 1}}, "unknown config key"),
        ({"config": {"dim": 3}}, "constructor argument"),
        ({"config": {"neighbor": "grid"}}, "neighbor must be one of"),
        ({"config": {"nystrom": 0.0}}, "nystrom must be None or a fraction"),
        ({"config": {"nystrom": True}}, "nystrom must be None or a fraction"),
        ({"dim": 0}, "dim must be a positive integer"),
        ({"val_fraction": 1.5}, "val_fraction"),
    ):
        with pytest.raises(ValueError, match=match):
            NOFE(**kwargs)

    data = _graph(20)
    with pytest.raises(TypeError, match="expected torch_geometric.data.Data"):
        model.train([[1, 2]])

    bad_dtype = data.clone()
    bad_dtype.edge_index = bad_dtype.edge_index.to(torch.int32)
    with pytest.raises(ValueError, match="edge_index must be torch.long"):
        model.train(bad_dtype)

    out_of_range = data.clone()
    out_of_range.edge_index = out_of_range.edge_index + data.num_nodes
    with pytest.raises(ValueError, match="outside"):
        model.train(out_of_range)

    missing_attr = data.clone()
    missing_attr.edge_attr = None
    with pytest.raises(ValueError, match="requires fields"):
        model.train(missing_attr)

    no_pos = data.clone()
    no_pos.pos = None
    model.train(no_pos)  # pos is optional for the dense path

    bad_pos = data.clone()
    bad_pos.pos = torch.zeros(data.num_nodes)
    with pytest.raises(ValueError, match="pos must be 2-D"):
        model.train(bad_pos)

    bad_mask = data.clone()
    bad_mask.input_mask = torch.ones(data.num_nodes - 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="input_mask must have"):
        model.train(bad_mask)

    not_subset = data.clone()
    not_subset.input_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    not_subset.output_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    not_subset.output_mask[:5] = True
    with pytest.raises(ValueError, match="input_mask must be a subset of output_mask"):
        model.train(not_subset)

    bad_truth = data.clone()
    bad_truth.y = torch.zeros(data.num_nodes, 7)
    with pytest.raises(ValueError, match="y must have the same shape as x"):
        model.train(bad_truth)

    degenerate = NOFE(dim=2, config=_config(val_fraction=0.9))
    with pytest.raises(ValueError, match="val_fraction"):
        degenerate._split(Data(x=torch.zeros(2, 1)))
