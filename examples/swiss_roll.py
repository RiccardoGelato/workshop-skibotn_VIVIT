"""Swiss-roll demo: exact graph-kernel embedding and Nyström approximation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from nofe import NOFE, build_graph


def swiss_roll(num_points: int = 600) -> Data:
    """Swiss roll: features are 3-D, intrinsic coordinates are 2-D."""
    positions, features = swiss_roll_arrays(num_points)
    return build_graph(positions, features, neighbor="knn", k=10)


def swiss_roll_arrays(num_points: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Raw swiss-roll positions and features, before any graph is built."""
    rng = np.random.default_rng(0)
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, num_points)
    height = rng.uniform(0.0, 10.0, num_points)
    positions = np.stack([angle * np.cos(angle), height], axis=1)
    positions = (
        (positions - positions.min(axis=0))
        / (positions.max(axis=0) - positions.min(axis=0))
        * (2 * np.pi)
    )
    features = np.stack([np.sin(angle), np.cos(angle), height / 10.0], axis=1)
    return positions.astype(np.float32), features.astype(np.float32)


def knn_overlap(
    features: torch.Tensor, embedding: torch.Tensor, k: int = 10, limit: int = 200
) -> float:
    """Mean share of shared k-nearest neighbours between two point sets."""
    reference = features[:limit].detach().cpu().numpy()
    candidate = embedding[:limit].detach().cpu().numpy()
    ref_tree = cKDTree(reference)
    cand_tree = cKDTree(candidate)
    _, ref_nn = ref_tree.query(reference, k=k + 1)
    _, cand_nn = cand_tree.query(candidate, k=k + 1)
    shares = []
    for i in range(limit):
        shared = len(set(ref_nn[i][1:]) & set(cand_nn[i][1:]))
        shares.append(shared / k)
    return float(np.mean(shares))


def main() -> None:
    positions, features = swiss_roll_arrays(600)
    data = swiss_roll(600)
    print(f"graph: {data.num_nodes} nodes, {data.num_edges} edges, features {data.num_features}")

    model = NOFE(dim=2, config={"epochs": 150, "seed": 42, "verbose": True, "log_every": 50})
    history = model.train(data)
    print(
        f"loss {history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.6f} "
        f"after {len(history)} epochs (best val {model.best_val_loss:.6f})"
    )

    embedding = model.embed(data)
    overlap = knn_overlap(data.x, embedding, k=10)
    print(f"10-NN overlap vs features {overlap:.3f} (chance {10 / 599:.4f})")
    print(f"loss on the full graph {model.loss(data):.6f}")

    with tempfile.TemporaryDirectory() as directory:
        path = model.save(Path(directory) / "swiss_roll.pt")
        restored = NOFE.load(path)
        rebuilt = restored.embed(data)
        assert torch.allclose(rebuilt, embedding, atol=1e-6)
        print(f"checkpoint round trip ok ({path.name})")

    # 500 observed locations plus 100 withheld ones: the added rows get an interpolated
    # substitute in `x`, keep their truth in `y`, and receive messages without sending any.
    merged = build_graph(
        positions[:500],
        features[:500],
        query_positions=positions[500:],
        query_features=features[500:],
        neighbor="knn",
        k=10,
    )
    print(
        f"merged graph: {merged.num_nodes} nodes "
        f"({int(merged.input_mask.sum())} observed), {merged.num_edges} edges"
    )

    nystrom = NOFE(
        dim=2,
        config={"nystrom": 0.2, "epochs": 50, "seed": 0, "device": "cpu", "log_every": 10},
    )
    nystrom_history = nystrom.train(merged)
    approximated_embedding = nystrom.embed(merged)
    print(
        f"nystrom: {len(nystrom_history)} epochs, last train loss "
        f"{nystrom_history[-1]['train_loss']:.6f}"
    )
    print(
        f"nystrom embedding {tuple(approximated_embedding.shape)}, loss {nystrom.loss(merged):.6f}"
    )
    merged_overlap = knn_overlap(merged.y, approximated_embedding, k=10)
    print(f"nystrom 10-NN overlap vs features {merged_overlap:.3f}")

    assert history[-1]["train_loss"] < 0.01
    assert model.best_val_loss < 0.01
    assert overlap > 0.2
    assert embedding.shape == (600, 2)
    assert approximated_embedding.shape == (600, 2)
    assert nystrom.loss(merged) < 0.01
    print("all assertions passed")


if __name__ == "__main__":
    main()
