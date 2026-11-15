"""k-means over the seismic embedding: correctness, determinism and the palette.

Synthetic blobs rather than the 4.3 MB F3 cache, so the tests pin the clustering itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "seismic" / "utilities"))
from clusters import cluster_colours, kmeans, sweep  # noqa: E402


def _blobs(centres, per_blob: int = 400, spread: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(scale=spread, size=(per_blob, len(centre))) + centre for centre in centres]
    ).astype(np.float32)


def test_recovers_separated_blobs_exactly() -> None:
    centres = [(-5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)]
    points = _blobs(centres)
    found, labels, inertia = kmeans(points, 3, seed=0)
    # The centres land on the planted means, and the spread is the planted noise: 3 dimensions of
    # variance 0.3^2 give a mean squared distance near 0.27, far below the between-blob scale (25).
    assert np.allclose(np.sort(found, axis=0), np.sort(np.array(centres), axis=0), atol=0.1)
    assert 0.0 < inertia < 1.0
    per_blob = [set(labels[i : i + 400].tolist()) for i in range(0, points.shape[0], 400)]
    assert all(len(found_labels) == 1 for found_labels in per_blob)
    assert len({next(iter(found_labels)) for found_labels in per_blob}) == 3


def test_k_equals_one_returns_the_global_mean() -> None:
    points = _blobs([(0.0, 0.0, 0.0)], per_blob=200)
    centres, labels, inertia = kmeans(points, 1)
    assert centres.shape == (1, 3)
    assert np.allclose(centres[0], points.mean(axis=0), atol=1e-6)
    assert np.array_equal(labels, np.zeros(points.shape[0], dtype=labels.dtype))
    assert inertia == pytest.approx(float(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


def test_spread_falls_as_k_grows_and_the_same_seed_is_reproducible() -> None:
    points = _blobs([(-4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (4.0, 0.0, 0.0)], per_blob=200)
    results = sweep(points, [1, 2, 3, 6], seed=7)
    inertias = [results[k][2] for k in (1, 2, 3, 6)]

    drops = [later - earlier for earlier, later in zip(inertias, inertias[1:], strict=False)]
    assert all(drop <= 1e-9 for drop in drops), inertias
    # k=3 is the planted structure, so k=6 cannot buy much more.
    assert inertias[-1] > 0.5 * inertias[2]
    assert np.array_equal(kmeans(points, 3, seed=7)[1], results[3][1])
    # A different seed is allowed to differ, but must still be a valid labelling.
    other = kmeans(points, 3, seed=8)[1]
    assert set(np.unique(other).tolist()) <= {0, 1, 2}


def test_fitting_on_a_sample_still_clusters_every_point() -> None:
    points = _blobs([(-6.0, 0.0, 0.0), (6.0, 0.0, 0.0)], per_blob=2_000)
    centres, labels, _ = kmeans(points, 2, seed=0, fit_sample=200)

    assert labels.shape == (points.shape[0],)
    # The sample is enough to find both blobs, so every point is assigned to its own side.
    sign = np.sign(points[:, 0])
    assert len(set(labels[sign < 0].tolist())) == 1
    assert len(set(labels[sign > 0].tolist())) == 1
    assert len(set(labels[sign < 0].tolist()) ^ set(labels[sign > 0].tolist())) == 2
    assert np.allclose(np.sort(centres[:, 0]), [-6.0, 6.0], atol=0.2)


def test_rejects_impossible_k_and_bad_shapes() -> None:
    points = _blobs([(0.0, 0.0, 0.0)], per_blob=10)
    with pytest.raises(ValueError, match="k must be in"):
        kmeans(points, 0)
    with pytest.raises(ValueError, match="k must be in"):
        kmeans(points, points.shape[0] + 1)
    with pytest.raises(ValueError, match="points must be 2-D"):
        kmeans(np.zeros(5), 2)


def test_palette_gives_one_distinct_opaque_colour_per_cluster() -> None:
    labels = np.repeat(np.arange(8), 3)
    colours = cluster_colours(labels, 8)

    assert colours.shape == (labels.size, 4)
    assert np.all(colours[:, 3] == 1.0)
    # One colour per cluster, and adjacent clusters are told apart.
    assert len(np.unique(colours[:, :3], axis=0)) == 8
    per_cluster = colours[::3, :3]
    gaps = np.square(per_cluster[1:] - per_cluster[:-1]).sum(axis=1)
    assert gaps.min() > 0.01
    # Past 20 clusters the hue wheel keeps every label in range rather than repeating a colour.
    wide = cluster_colours(np.arange(40), 40)
    assert len(np.unique(wide[:, :3], axis=0)) == 40
