"""k-means over the F3 embedding, for sweeping ``k`` and drawing the clusters as a map.

The embedding is only 3-D, so k-means is written here rather than pulling in scikit-learn: fitting
on a subsample and assigning every trace is a few seconds for 541,501 points at any ``k`` in reach,
and the package keeps its three runtime dependencies.

Deterministic by construction: k-means++ seeding from a numpy ``Generator`` with a fixed seed, so a
given ``(X, k, seed)`` always yields the same labels — which matters, because the point of the sweep
is to compare ``k`` against ``k``, and unstable initialisation would confound that with the sweep.
"""

from __future__ import annotations

import numpy as np


def kmeans(
    points: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    iterations: int = 100,
    tolerance: float = 1e-6,
    fit_sample: int | None = 50_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster ``points`` into ``k`` groups; return ``(centres, labels, inertia)``.

    ``fit_sample`` fits the centres on a random subset and then assigns every point, which keeps
    541,501 three-dimensional points cheap: the expensive step is the per-iteration distance matrix,
    and the centres barely move once the sample is a few percent of the data. Pass ``None`` to fit
    on everything.

    ``inertia`` is the mean squared distance of a point to its own centre, so it is comparable
    across ``k`` and does not grow with the point count.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"points must be 2-D [num_points, dims], got shape {values.shape}")
    if not 1 <= k <= values.shape[0]:
        raise ValueError(f"k must be in [1, {values.shape[0]}], got {k}")

    rng = np.random.default_rng(seed)
    if fit_sample is not None and fit_sample < values.shape[0]:
        fit = values[rng.choice(values.shape[0], fit_sample, replace=False)]
    else:
        fit = values
    centres = _seed_centres(fit, k, rng)
    for _ in range(iterations):
        labels = _assign(fit, centres)
        updated = _update(fit, labels, centres)
        shift = float(np.square(updated - centres).sum(axis=1).max())
        centres = updated
        if shift < tolerance:
            break

    labels = _assign(values, centres)
    distances = np.square(values - centres[labels]).sum(axis=1)
    return centres, labels, float(distances.mean())


def _seed_centres(fit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: each new centre is drawn with probability proportional to ``D^2``."""
    centres = np.empty((k, fit.shape[1]), dtype=fit.dtype)
    centres[0] = fit[rng.integers(fit.shape[0])]
    closest = np.square(fit - centres[0]).sum(axis=1)
    for index in range(1, k):
        total = closest.sum()
        if total <= 0:  # every remaining point coincides with a centre
            centres[index] = centres[index - 1]
            continue
        position = np.searchsorted(np.cumsum(closest / total), rng.random())
        centres[index] = fit[min(position, fit.shape[0] - 1)]
        closest = np.minimum(closest, np.square(fit - centres[index]).sum(axis=1))
    return centres


def _assign(points: np.ndarray, centres: np.ndarray, chunk: int = 65_536) -> np.ndarray:
    """Index of the nearest centre for every point, walked in row chunks to bound peak memory."""
    labels = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        distances = np.square(block[:, None, :] - centres[None]).sum(axis=2)
        labels[start : start + chunk] = distances.argmin(axis=1)
    return labels


def _update(points: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Centre of each cluster; an empty cluster keeps its previous centre."""
    updated = centres.copy()
    for index in range(centres.shape[0]):
        members = labels == index
        if members.any():
            updated[index] = points[members].mean(axis=0)
    return updated


def sweep(
    points: np.ndarray, ks: list[int], *, seed: int = 0, **kwargs
) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """``kmeans`` at every ``k`` in ``ks``, keyed by ``k``."""
    return {k: kmeans(points, k, seed=seed, **kwargs) for k in ks}


def cluster_colours(labels: np.ndarray, k: int) -> np.ndarray:
    """Per-point RGBA, one distinct colour per cluster, all fully opaque.

    ``tab20`` read as ``(k // 20)`` repetitions needs more than 20 colours for a wide sweep, so the
    hue comes from an evenly spaced HSV wheel: adjacent clusters then differ in hue and stay
    distinguishable up to the ``k`` a 3-D embedding can plausibly support.
    """
    import colorsys

    hues = np.linspace(0.0, 1.0, k, endpoint=False)
    palette = np.array(
        [colorsys.hsv_to_rgb(float(hue), 0.72, 0.95) for hue in hues], dtype=np.float64
    )
    rgba = np.ones((labels.size, 4), dtype=np.float64)
    rgba[:, :3] = palette[labels % k]
    return rgba
