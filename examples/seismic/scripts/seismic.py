"""Seismic demo: 3-D embedding of the F3 cube's seismic traces, drawn as an RGB map.

Domain: the horizontal plane of the Netherlands F3 survey, one node per (inline, crossline)
location, placed on a regular raster so a radius neighbourhood is a fixed set of lattice offsets.
Feature space: the 253 vertical amplitude samples of that location's seismic trace — the depth axis
is what the embedding compresses, from 253 numbers down to 3.
The map is the 3-D output of NOFE, drawn like the ERA5 figures: one RGB channel per embedding
dimension and one `pcolormesh` quad per location, so every trace gets its own colour.

Training and inference use the **same** graph: all 541,501 traces of the cube. Training needs the
whole edge set resident, and at a 50 m radius that is 6.5M edges and about 9 s per epoch on CPU, so
no sub-sampling is needed and the reported scores are like-for-like.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from nofe import NOFE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utilities"))
from f3_graph import BIN_KM, DEFAULT_DATA_DIR, F3Cube  # noqa: E402

# 50 m of lateral support: 12 neighbours per trace at 25 m bins, the 4-connected ring of radius 2.
# Wide enough to see the fault and channel fabric, narrow enough that the full 541,501-trace graph
# trains in ~9 s/epoch.
RADIUS_KM = 0.05
NYSTROM = 0.25
PERCENTILE = 5

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "f3_embedding.png"
# The embedding, saved for the clustering notebook (`notebooks/clusters.py`): training takes ~10
# minutes and the notebook only needs the result.
CACHE = Path(__file__).resolve().parents[1] / "f3_embedding.npz"
# width=8, ker_width=32 are the reference NOFE settings; the message tensor is [edges, width], so
# width drives peak memory. chunk_size is the Nyström query chunk: 262144 keeps the 541,501 queries
# to three passes per step, which is what makes a full-cube epoch affordable on CPU.
CONFIG = {
    "width": 8,
    "ker_width": 32,
    "epochs": 60,
    "patience": 15,
    "seed": 0,
    "device": "cpu",
    "chunk_size": 262_144,
}


def colour_scale(
    values: np.ndarray, percentile: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(centre, spread)`` that maps ``values`` into ``[0, 1]``.

    ``percentile=None`` uses the global min/max; a percentile clips the tail first. A handful of
    extreme traces — one salt flank, one bright reflector — otherwise compress the whole map into a
    grey wash. Non-finite rows are holes: the scale comes from the finite rows only, and a channel
    with no spread falls back to a unit scale so it renders grey rather than NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.isfinite(values).all():
        centre = values.mean(axis=0)
        if percentile is None:
            low, high = values.min(axis=0), values.max(axis=0)
        else:
            low = np.percentile(values, percentile, axis=0)
            high = np.percentile(values, 100.0 - percentile, axis=0)
    else:
        with warnings.catch_warnings():
            # An all-NaN channel is legal here and handled by the fallbacks below.
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmean(values, axis=0)
            if percentile is None:
                low, high = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
            else:
                low = np.nanpercentile(values, percentile, axis=0)
                high = np.nanpercentile(values, 100.0 - percentile, axis=0)
    spread = 2.0 * np.maximum(np.abs(centre - low), np.abs(centre - high))
    spread = np.where(np.isfinite(spread) & (spread != 0), spread, 1.0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    return centre, spread


def apply_colour_scale(values: np.ndarray, centre: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """Map ``values`` into ``[0, 1]`` with a precomputed scale, preserving non-finite holes."""
    values = np.asarray(values, dtype=np.float64)
    scaled = np.clip((values - centre) / spread + 0.5, 0.0, 1.0)
    scaled[~np.isfinite(values)] = np.nan
    return scaled


def rgba_colours(
    values: np.ndarray,
    scale: tuple[np.ndarray, np.ndarray] | None = None,
    percentile: float | None = PERCENTILE,
) -> np.ndarray:
    """Per-point RGBA in ``[0, 1]``: dim 1/2/3 -> R/G/B, alpha 0 for non-finite holes."""
    centre, spread = colour_scale(values, percentile) if scale is None else scale
    scaled = apply_colour_scale(values, centre, spread)
    rgba = np.ones((scaled.shape[0], 4), dtype=np.float64)
    rgba[:, :3] = np.nan_to_num(scaled, nan=0.0)  # colour is irrelevant where alpha is 0
    rgba[:, 3] = np.isfinite(scaled).all(axis=1).astype(np.float64)
    return rgba


def draw_map(
    axis,
    rows_km: np.ndarray,
    cols_km: np.ndarray,
    values: np.ndarray,
    scale: tuple[np.ndarray, np.ndarray] | None = None,
    percentile: float | None = PERCENTILE,
):
    """Colour one quad per trace by its own RGB triple, and return the mesh.

    One location is one cell, so the raster is drawn with ``pcolormesh`` and no interpolation: the
    quads tile the survey exactly, one cell per 25 m bin.
    """
    embed = np.asarray(values, dtype=np.float64)
    rgba = rgba_colours(embed, scale, percentile).reshape(rows_km.size, cols_km.size, 4)
    mesh = axis.pcolormesh(
        cols_km,
        rows_km,
        rgba,
        shading="nearest",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()  # inline index increases downwards, as in a seismic section
    return mesh


def plot(
    rows_km: np.ndarray,
    cols_km: np.ndarray,
    embedding: np.ndarray,
    title: str,
    target: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11.0, 7.5), constrained_layout=True)
    draw_map(axis, rows_km, cols_km, embedding)
    axis.set_xlabel("crossline (km)")
    axis.set_ylabel("inline (km)")
    axis.set_title(f"{title}: NOFE embedding (dim=3), one RGB colour per trace")
    figure.savefig(target, dpi=140)
    plt.close(figure)


def raster_coordinates(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical axis coordinates in km, in the row-major order the graph uses."""
    return (
        np.asarray(rows, dtype=np.float64) * BIN_KM,
        np.asarray(cols, dtype=np.float64) * BIN_KM,
    )


def run(cube: F3Cube, epochs: int):
    """Train on the cube, embed every trace, and return the plot inputs and the scores."""
    started = time.time()
    source_rows, source_cols, depth = cube.geometry()
    features, rows, cols = cube.load()
    data = cube.build(features, rows, cols)
    print(
        f"F3 cube {source_rows}x{source_cols} bins ({source_rows * BIN_KM:.1f} km x "
        f"{source_cols * BIN_KM:.1f} km), {depth} vertical samples per trace"
    )
    print(
        "domain: one node per horizontal location; feature space: the trace of vertical amplitudes"
    )

    model = NOFE(dim=3, config={**CONFIG, "epochs": epochs, "nystrom": NYSTROM})
    history = model.train(data)
    print(
        f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} after "
        f"{len(history)} epochs (best val {model.best_val_loss:.6f})"
    )
    # Training and inference share one graph, so these two numbers are the same quantity: the score
    # is not an artefact of scoring a different neighbourhood than the one trained on.
    embedding = model.embed(data).numpy()
    mean_kl = model.loss(data)
    print(
        f"embedding {tuple(embedding.shape)} over all {data.num_nodes} traces; "
        f"mean KL {mean_kl:.6f}"
    )
    print(f"finished in {time.time() - started:.0f} s")

    rows_km, cols_km = raster_coordinates(rows, cols)
    save_cache(rows_km, cols_km, embedding)
    return rows_km, cols_km, embedding, model, history, mean_kl


def save_cache(rows_km: np.ndarray, cols_km: np.ndarray, embedding: np.ndarray) -> Path:
    """Write the embedding beside the plot so the clustering notebook need not retrain.

    ``np.savez`` keeps this to one small file (4.3 MB for 541,501 x 3) and the notebook loads it
    with ``np.load``: no dependency on this script's module path, no re-run of a 10-minute training.
    """
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, rows_km=rows_km, cols_km=cols_km, embedding=embedding)
    print(f"wrote {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    return CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory holding seismic.npy / labels.npy / metadata.json",
    )
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM, help="lateral support")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="destination PNG")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="inline/crossline stride; 1 is every trace, 2 halves the graph for a quick look",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    args = parser.parse_args()

    cube = F3Cube(radius_km=args.radius_km, stride=args.stride, data_dir=args.data)
    rows_km, cols_km, embedding, model, history, mean_kl = run(cube, args.epochs)

    plot(
        rows_km,
        cols_km,
        embedding,
        f"F3 cube, {embedding.shape[0]} traces, radius {args.radius_km * 1000:.0f} m",
        args.out,
    )
    image_bytes = args.out.stat().st_size
    print(f"wrote {args.out} ({image_bytes / 1024:.0f} KiB)")

    # A trained operator has to beat its own first epoch, and the embedding must not have collapsed
    # onto one channel or left traces undrawn.
    assert model.best_val_loss < history[0]["train_loss"], model.best_val_loss
    assert embedding.shape == (rows_km.size * cols_km.size, 3), embedding.shape
    spread = embedding.std(axis=0)
    assert spread.min() > 0.05, spread
    colours = rgba_colours(embedding)
    assert colours[:, 3].min() == 1.0, "every trace must be drawn"
    assert colours[:, :3].std(axis=0).min() > 0.1, colours[:, :3].std(axis=0)
    assert image_bytes > 50_000, image_bytes
    print(f"mean KL {mean_kl:.6f}; all assertions passed")


if __name__ == "__main__":
    main()
