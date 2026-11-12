"""ERA5 demo: 3-D embedding of the global weather field, drawn as an RGB globe.

Domain: every ERA5 grid point on the globe, placed on the unit sphere so the
longitude seam and the poles need no special casing.
Feature space: the 11 physical variables stored for that grid point.
The map is the 3-D output of NOFE, drawn like the reference NOFE figures: one RGB
channel per embedding dimension and one `pcolormesh` quad per grid point, so every
point carries its own colour on a cartopy projection over the whole globe.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

from nofe import NOFE, build_graph

DEFAULT_DATA_DIR = "/Volumes/T7-storage/ERA5/data"
DEFAULT_DATE = "2018-01-01"
VARIABLES = ("w", "v", "z", "q", "r", "t", "d", "o3", "pv", "u", "vo")
STRIDE = 2
K = 10
PERCENTILE = 5

DEFAULT_OUTPUT = Path(__file__).with_name("era5_embedding.png")
CONFIG = {"width": 8, "ker_width": 32, "epochs": 80, "patience": 20, "seed": 0}


def load_fields(path: Path, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised ``[lat, lon, variable]`` block over the globe plus its coordinates."""
    import netCDF4

    dataset = netCDF4.Dataset(path)
    try:
        latitudes = np.asarray(dataset["latitude"][:], dtype=np.float64)
        longitudes = np.asarray(dataset["longitude"][:], dtype=np.float64)
        # ERA5 stores longitude in [0, 360) and latitude north to south; both axes are
        # reordered to increasing order so the block reshapes straight into a raster.
        longitude_order = np.argsort((longitudes + 180.0) % 360.0 - 180.0)
        longitudes = ((longitudes + 180.0) % 360.0 - 180.0)[longitude_order]
        latitude_order = np.argsort(latitudes)
        latitudes = latitudes[latitude_order]

        # The two pole rows repeat one value along every longitude: keeping them would add
        # 1440 coincident nodes with zero-length edges. Sample the rest evenly, aligned on
        # the equator so the two hemispheres are covered symmetrically.
        usable = np.flatnonzero(np.abs(latitudes) < 90.0)
        lat_rows = usable[usable.size // 2 % stride :: stride]
        lon_cols = np.arange(0, longitudes.size, stride)

        channels = []
        for name in VARIABLES:
            if name not in dataset.variables:
                raise SystemExit(f"{path} has no variable {name!r}")
            field = np.asarray(dataset[name][0, 0], dtype=np.float32)
            block = field[np.ix_(latitude_order[lat_rows], longitude_order)][:, lon_cols]
            channels.append((block - block.mean()) / (block.std() + 1e-8))
    finally:
        dataset.close()
    return np.stack(channels, axis=-1), latitudes[lat_rows], longitudes[lon_cols]


def unit_sphere(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Lat/lon grid to 3-D unit vectors: the domain the neighbourhood graph lives in.

    Directions on the sphere make the ±180° seam and the pole crossings ordinary
    neighbours, so a global kNN graph needs no longitude duplication.
    """
    phi = np.deg2rad(np.repeat(latitudes, longitudes.size))
    lam = np.deg2rad(np.tile(longitudes, latitudes.size))
    return np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=1
    ).astype(np.float32)


def colour_scale(
    values: np.ndarray, percentile: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(centre, spread)`` that maps ``values`` into ``[0, 1]``.

    ``percentile=None`` uses the global min/max (the reference ``rgb``); a percentile clips the
    tail first (the reference ``rgb_scale``). A single extreme point — one cyclone, one front —
    otherwise compresses the whole field into a grey wash.

    Non-finite rows are holes to leave blank, not data: the scale comes from the finite rows only,
    because ``mean``/``percentile`` over any NaN return NaN for the whole channel. A channel with
    no finite row (or a constant one) falls back to a unit spread, so it renders grey rather than
    NaN.
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
    """Per-point RGBA in ``[0, 1]``: dim 1/2/3 -> R/G/B, alpha 0 for non-finite holes.

    ``scale`` is a ``colour_scale`` result to reuse; pass it when several panels must share one
    colour mapping, otherwise each panel would rescale independently and identical points would
    render in different colours. Holes get alpha 0, so pcolormesh leaves the map underneath
    visible instead of painting them black.
    """
    centre, spread = colour_scale(values, percentile) if scale is None else scale
    scaled = apply_colour_scale(values, centre, spread)
    rgba = np.ones((scaled.shape[0], 4), dtype=np.float64)
    rgba[:, :3] = np.nan_to_num(scaled, nan=0.0)  # colour is irrelevant where alpha is 0
    rgba[:, 3] = np.isfinite(scaled).all(axis=1).astype(np.float64)
    return rgba


def draw_globe(
    axis,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    scale: tuple[np.ndarray, np.ndarray] | None = None,
    percentile: float | None = PERCENTILE,
):
    """Colour one quad per grid point by its own RGB triple, and return the mesh.

    A grid point is a cell, not a pixel: ``pcolormesh`` draws one quad per point on a cartopy
    projection, so the raster needs no repeated seam column and no ``imshow`` extrapolation past
    180° E. The coordinates are the cell *centres* with ``shading="nearest"``, which is what makes
    the quad straddle the point: cartopy then lays the boundary half a step past the outermost
    centre on both sides, so the quads cover exactly 360° of longitude and reach the poles. Passing
    hand-built edges instead leaves a one-step gap at the antimeridian for a full circle.

    ``values`` is in graph order, ``latitudes.size * longitudes.size`` rows of three channels.
    ``axis`` must already have a cartopy projection; the caller owns coastlines, titles and saving.
    """
    import cartopy.crs as ccrs

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    num_lat, num_lon = latitudes.size, longitudes.size
    if values.shape != (num_lat * num_lon, 3):
        raise ValueError(
            f"expected {num_lat * num_lon} rows of three channels (one per grid point), "
            f"got {values.shape}"
        )
    if num_lon > 1 and not np.isclose(
        float(np.median(np.diff(longitudes))), float(longitudes[1] - longitudes[0])
    ):
        raise ValueError("longitudes must be regularly spaced for one quad per grid point")

    rgba = rgba_colours(values, scale, percentile).reshape(num_lat, num_lon, 4)
    # RGBA goes in as ``C``, not as ``color=``: ``color`` only sets one facecolour that the
    # scalar array then overrides at draw time, which renders the whole map in the colormap's
    # lowest colour. With RGB(A) ``C`` matplotlib paints each quad from its own triple.
    return axis.pcolormesh(
        longitudes,
        latitudes,
        rgba,
        shading="nearest",
        transform=ccrs.PlateCarree(),
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )


def globe_figure(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    embedding: np.ndarray,
    title: str,
    scale: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Cartopy figure of the embedding, one RGB colour per point; the caller saves or shows it."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(12.0, 6.5), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0.0)}
    )
    axis.set_global()
    draw_globe(axis, latitudes, longitudes, embedding, scale=scale)
    axis.coastlines(linewidth=0.5)
    axis.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)
    axis.set_title(f"{title}: NOFE embedding (dim=3), one RGB colour per point")
    return figure


def plot(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    embedding: np.ndarray,
    title: str,
    target: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = globe_figure(latitudes, longitudes, embedding, title)
    figure.savefig(target, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NOFE_ERA5_DATA", DEFAULT_DATA_DIR)),
        help="directory holding ERA5 netCDF files (env: NOFE_ERA5_DATA)",
    )
    parser.add_argument(
        "--date",
        default=os.environ.get("NOFE_ERA5_DATE", DEFAULT_DATE),
        help="file stem to read, e.g. 2018-01-01 (env: NOFE_ERA5_DATE)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="destination PNG")
    args = parser.parse_args()

    source = args.data / f"{args.date}.nc"
    if not source.exists():
        raise SystemExit(
            f"no ERA5 file at {source}; set NOFE_ERA5_DATA to the directory that holds "
            f"<date>.nc (default {DEFAULT_DATA_DIR}) or pass --data"
        )

    features, latitudes, longitudes = load_fields(source)
    num_lat, num_lon = features.shape[:2]
    flat = features.reshape(-1, features.shape[2])
    positions = unit_sphere(latitudes, longitudes)
    data = build_graph(positions, flat, neighbor="knn", k=K)
    print(
        f"{source.name}: {num_lat}x{num_lon} = {data.num_nodes} nodes over the globe, "
        f"{flat.shape[1]} variables, {data.num_edges} edges"
    )
    print(
        "domain: every ERA5 grid point on the unit sphere; "
        "feature space: the 11 variables at each point"
    )

    model = NOFE(dim=3, config={**CONFIG, "neighbor": "knn", "k": K})
    history = model.train(data)
    print(
        f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} after "
        f"{len(history)} epochs (best val {model.best_val_loss:.6f})"
    )

    embedding = model.embed(data).numpy()
    mean_kl = model.loss(data)
    print(f"embedding {tuple(embedding.shape)}; mean KL {mean_kl:.6f}")

    plot(latitudes, longitudes, embedding, f"ERA5 {args.date} 500 hPa, global", args.out)
    image_bytes = args.out.stat().st_size
    print(f"wrote {args.out} ({image_bytes / 1024:.0f} KiB)")

    colours = rgba_colours(embedding)
    spread = embedding.std(axis=0)
    assert history[-1]["train_loss"] < 0.02, history[-1]["train_loss"]
    assert model.best_val_loss < 0.02, model.best_val_loss
    assert mean_kl < 0.02, mean_kl
    assert embedding.shape == (data.num_nodes, 3)
    assert spread.min() > 0.05, spread
    # One RGBA colour per grid point, all opaque (the demo map has no holes).
    assert colours.shape == (num_lat * num_lon, 4), colours.shape
    assert np.all(colours[:, 3] == 1.0), np.unique(colours[:, 3])
    # A grey map means the channels collapsed; the reference figures show full saturation.
    assert colours[:, :3].std(axis=0).min() > 0.1, colours[:, :3].std(axis=0)
    assert image_bytes > 50_000, image_bytes
    print("all assertions passed")


if __name__ == "__main__":
    main()
