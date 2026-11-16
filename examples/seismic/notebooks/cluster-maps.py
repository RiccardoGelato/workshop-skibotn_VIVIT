import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # k-means over the F3 embedding

    The embedding from `examples/seismic/scripts/seismic.py` maps each seismic trace to three
    numbers. Here those numbers are clustered at several `k`, and each labelling is drawn back
    onto the survey: one colour per cluster, one cell per 25 m bin.

    Clusters of the *embedding* group traces whose lateral neighbourhoods look alike, so
    coherent colour patches are geology the operator considers similar — salt bodies, the
    channel system, fault blocks — not noise, because neighbouring traces feed each other's
    embedding.

    The cache is written by the training script, so nothing here retrains:

    ```bash
    uv run python examples/seismic/scripts/seismic.py    # writes f3_embedding.npz
    uv run marimo edit examples/seismic/notebooks/clusters.py
    ```
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    REPO = next(
        (
            ancestor
            for ancestor in (Path.cwd(), *Path.cwd().parents)
            if (ancestor / "pyproject.toml").is_file()
        ),
        None,
    )
    if REPO is None:
        raise RuntimeError(f"run this notebook from inside the nofe checkout, not {Path.cwd()}")
    CACHE = REPO / "examples" / "seismic" / "f3_embedding.npz"
    # The clustering lives in the example's utilities, so the notebook and any script share one
    # implementation of k-means, the sweep and the palette.
    sys.path.insert(0, str(REPO / "examples" / "seismic" / "utilities"))
    from clusters import cluster_colours, kmeans, sweep  # noqa: E402
    from f3_graph import BIN_KM  # noqa: E402

    return CACHE, cluster_colours, kmeans, mo, np, plt, sweep


@app.cell
def _(CACHE, mo, np):
    mo.stop(
        not CACHE.exists(),
        mo.callout(
            mo.md(
                f"No embedding at `{CACHE}`. Train it first:\n\n"
                "`uv run python examples/seismic/scripts/seismic.py`"
            ),
            kind="warn",
        ),
    )
    cache = np.load(CACHE)
    embedding = cache["embedding"]
    rows_km = cache["rows_km"]
    cols_km = cache["cols_km"]
    mo.md(
        f"**{embedding.shape[0]:,} traces** on a {rows_km.size} x {cols_km.size} grid, "
        f"embedded in {embedding.shape[1]} dimensions. "
        f"Domain extent {rows_km[-1]:.1f} km x {cols_km[-1]:.1f} km."
    )
    return cols_km, embedding, rows_km


@app.cell
def _(mo):
    ks = mo.ui.multiselect(
        options=[2, 3, 4, 5, 6, 8, 10, 12, 16],
        value=[2, 3, 4, 6, 8],
        label="cluster counts",
    )
    seed = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="seed")
    fit_sample = mo.ui.number(
        start=1_000, stop=200_000, step=1_000, value=50_000, label="points to fit centres"
    )
    mo.hstack([ks, seed, fit_sample], justify="start", gap=2)
    return fit_sample, ks, seed


@app.cell
def _(embedding, fit_sample, ks, mo, seed, sweep):
    # One sweep over the selected k, each fit deterministic in `seed`: comparing k against k is the
    # point, so an unstable initialisation would confound that with the sweep itself.
    mo.stop(not ks.value, mo.md("*Pick at least one `k`.*"))
    results = sweep(
        embedding,
        sorted(ks.value),
        seed=int(seed.value),
        fit_sample=int(fit_sample.value),
    )
    return (results,)


@app.cell
def _(mo, np, plt, results):
    ks_sorted = sorted(results)
    _spread = [results[k][2] for k in ks_sorted]
    _figure, _axis = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
    _axis.plot(ks_sorted, _spread, marker="o")
    _axis.set_xlabel("k")
    _axis.set_ylabel("mean squared distance to centre")
    _axis.set_title("within-cluster spread")
    _axis.grid(alpha=0.3)
    if len(ks_sorted) > 1:
        # Where the curve bends hardest: the classic elbow, stated rather than eyeballed.
        _drops = np.diff(_spread) / np.diff(ks_sorted)
        _elbow = ks_sorted[int(np.argmax(np.abs(_drops)))]
        _axis.axvline(_elbow, linestyle=":", linewidth=1, alpha=0.6)
        _axis.annotate(
            f"steepest drop before k={_elbow}",
            xy=(0.02, 0.06),
            xycoords="axes fraction",
            fontsize=9,
        )
    mo.as_html(_figure)
    return


@app.cell
def _(cluster_colours, cols_km, mo, np, plt, results, rows_km):

    _ks = sorted(results)
    _columns = min(3, len(_ks))
    _grid_rows = int(np.ceil(len(_ks) / _columns))
    _figure, _axes = plt.subplots(
        _grid_rows, _columns, figsize=(5.0 * _columns, 4.3 * _grid_rows), constrained_layout=True
    )
    _axes = np.atleast_1d(_axes).ravel()
    for _axis, _k in zip(_axes, _ks, strict=False):
        _, labels, _spread = results[_k]
        _colours = cluster_colours(labels, _k).reshape(rows_km.size, cols_km.size, 4)
        _axis.pcolormesh(
            cols_km,
            rows_km,
            _colours,
            shading="nearest",
            linewidth=0,
            antialiased=False,
            rasterized=True,
        )
        _axis.set_aspect("equal")
        _axis.invert_yaxis()  # inline index increases downwards, as in a seismic section
        _counts = np.bincount(labels, minlength=_k)
        _axis.set_title(
            f"k={_k}, spread {_spread:.3f}\nsmallest cluster {_counts.min():,} traces",
            fontsize=9,
        )
        _axis.set_xticks([])
        _axis.set_yticks([])
    for _axis in _axes[len(_ks) :]:
        _axis.set_visible(False)
    _figure.suptitle("k-means clusters of the NOFE embedding, one colour per cluster")
    mo.as_html(_figure)
    return


@app.cell
def _(mo, np, results):
    _table = [
        {
            "k": k,
            "spread": round(_spread, 4),
            "smallest cluster": int(round(np.bincount(labels, minlength=k).min())),
            "largest cluster": int(round(np.bincount(labels, minlength=k).max())),
        }
        for k, (_, labels, _spread) in sorted(results.items())
    ]
    mo.vstack([mo.md("### Balances"), mo.ui.table(_table, selection=None)])
    return


@app.cell
def _(cols_km, mo, results, rows_km):
    """How much a cluster map is *spatially* coherent, which the spread alone does not say.

    Chance agreement between two neighbouring traces is ``1/k``; anything above that is structure
    the operator put there. The maps are speckled at small scale because the embedding is a function
    of the local trace neighbourhood, and that neighbourhood changes from trace to trace — this
    measures it instead of leaving the eye to guess.
    """
    _stats = []
    for _k, (_, _labels, _) in sorted(results.items()):
        _grid = _labels.reshape(rows_km.size, cols_km.size)
        _agreement = 0.5 * (
            float((_grid[:-1, :] == _grid[1:, :]).mean())
            + float((_grid[:, :-1] == _grid[:, 1:]).mean())
        )
        _stats.append(
            {
                "k": _k,
                "neighbour agreement": round(_agreement, 3),
                "chance (1/k)": round(1 / _k, 3),
                "x chance": round(_agreement * _k, 2),
            }
        )
    mo.vstack(
        [
            mo.md(
                "### Spatial coherence\n\n"
                "Share of horizontally adjacent trace pairs landing in the same cluster, against "
                "the chance rate `1/k`. Above chance means neighbouring traces cluster together."
            ),
            mo.ui.table(_stats, selection=None),
        ]
    )
    return


@app.cell
def _(embedding, kmeans, mo, results):
    # The centres of the largest k on screen, so the groups shown as colours are also readable as
    # positions in the embedding.
    _k_top = max(results)
    _centres, _labels, _ = kmeans(embedding, _k_top, seed=0)
    _lines = [
        f"- cluster {index}: "
        + ", ".join(f"{value:+.2f}" for value in centre)
        + f" — {int((_labels == index).sum()):,} traces"
        for index, centre in enumerate(_centres)
    ]
    mo.md(
        f"### Where the k={_k_top} cluster centres sit\n\n"
        "A centre is a position in the 3-D embedding, i.e. a typical *neighbourhood signature* of "
        "its group — not a seismic trace.\n\n" + "\n".join(_lines)
    )
    return


if __name__ == "__main__":
    app.run()
