import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Working with `nofe` on ERA5

        Three steps, each with the same shape: build graphs, train on them, embed another one.

        1. **Training data** — several random subsets of *one* ERA5 file.
        2. **Model and training** — `NOFE` fits one operator on all those subsets.
        3. **Inference** — the whole, separate ERA5 file is embedded by that operator and drawn.

        The split by file is the point: the model never sees the inference date.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    # Resolve paths from the repository layout, not the process cwd: VS Code and `marimo edit` can
    # start the kernel anywhere inside the checkout.
    _repo = next(
        (
            ancestor
            for ancestor in (Path.cwd(), *Path.cwd().parents)
            if (ancestor / "pyproject.toml").is_file()
        ),
        None,
    )
    if _repo is None:
        raise RuntimeError(f"run this notebook from inside the nofe checkout, not {Path.cwd()}")

    _era5 = _repo / "examples" / "era5"
    DATA = _era5 / "data"
    # Import the example's own helpers, so the loader and the globe plot have exactly one
    # implementation in the repo rather than a copy that drifts. `scripts/` must come first:
    # `import era5` there is the plotting module, while `examples/era5` is the directory.
    sys.path.insert(0, str(_repo / "examples"))
    sys.path.insert(0, str(_era5 / "utilities"))
    sys.path.insert(0, str(_era5 / "scripts"))

    import era5  # noqa: E402
    import torch  # noqa: E402
    from era5_graph import Era5Graph  # noqa: E402

    from nofe import NOFE, describe  # noqa: E402

    available = sorted(path.stem for path in DATA.glob("*.nc"))
    if len(available) < 2:
        raise RuntimeError(f"need two ERA5 dates in {DATA} to split training from inference")

    mo.md(f"**Backend** `{describe()}` · **dates on disk** {', '.join(available)}")
    return DATA, Era5Graph, NOFE, available, era5, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Which file is which

    Pick the training file; the *other* date becomes the inference field. The two are separate
    days, so the model cannot have memorised the inference grid.
    """)
    return


@app.cell
def _(available, mo):
    train_date = mo.ui.dropdown(options=available, value=available[0], label="training file")
    mo.hstack([train_date], justify="start")
    return (train_date,)


@app.cell
def _(available, mo, train_date):
    # Read `.value` only here: a cell cannot resolve its own widget's value while creating it.
    infer_date = next(date for date in available if date != train_date.value)
    mo.md(
        "| role | file | used for |\n|---|---|---|\n"
        f"| **train** | `{train_date.value}` | random subsets, and only these |\n"
        f"| **infer** | `{infer_date}` | the whole field, never trained on |"
    )
    return (infer_date,)


@app.cell
def _(mo):
    train_stride = mo.ui.slider(4, 16, value=8, step=4, label="training stride", show_value=True)
    subsets = mo.ui.slider(1, 5, value=3, step=1, label="number of subsets", show_value=True)
    subset_size = mo.ui.slider(
        2_000, 16_000, value=8_000, step=2_000, label="points per subset", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Training data: random subsets of one file

                Each subset is a random draw of `points per subset` grid points, with its own radius
                neighbourhood. Training sees several of them as a *list*, which is how `nofe` covers
                more of the domain than one graph would fit in memory.

                The stride thins the grid before sampling; the radius follows from it, because the
                neighbourhood has to stay the same physical size (`radius = 0.005454 x stride`, a
                chord on the unit sphere).
                """
            ),
            mo.hstack([train_stride, subsets, subset_size], justify="start", gap=2),
        ]
    )
    return subset_size, subsets, train_stride


@app.cell
def _(DATA, Era5Graph, mo, subset_size, subsets, train_date, train_stride):
    # One builder per subset, each with its own seed, so the draws differ from each other and stay
    # reproducible run to run.
    stride = int(train_stride.value)
    radius = 0.005454 * stride
    builders = [
        Era5Graph(
            radius=radius, stride=stride, data_dir=DATA, sample=int(subset_size.value), seed=seed
        )
        for seed in range(int(subsets.value))
    ]

    # The loader returns [lat, lon, variable] plus the two axes; `build` wants one row per point.
    train_features, train_lat, train_lon = builders[0].load(train_date.value)
    _flat = train_features.reshape(-1, train_features.shape[2])
    graphs = [builder.build(_flat, builder.positions(train_lat, train_lon)) for builder in builders]

    _rows = "\n".join(
        f"| {index} | {graph.num_nodes:,} | {graph.edge_index.shape[1]:,} |"
        for index, graph in enumerate(graphs)
    )
    mo.vstack(
        [
            mo.md("### The training graphs"),
            mo.md(
                f"`Era5Graph(radius={radius:.5f}, stride={stride}, "
                f"sample={int(subset_size.value):,})`, one builder per subset with its own seed. "
                f"All draws come from the same {_flat.shape[0]:,}-point grid "
                f"(`{train_features.shape[0]} x {train_features.shape[1]}`, "
                f"{train_features.shape[2]} variables per point):\n\n"
                "| subset | nodes | edges |\n|---|---|---|\n" + _rows
            ),
        ]
    )
    return graphs, radius


@app.cell
def _(graphs, mo, radius, torch):
    _graph = graphs[0]
    _counts = torch.bincount(_graph.edge_index[1], minlength=_graph.num_nodes)
    _pos_dims = _graph.pos.shape[1]

    _fields = [
        ("x", _graph.x.shape, "features, one row per node"),
        ("pos", _graph.pos.shape, "the locations the neighbour search used"),
        ("pos_raw", _graph.pos_raw.shape, "the caller's own locations, row-aligned"),
        ("edge_index", _graph.edge_index.shape, "`row 0` senders, `row 1` receivers"),
        ("edge_attr", _graph.edge_attr.shape, "`[pos[receiver], direction, rel. distance]`"),
        ("edge_dist", _graph.edge_dist.shape, "edge length in the scaled space"),
        ("affinity", _graph.affinity.shape, "the training target"),
    ]
    _table = "\n".join(f"| `{n}` | `{tuple(s)}` | {h} |" for n, s, h in _fields)

    mo.vstack(
        [
            mo.md("### What a training graph contains"),
            mo.md(
                f"Take subset 0 above. It has **{_graph.num_nodes:,} nodes** and "
                f"**{_graph.edge_index.shape[1]:,} edges** ({float(_counts.float().mean()):.2f} "
                f"incoming per node, radius {radius:.5f}):\n\n"
                "| field | shape | holds |\n|---|---|---|\n" + _table
            ),
            mo.md(
                f"`edge_attr` is `2P + 1 = {2 * _pos_dims + 1}` wide because the locations are "
                f"{_pos_dims}-D. `affinity` is the student-t weight of each neighbour normalised "
                "per receiver: it is what the loss compares the embedding against, so it must "
                "travel with the graph. `x`, `edge_index` and `edge_attr` are all the operator "
                "itself reads."
            ),
        ]
    )
    return


@app.cell
def _(NOFE, mo):
    model = NOFE(3, width=8, ker_width=32, nystrom=0.25, chunk_size=65_536)
    mo.vstack(
        [
            mo.md(
                "## 3. The model\n\n"
                f"`type(model.net)` is `{type(model.net).__name__}` — a real `torch` module. "
                "`in_channels` and `edge_channels` are not resolved yet; they come from the first "
                "graph the model sees, so the graph need not be described up front."
            ),
            mo.md(
                "| argument | here | why |\n|---|---|---|\n"
                "| `out_dim` | `3` | three numbers per node, so the map can be drawn as RGB |\n"
                "| `width` | `8` | hidden width; peak memory grows as `edges x width²` |\n"
                "| `ker_width` | `32` | hidden width of the edge-kernel MLP |\n"
                "| `nystrom` | `0.25` | landmark approximation, so a subset stays cheap to fit |\n"
                "| `chunk_size` | `65_536` | query rows per pass; memory only, never the result |"
            ),
        ]
    )
    return (model,)


@app.cell
def _(graphs, mo, model):
    # One graph or a list: a list returns one history per graph, and an epoch is one optimizer step
    # per graph. That is what makes this "train on several random samples" rather than on one.
    histories = model.training(graphs, epochs=40, patience=15, seed=0)

    _rows = "\n".join(
        f"| {index} | {len(history)} | {history[0]['train_loss']:.4f} | "
        f"{history[-1]['train_loss']:.4f} |"
        for index, history in enumerate(histories)
    )
    mo.vstack(
        [
            mo.md("### Train on the list of subsets"),
            mo.md(
                f"`model.training(graphs, ...)` trained on {len(graphs)} subset(s), "
                f"{len(histories[0])} epochs each:\n\n"
                "| subset | epochs | first train loss | last train loss |\n|---|---|---|---|\n"
                + _rows
            ),
            mo.md(
                f"`model.net.in_channels` is now **{model.net.in_channels}** and "
                f"`model.net.edge_channels` is **{model.net.edge_channels}** — the widths the "
                "first graph settled. The next graph must match them, which is the only "
                "requirement for embedding a different date."
            ),
        ]
    )
    return


@app.cell
def _(infer_date, mo, train_date):
    infer_stride = mo.ui.slider(
        1, 8, value=2, step=1, label="inference stride (1 = the full file)", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                f"## 4. Inference on the other file\n\n"
                f"Training used subsets of **`{train_date.value}`**. This section embeds "
                f"**`{infer_date}`**, whole: a date the model has never seen, with its own "
                "features and its own edges. Nothing is re-fitted — `embed` reads `x`, "
                "`edge_index` and `edge_attr` from whatever graph it is handed."
            ),
            mo.md(
                "The stride here is independent of the training stride: it sets how fine the map "
                "is drawn. Stride 1 is the whole 1,035,360-point field (~22M edges), which takes a "
                "while in a notebook; stride 2 is the default. The radius scales with it, keeping "
                "the neighbourhood the same physical size as in training."
            ),
            mo.hstack([infer_stride], justify="start"),
        ]
    )
    return (infer_stride,)


@app.cell
def _(DATA, Era5Graph, infer_date, infer_stride, mo, model, torch):
    infer_radius = 0.005454 * int(infer_stride.value)
    infer_builder = Era5Graph(radius=infer_radius, stride=int(infer_stride.value), data_dir=DATA)
    infer_features, infer_lat, infer_lon = infer_builder.load(infer_date)
    infer_graph = infer_builder.build(
        infer_features.reshape(-1, infer_features.shape[2]),
        infer_builder.positions(infer_lat, infer_lon),
    )
    infer_embedding = model.embed(infer_graph).numpy()

    _finite = bool(torch.isfinite(torch.as_tensor(infer_embedding)).all())
    mo.md(
        f"""
        ### The inference graph and its embedding

        | | |
        |---|---|
        | date | `{infer_date}` |
        | nodes | {infer_graph.num_nodes:,} |
        | edges | {infer_graph.edge_index.shape[1]:,} |
        | `x` width | {infer_graph.x.shape[1]} (same as training) |
        | `edge_attr` width | {infer_graph.edge_attr.shape[1]} (same `2P + 1`) |
        | embedding | `{tuple(infer_embedding.shape)}`, one row per node, all finite: {_finite} |
        """
    )
    return infer_embedding, infer_lat, infer_lon


@app.cell
def _(era5, infer_date, infer_embedding, infer_lat, infer_lon, mo):
    _figure = era5.globe_figure(
        infer_lat, infer_lon, infer_embedding, f"{infer_date}: embedded by the trained operator"
    )
    mo.as_html(_figure)
    return


@app.cell
def _(infer_date, infer_stride, mo, train_date):
    mo.md(f"""
    ### Summary

    | step | call | data |
    |---|---|---|
    | build | `builder.build(features, locations)` | subsets of `{train_date.value}` |
    | train | `model.training(graphs, ...)` | that list of subsets |
    | embed | `model.embed(graph)` | all of `{infer_date}`, stride {int(infer_stride.value)} |

    That is the whole workflow: one graph builder subclass, one `NOFE`, and `training` /
    `embed` for the two directions. `model.net.state_dict()` is the checkpoint if you want to
    keep the fitted operator.
    """)
    return


if __name__ == "__main__":
    app.run()
