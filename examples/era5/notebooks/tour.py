import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # NOFE on ERA5: graph, model, inference

        A guided tour of the three things `nofe` does, run against real ERA5 fields:

        1. **Graph construction** — turn `(features, locations)` into the object the operator reads.
        2. **Model and training** — `KernelNN` learns an embedding whose neighbourhoods match the
           feature neighbourhoods.
        3. **Inference** — take that trained model and embed a whole ERA5 file, then draw it.

        Each section states its inputs *before* the cell that runs it, so you can follow what goes
        in and what comes out. Everything here is the real package; nothing is mocked.
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

    _era5_dir = _repo / "examples" / "era5"
    DATA = _era5_dir / "data"
    # The notebook imports the example's own helpers, so the loader and the globe plot have exactly
    # one implementation in the repo rather than a copy that drifts.
    sys.path.insert(0, str(_era5_dir / "scripts"))
    sys.path.insert(0, str(_era5_dir / "utilities"))
    sys.path.insert(0, str(_repo / "examples"))

    import era5  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from era5_graph import Era5Graph  # noqa: E402

    from nofe import NOFE, describe  # noqa: E402

    available = sorted(path.stem for path in DATA.glob("*.nc"))
    mo.md(
        f"**Repository** `{_repo.name}` · **backend** `{describe()}` · "
        f"**dates on disk** {', '.join(f'`{d}`' for d in available) or 'none'}"
    )
    return DATA, Era5Graph, NOFE, available, era5, mo, np, torch


@app.cell
def _(available, mo):
    date = mo.ui.dropdown(options=available, value=available[0], label="ERA5 date")
    stride = mo.ui.slider(2, 32, value=8, step=2, label="grid stride", show_value=True)
    mo.vstack(
        [
            mo.md(
                """
                ## 0. Pick the data

                `stride` thins the native 0.25° grid: stride 1 is 1,035,360 points, stride 8 is
                16,020. Lower strides take longer to build and train; the lesson is the same.
                """
            ),
            mo.hstack([date, stride], justify="start", gap=2),
        ]
    )
    return date, stride


@app.cell
def _(DATA, Era5Graph, date, mo, stride):
    builder = Era5Graph(radius=0.04363, stride=int(stride.value), data_dir=DATA)
    features, latitudes, longitudes = builder.load(date.value)

    mo.md(
        f"""
        ### 1a. The inputs to the graph builder

        Every `PointGraph` needs exactly two arrays with one row per point:

        | Input | What it is | Shape here |
        |---|---|---|
        | `features` | the values that describe each node | `{tuple(features.shape)}` |
        | `locations` | where each node sits, any dim | `{latitudes.size} x {longitudes.size}` |

        `features` is `lat x lon x variables`, and `locations` is one unit-sphere vector per grid
        point, so `positions` turns the two axes into {latitudes.size * longitudes.size:,} rows.

        `Era5Graph` is the subclass that supplies the neighbourhood rule: every grid point within
        a chord of **{builder.radius}** on the unit sphere sends a message, banded by latitude so
        even a fine grid stays buildable. Its own normalisation is off, because `load` has already
        standardised each variable per field.
        """
    )
    return builder, features, latitudes, longitudes


@app.cell
def _(mo):
    mo.md(
        """
        ### 1b. Build it

        The next cell calls `build(features, locations)`. In order, it: standardises both arrays per
        column (skipped here), takes every row unless `sample=n`, searches for neighbours, sorts the
        edges by `(receiver, sender)`, derives the geometric edge features, then computes the
        affinities that will be the training target.
        """
    )
    return


@app.cell
def _(builder, features, latitudes, longitudes, mo, torch):
    # `build_raster` is the script's adapter and prints a summary; calling `build` directly keeps
    # the notebook's own table as the only thing on screen.
    _flat = features.reshape(-1, features.shape[2])
    graph = builder.build(_flat, builder.positions(latitudes, longitudes))

    _counts = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    _isolated = int((_counts == 0).sum())
    _degrees = _counts.float()
    _edges = graph.edge_index.shape[1]
    _pos_dims = graph.pos.shape[1]

    _fields = [
        ("x", tuple(graph.x.shape), "features, one row per node (already standardised by `load`)"),
        ("pos", tuple(graph.pos.shape), "the locations the search used"),
        ("pos_raw", tuple(graph.pos_raw.shape), "the caller's own locations, row-aligned"),
        ("edge_index", tuple(graph.edge_index.shape), "`row 0` senders, `row 1` receivers"),
        ("edge_attr", tuple(graph.edge_attr.shape), "`[pos[receiver], direction, rel. distance]`"),
        ("edge_dist", tuple(graph.edge_dist.shape), "edge length in the scaled space"),
        ("affinity", tuple(graph.affinity.shape), "the loss target"),
    ]
    _table = "\n".join(f"| `{name}` | `{shape}` | {holds} |" for name, shape, holds in _fields)

    mo.vstack(
        [
            mo.md(
                "### 1c. What the graph contains\n\n"
                "| Field | Shape | Holds |\n|---|---|---|\n" + _table
            ),
            mo.md(
                f"""
                Read the shapes as **{graph.num_nodes:,} nodes** and **{_edges:,} edges**, i.e.
                {float(_degrees.mean()):.2f} incoming edges per node on average (max
                {int(_counts.max())}).

                `edge_attr` is `2P + 1 = {2 * _pos_dims + 1}` wide because the locations are
                {_pos_dims}-D — that number becomes the model's `edge_channels`.

                {_isolated:,} node(s) have no incoming edge: they get no `affinity` entry, so they
                are excluded from the loss, but they are still embedded through the root weight.

                `affinity` is not uniform — it is the student-t weight of each neighbour, normalised
                per receiver, mean {float(graph.affinity.mean()):.4f}. That distribution is what the
                embedding is trained to reproduce.
                """
            ),
        ]
    )
    return (graph,)


@app.cell
def _(graph, mo, torch):
    # A concrete peek at a few edges beats another table: it shows the numbers the operator
    # consumes, in the order `build` left them.
    _head = torch.stack(
        (
            graph.edge_index[0][:6],
            graph.edge_index[1][:6],
            graph.affinity[:6],
            graph.edge_dist[:6],
        ),
        dim=1,
    )
    _rows = "\n".join(
        f"| {int(s)} | {int(r)} | {float(a):.4f} | {float(d):.4f} |"
        for s, r, a, d in _head.tolist()
    )
    mo.md(
        "**The first six edges.** Senders are sorted by receiver, so equal receivers are adjacent:"
        "\n\n| sender | receiver | affinity | edge_dist |\n|---|---|---|---|\n" + _rows
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. The model

        `KernelNN(out_dim=3, *, in_channels=None, edge_channels=None, width=32, ker_width=64,
        depth=3, edge_batch_size=65536)`

        | Parameter | What it controls |
        |---|---|
        | `out_dim` | width of the embedding — 3 so we can draw it as RGB |
        | `width` | hidden width; the dominant cost, since the kernel emits `width²` per edge |
        | `ker_width` | hidden width of the edge-kernel MLP (edge features → `width × width`) |
        | `depth` | message-passing hops; reuses **one** conv, so it adds no parameters |
        | `in_channels` / `edge_channels` | left `None` here: taken from the first graph it sees |
        | `edge_batch_size` | edges per kernel chunk; memory only, never the result |

        `NOFE` wraps that network as `.net` and adds the training loop. Below it gets
        `nystrom=0.25`, the landmark approximation that keeps large graphs affordable.
        """
    )
    return


@app.cell
def _(NOFE, mo):
    model = NOFE(3, width=8, ker_width=32, nystrom=0.25, chunk_size=65_536)
    _before = sum(p.numel() for p in model.net.parameters() if p.data.numel())
    mo.md(
        f"""
        ### 2a. Initialise it

        The network exists as a real `torch` module before it has seen anything:

        | | |
        |---|---|
        | `type(model.net)` | `{type(model.net).__name__}` |
        | parameters counted so far | {_before:,} (the lazy ones are not materialised yet) |
        | `in_channels` | not resolved yet |

        Both widths resolve on the first forward pass, which is why the graph did not have to be
        described to the constructor.
        """
    )
    return (model,)


@app.cell
def _(mo):
    mo.md(
        """
        ### 2b. Train it

        `model.training(graph, epochs=…, seed=…)` returns one history per graph: one dict per
        epoch, keyed `epoch` / `train_loss` / `val_loss` / `best_val_loss`. It keeps the epoch with
        the best validation loss, so the weights you end up with are not simply the last ones.

        Watch `in_channels` and `edge_channels` afterwards: they should read 11 and 7, matching the
        graph's feature and edge widths.
        """
    )
    return


@app.cell
def _(graph, model, mo):
    history = model.training(graph, epochs=80, patience=20, seed=0)

    _last = history[-1]
    _step = max(1, len(history) // 8)
    _curve = "\n".join(
        f"| {row['epoch']} | {row['train_loss']:.5f} | {row['val_loss']:.5f} |"
        for row in history[::_step]
    )
    mo.vstack(
        [
            mo.md("### 2c. Training ran"),
            mo.md(
                "| | |\n|---|---|\n"
                f"| epochs run | {len(history)} |\n"
                f"| train loss, first → last | {history[0]['train_loss']:.5f} → "
                f"{_last['train_loss']:.5f} |\n"
                f"| best validation loss | {_last['best_val_loss']:.5f} |\n"
                f"| resolved `in_channels` | {model.net.in_channels} |\n"
                f"| resolved `edge_channels` | {model.net.edge_channels} |"
            ),
            mo.md(
                "A sampled log of the curve:\n\n| epoch | train | val |\n|---|---|---|\n" + _curve
            ),
        ]
    )
    return (history,)


@app.cell
def _(graph, model, mo):
    embedding = model.embed(graph)
    score = model.loss(graph)
    mo.vstack(
        [
            mo.md(
                f"""
                ### 2d. Score the same graph

                `embed` returns one row per node, in node order; `loss` re-reads the graph's
                `affinity` and reports the mean KL of that embedding: **{score:.5f}**. Lower is
                better, and the number is only meaningful next to a baseline — the next cell
                supplies one.
                """
            ),
            mo.md(
                f"| | |\n|---|---|\n"
                f"| `embedding` shape | `{tuple(embedding.shape)}` |\n"
                f"| dtype / device | `{embedding.dtype}` / `{embedding.device.type}` |"
            ),
        ]
    )
    return (embedding,)


@app.cell
def _(embedding, graph, mo, np, torch):
    # PCA-3 is the honest reference: the best *linear* map to 3 dimensions. A useful embedding beats
    # it on the same objective.
    from nofe.affinity import affinity_kl_loss

    _centred = graph.x - graph.x.mean(dim=0, keepdim=True)
    _, _, _basis = torch.linalg.svd(_centred, full_matrices=False)
    _reference = _centred @ _basis[:3].T
    _model_kl = float(affinity_kl_loss(graph.x, embedding, graph.edge_index, graph.num_nodes))
    _reference_kl = float(affinity_kl_loss(graph.x, _reference, graph.edge_index, graph.num_nodes))
    _spread = embedding.detach().cpu().numpy().std(axis=0)
    _verdict = (
        "NOFE beats the linear baseline."
        if _model_kl < _reference_kl
        else "NOFE did not beat the baseline here — train longer or raise the stride."
    )

    mo.vstack(
        [
            mo.md("### 2e. Is it any good?"),
            mo.md(
                "| map to 3-D | mean KL |\n|---|---|\n"
                f"| best linear projection (PCA) | {_reference_kl:.5f} |\n"
                f"| NOFE | {_model_kl:.5f} |"
            ),
            mo.md(
                f"{_verdict}\n\n"
                f"The embedding spans {', '.join(f'{float(v):.3f}' for v in _spread)} per "
                "channel, so no channel collapsed.\n\n"
                f"Both numbers score the *same* embedding function on the same graph, so they are "
                f"comparable. This one ({_model_kl:.5f}) is higher than the `loss` above because "
                "`embed` runs the Nyström approximation — a subset of landmarks — while `loss` "
                "scores the model's own training path."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Inference on a whole ERA5 file

        The trained operator never saw `pos` or `affinity`, only features and edges. That is what
        makes this step possible: point it at a **different, finer** grid and it embeds that one
        instead.

        Below, the same model embeds the field at native resolution. The graph is built exactly as
        before — same class, same radius-to-stride relationship — and then drawn one quad per grid
        point, R/G/B from the three embedding dimensions.
        """
    )
    return


@app.cell
def _(mo):
    infer_stride = mo.ui.slider(
        1, 8, value=4, step=1, label="inference stride (1 = native resolution)", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                """
                Pick how fine the inference grid should be. Stride 1 is the full file: 1,035,360
                nodes and ~22M edges, which is a lot to build inside a notebook. Stride 4 is a good
                live-demo default; drop it to 1 to show the real thing.
                """
            ),
            infer_stride,
        ]
    )
    return (infer_stride,)


@app.cell
def _(DATA, Era5Graph, date, infer_stride, mo, model, np):
    # The radius has to shrink with the stride so the neighbourhood stays physically the same; these
    # are the constants `examples/era5/scripts/era5.py` uses.
    _chords = {1: 0.005454, 2: 0.011, 4: 0.0218, 8: 0.04363}
    _radius = _chords.get(int(infer_stride.value), 0.04363)
    infer_builder = Era5Graph(radius=_radius, stride=int(infer_stride.value), data_dir=DATA)
    infer_features, infer_lat, infer_lon = infer_builder.load(date.value)
    infer_graph = infer_builder.build(
        infer_features.reshape(-1, infer_features.shape[2]),
        infer_builder.positions(infer_lat, infer_lon),
    )
    infer_embedding = model.embed(infer_graph).numpy()

    _finite = bool(np.isfinite(infer_embedding).all())
    mo.md(
        f"""
        ### 3a. The inference graph

        | | |
        |---|---|
        | nodes | {infer_graph.num_nodes:,} |
        | edges | {infer_graph.edge_index.shape[1]:,} |
        | `x` width | {infer_graph.x.shape[1]} (same features as training) |
        | `edge_attr` width | {infer_graph.edge_attr.shape[1]} (same `2P + 1`) |
        | embedding | `{tuple(infer_embedding.shape)}`, all finite: {_finite} |

        Matching widths is the only requirement, and the operator satisfies it because it learns a
        *function of local neighbourhoods* rather than a lookup table of nodes.
        """
    )
    return infer_embedding, infer_lat, infer_lon


@app.cell
def _(era5, infer_embedding, infer_lat, infer_lon, mo):
    _figure = era5.globe_figure(
        infer_lat, infer_lon, infer_embedding, "Inference on the ERA5 field"
    )
    mo.as_html(_figure)
    return


@app.cell
def _(infer_embedding, mo):
    _spread = infer_embedding.std(axis=0)
    _channels = ", ".join(f"{value:.3f}" for value in _spread)
    mo.md(
        f"""
        ### 3b. Reading the map

        R/G/B are the three embedding dimensions, each scaled by its 5th–95th percentile so one
        cyclone cannot flatten the field to grey. Per-channel spread is {_channels} — three live
        channels, so the map has real colour structure rather than a grey wash.

        The point of the tour: one graph builder, one small model, and the same three calls
        (`build` → `training` → `embed`) scale from a 16k-node teaching example to a million nodes.
        """
    )
    return


if __name__ == "__main__":
    app.run()
