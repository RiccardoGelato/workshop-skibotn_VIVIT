"""Generate the workflow notebooks.

Run from the repository root: ``uv run python notebooks/_build.py``.
Notebook sources live here so the JSON is reproducible; ``notebooks/_execute.py`` then executes
every notebook to prove the cells run and to embed their outputs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
SHARED = """\
%matplotlib inline

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from nofe import NOFE, build_graph
from nofe.affinity import affinity_kl_loss

# Resolve paths from the repository layout, not the process cwd: VS Code always starts kernels in
# the workspace folder (`jupyter.notebookFileRoot` is ignored since 2024.4), while `_execute.py`
# runs from `notebooks/`. Walk up from the cwd looking for the repo root, so both agree.
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
NOTEBOOKS = REPO / "notebooks"
DATA = NOTEBOOKS / "data"

# The example data is not in Git (21 MB per date), so fail with the fetch command rather than the
# opaque `StopIteration`/`FileNotFoundError` the loader would otherwise raise.
if not (DATA / "2018-01-01.nc").exists():
    raise FileNotFoundError(
        f"missing {DATA / '2018-01-01.nc'}\\n"
        "The ERA5 inputs are not committed. Fetch them once with:\\n"
        f"  mkdir -p {DATA}\\n"
        f"  cp /Volumes/T7-storage/ERA5/data/2018-01-0{{1,2}}.nc {DATA}/"
    )

# The notebooks import the ERA5 helpers straight from the example script, so the loader and the
# globe plot have exactly one implementation in the repo. `era5` is imported after `sys.path` is
# extended, which is why the noqa marker below is needed.
sys.path.insert(0, str(REPO / "examples"))
import era5  # noqa: E402


def kl(on, embedding) -> float:
    \"\"\"Mean KL between a graph's own feature affinities and an embedding of its rows.\"\"\"
    return float(affinity_kl_loss(on.x, embedding, on.edge_index, on.num_nodes))


def pca_baseline(data, dims: int = 3) -> torch.Tensor:
    \"\"\"Best linear projection to ``dims`` dimensions: the reference the model has to beat.\"\"\"
    centred = data.x - data.x.mean(dim=0, keepdim=True)
    _, _, basis = torch.linalg.svd(centred, full_matrices=False)
    return centred @ basis[:dims].T


def knn_agreement(reference: torch.Tensor, candidate: torch.Tensor, k: int = 10) -> float:
    \"\"\"Mean share of exact k-nearest neighbours two point sets have in common.\"\"\"
    ref = reference.detach().cpu().numpy()
    cand = candidate.detach().cpu().numpy()
    _, ref_nn = cKDTree(ref).query(ref, k=k + 1)
    _, cand_nn = cKDTree(cand).query(cand, k=k + 1)
    shared = [len(set(a) & set(b)) / k for a, b in zip(ref_nn[:, 1:], cand_nn[:, 1:], strict=True)]
    return float(np.mean(shared))


def close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4) -> bool:
    \"\"\"Relative agreement in units of the embedding's own scale.\"\"\"
    return bool((a - b).abs().max() <= rtol * a.abs().max())


print("nofe ready; the faiss fallback warning is expected on macOS arm64")
"""

LOAD = """\
# STRIDE=8 is the whole globe at 89 x 180 = 16,020 grid points. Lower the stride for a finer
# grid (4 gives 64,440 nodes, 2 gives the full 258,480-node field).
STRIDE = 8
features, latitudes, longitudes = era5.load_fields(DATA / "2018-01-01.nc", stride=STRIDE)
positions = era5.unit_sphere(latitudes, longitudes)
flat = features.reshape(-1, features.shape[2])
print(
    f"grid {features.shape[0]}x{features.shape[1]} -> {positions.shape[0]} nodes, "
    f"{flat.shape[1]} variables each"
)
"""

CONFIG = """\
CONFIG = {
    "width": 8,
    "ker_width": 32,
    "epochs": 200,
    "seed": 0,
    "early_stopping": False,
    "verbose": True,
    "log_every": 50,
}"""

CONFIG_DOC = """\
`width=8, ker_width=32` are the reference ERA5 settings: `NNConv` materialises an
`[num_edges, width]` message tensor, so width drives memory. 200 epochs at this resolution takes
well under a minute on an M-series GPU; `verbose=True, log_every=50` prints progress and the final
epoch always prints."""

METRIC_DOC = """\
### How to read the score

The objective is the KL between the graph's feature affinities and the embedding's. It only ranks
a node's 10 graph neighbours relative to each other, so **exact** 10-NN set overlap is a weak proxy:
the 11 variables are close to isotropic (the top three principal components carry only 54% of the
variance), so no 3-D map can reproduce the 11-D neighbour sets exactly.

`pca_baseline` gives the honest reference — the best *linear* projection to 3 dimensions. A useful
embedding beats it on the objective."""


def notebook(cells: list) -> nbf.NotebookNode:
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


# 1 --------------------------------------------------------------------------------------
in_sample = notebook(
    [
        new_markdown_cell(
            """\
# 1 — Everything observed: train and infer on one graph

The simplest workflow. Every grid point has features, so nothing is withheld:

- `build_graph(positions, features)` returns a `Data` with **no** `input_mask`, `output_mask` or
  `y`.
- `train` scores every row. The internal train/validation split chooses which rows enter the
  objective; it never hides neighbours.
- `embed` returns exactly `num_nodes` rows.

This is the baseline the other two notebooks build on."""
        ),
        new_code_cell(SHARED),
        new_markdown_cell(
            """\
## Load one date at coarse resolution

`era5.load_fields` standardises each variable per field and drops the two duplicated pole rows;
`era5.unit_sphere` turns lat/lon into 3-D unit vectors, so the ±180° seam and the poles are
ordinary neighbours for the kNN graph."""
        ),
        new_code_cell(LOAD),
        new_markdown_cell(
            """\
## Build the graph

`neighbor="knn", k=10` matches the ERA5 example. Edges are built once, here; the operator only
consumes `edge_index`/`edge_attr`, so a radius neighbourhood works identically
(`neighbor="radius", radius=0.05` on the unit sphere)."""
        ),
        new_code_cell(
            """\
data = build_graph(positions, flat, neighbor="knn", k=era5.K)
print(data)
print(
    "masks:",
    getattr(data, "input_mask", None),
    getattr(data, "output_mask", None),
    "| truth in y:",
    getattr(data, "y", None) is not None,
)
print(
    "every node both sends and receives:",
    set(data.edge_index[0].tolist()) == set(range(data.num_nodes)),
)
"""
        ),
        new_markdown_cell("## Train\n\n" + CONFIG_DOC),
        new_code_cell(
            CONFIG
            + """

model = NOFE(dim=3, config=CONFIG)
history = model.train(data)
print(
    f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} "
    f"| best val {model.best_val_loss:.6f}"
)
"""
        ),
        new_markdown_cell(
            """\
## Infer on the same data

`embed` returns one row per `output_mask` node — here every node — in node order (lat-major,
matching the raster). `loss` is the mean KL over the same rows."""
        ),
        new_code_cell(
            """\
embedding = model.embed(data)
print(
    "embedding:",
    tuple(embedding.shape),
    "| device:",
    embedding.device.type,
    "| dtype:",
    embedding.dtype,
)
print(f"mean KL on the training graph: {model.loss(data):.6f}")
assert embedding.shape == (data.num_nodes, 3)
assert torch.isfinite(embedding).all()
"""
        ),
        new_markdown_cell("## Compare against the linear baseline\n\n" + METRIC_DOC),
        new_code_cell(
            """\
reference = pca_baseline(data, dims=3)
print(f"PCA-3 KL: {kl(data, reference):.6f}  |  10-NN {knn_agreement(data.x, reference):.3f}")
print(f"NOFE  KL: {kl(data, embedding):.6f}  |  10-NN {knn_agreement(data.x, embedding):.3f}")
assert kl(data, embedding) < kl(data, reference), "the model should beat linear PCA"
print("the trained operator beats the best linear projection: True")
"""
        ),
        new_markdown_cell(
            """\
## Plot the embedding as an RGB globe

R/G/B are the three embedding dimensions, one `pcolormesh` quad per grid point. Each channel is
centred and scaled by its 5th/95th percentile spread (`era5.PERCENTILE`), so one extreme cyclone
cannot flatten the whole field to grey."""
        ),
        new_code_cell(
            """\
import matplotlib.pyplot as plt

figure = era5.globe_figure(
    latitudes, longitudes, embedding.numpy(), f"2018-01-01, stride {STRIDE}: in-sample"
)
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Save and reload

Checkpoints are self-describing (`nofe-checkpoint-2`): dim, config, model config, weights and the
best validation loss. Reloading reproduces the embedding to floating-point tolerance.

On MPS the comparison is *not* bit-exact: repeating the same forward pass twice already differs by
~4e-6 (measured here), and reloading adds ~6e-6, on embeddings whose magnitude is ~20. That is
Metal kernel scheduling, not a checkpoint defect — `device="cpu"` reproduces bit-identically. The
assertion below uses a relative tolerance for that reason."""
        ),
        new_code_cell(
            """\
path = model.save("in_sample.pt")
restored = NOFE.load(path)
rebuilt = restored.embed(data)
print(
    f"restore max abs diff: {(rebuilt - embedding).abs().max():.2e}"
    f" on a scale of {embedding.abs().max():.1f}"
)
assert close(embedding, rebuilt), "reload must reproduce the embedding"
print("round trip reproducible within relative tolerance: True")
path.unlink()
"""
        ),
        new_markdown_cell(
            """\
## Takeaways

|Fact|Why it matters|
|---|---|
|No `input_mask`, no `y`|A fully observed graph needs neither; `embed` covers every node|
|`embed` returns `num_nodes` rows|Default `output_mask` is every node|
|One graph, one pass|The dense operator sees the entire edge set on every epoch|

Next: **2-held-out-training.ipynb** withholds part of the same date and trains on a merged graph."""
        ),
    ]
)

# 2 --------------------------------------------------------------------------------------
held_out = notebook(
    [
        new_markdown_cell(
            """\
# 2 — Withhold locations, train on a merged graph, predict two ways

Out-of-sample embedding: some locations have no features. `build_graph` merges **locations, not
features** — the node set is `positions ∪ query_positions`, the added rows get a `fill` substitute
in `x`, keep their truth in `y`, and receive messages without sending any (`input_mask=False`).

Three graphs come out of one date:

|Graph|Built with|What it can do|
|---|---|---|
|`observed`|`build_graph(p, f)` on the retained rows|in-sample baseline|
|`merged`|`+ query_positions=q, query_features=fq`|train with truth for both parts|
|`blind`|`+ query_positions=q` only|true out-of-sample: `embed` only|

The last sections train on `merged` and predict twice on the **same coarse graph**: once for the
observed rows only, once for every location."""
        ),
        new_code_cell(SHARED),
        new_code_cell(LOAD),
        new_markdown_cell(
            """\
## Hold out a random subset of locations

A fixed fraction of grid points is withheld at random, mimicking scattered missing sensors. Any
subset works: the caller decides what to withhold, and there is no holdout fraction anywhere in the
API. The seed is fixed so the split is reproducible.

Random selection spreads the withheld points across the whole globe, which is a fairer test of
out-of-sample prediction than a contiguous gap: nearly every withheld point has observed
neighbours right beside it, rather than sitting in the middle of a large empty region."""
        ),
        new_code_cell(
            """\
def split_nodes(num_nodes: int, fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"Boolean masks: withhold ``fraction`` of the locations, chosen uniformly at random.\"\"\"
    rng = np.random.default_rng(seed)
    withheld = rng.random(num_nodes) < fraction
    return ~withheld, withheld


HELD_OUT = 0.2
observed_mask, withheld_mask = split_nodes(positions.shape[0], HELD_OUT)
num_observed = int(observed_mask.sum())
print(
    f"observed {num_observed}  withheld {int(withheld_mask.sum())}  "
    f"of {observed_mask.size} locations ({HELD_OUT:.0%} held out)"
)
assert withheld_mask.any() and observed_mask.any()
"""
        ),
        new_markdown_cell(
            """\
## Build the three graphs

`observed` holds the retained rows only. `merged` adds the withheld locations and keeps their
truth in `y`. `blind` is the same node set without `y`, which is exactly what inference looks like
when no truth exists.

`merged` and `blind` are bit-identical apart from `y`: same `x`, `pos`, `input_mask` and
`edge_index`. That is the training/inference parity guarantee."""
        ),
        new_code_cell(
            """\
observed = build_graph(positions[observed_mask], flat[observed_mask], neighbor="knn", k=era5.K)
merged = build_graph(
    positions[observed_mask],
    flat[observed_mask],
    query_positions=positions[withheld_mask],
    query_features=flat[withheld_mask],
    neighbor="knn",
    k=era5.K,
)
blind = build_graph(
    positions[observed_mask],
    flat[observed_mask],
    query_positions=positions[withheld_mask],
    neighbor="knn",
    k=era5.K,
)

for name, graph in (("observed", observed), ("merged", merged), ("blind", blind)):
    truth = "yes" if getattr(graph, "y", None) is not None else "no"
    print(f"{name:9s} nodes {graph.num_nodes:6d}  edges {graph.num_edges:7d}  y {truth}")

print(
    "merged/blind differ only in y:",
    torch.equal(merged.x, blind.x) and torch.equal(merged.edge_index, blind.edge_index),
)
# `merged` orders nodes as observed-first then added, so senders are the prefix 0..num_observed-1.
print(
    "added rows receive only:",
    set(merged.edge_index[0].tolist()) == set(range(num_observed)),
    "| every node receives:",
    set(merged.edge_index[1].tolist()) == set(range(merged.num_nodes)),
)
"""
        ),
        new_markdown_cell(
            """\
## The substitute is not the truth

`x` on the added rows holds the inverse-distance interpolation of the `k` nearest observed rows.
The real features live only in `y` and never enter the model input. Swap `fill=` to compare the
three strategies."""
        ),
        new_code_cell(
            """\
added = slice(num_observed, None)
print("added rows hold the fill substitute:", not torch.equal(merged.x[added], merged.y[added]))
print(f"substitute max abs error vs truth: {(merged.x[added] - merged.y[added]).abs().max():.4f}")
print(
    "observed rows are exact:",
    torch.equal(merged.x[:num_observed], merged.y[:num_observed]),
)
for mode in ("interpolate", "zeros", "mean"):
    candidate = build_graph(
        positions[observed_mask],
        flat[observed_mask],
        query_positions=positions[withheld_mask],
        fill=mode,
    )
    delta = (candidate.x[added] - merged.y[added]).abs().mean()
    print(f"fill={mode:12s} mean abs error on the added rows: {delta:.4f}")
"""
        ),
        new_markdown_cell(
            """\
## Train on the merged graph

The loss is the KL divergence against `y` — real features for **both** the observed and withheld
rows — while the model only ever sees the substitute on the withheld ones.

`blind` is refused by `train`/`loss`: its scored rows hold substitutes and there is no truth to
score them against. `embed` still works on it.

"""
            + CONFIG_DOC
        ),
        new_code_cell(
            CONFIG
            + """

model = NOFE(dim=3, config=CONFIG)
history = model.train(merged)
print(
    f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} "
    f"| mean KL on merged {model.loss(merged):.6f}"
)

try:
    model.train(blind)
except ValueError as error:
    print("blind graph refused, as designed:", str(error)[:78], "...")

reference = pca_baseline(merged, dims=3)
print(f"PCA-3 KL on the same graph: {kl(merged, reference):.6f} (model must beat it)")
"""
        ),
        new_markdown_cell(
            """\
## Two predictions on the same coarse graph

Both calls embed the graph the model was trained on and differ only in `output_mask`:

- **all locations** — `merged` as built: every node is scored, the withheld points included.
- **observed only** — clone `merged` and clear `output_mask` on the withheld rows. The graph, its
  edges and the substitute in `x` are untouched, so the model sees exactly the same input; only
  which rows come back changes.

`validate_graph` enforces `input_mask ⊆ output_mask`, so a restriction may only narrow to a
superset of the observed rows. Rows come back in ascending node order."""
        ),
        new_code_cell(
            """\
all_locations = model.embed(merged)
assert all_locations.shape == (merged.num_nodes, 3)

# `merged` reorders nodes: observed rows first, then the added ones. Its own `input_mask` already
# marks exactly those rows, so it is the restriction — a mask in original grid order would not be.
observed_only = merged.clone()
observed_only.output_mask = merged.input_mask
observed_prediction = model.embed(observed_only)
assert observed_prediction.shape == (num_observed, 3)

print("all locations :", tuple(all_locations.shape))
print("observed only :", tuple(observed_prediction.shape))
max_diff = (observed_prediction - all_locations[:num_observed]).abs().max()
print(f"max diff vs the matching prefix: {max_diff:.2e} (MPS re-runs differ at this order)")
assert close(observed_prediction, all_locations[:num_observed])
print("observed-only rows are the matching prefix of the full prediction: True")
"""
        ),
        new_markdown_cell(
            """\
## Score both predictions

`loss` scores the `output_mask` rows and reads the truth from `y`. The full view includes the
withheld points, so the difference between the two numbers is the out-of-sample cost.

The contract requires `input_mask ⊆ output_mask`, so the withheld points cannot be scored *alone*:
observed-rows-only and every-row are the two supported views."""
        ),
        new_code_cell(
            """\
all_kl = model.loss(merged)
observed_kl = model.loss(observed_only)
print(f"KL on all rows      : {all_kl:.6f}")
print(f"KL on observed rows : {observed_kl:.6f}")
print(f"out-of-sample penalty from the withheld points: {all_kl - observed_kl:+.6f}")
assert all_kl < kl(merged, reference), "the model should beat linear PCA on the merged graph"
"""
        ),
        new_markdown_cell(
            """\
## Visualise all locations vs observed only

The panels differ **only** in which rows are drawn, so they share one colour scale — computed
from the full prediction — and identical nodes get identical colours in both.

Top: every location, including the withheld ones (the added rows receive messages from their
observed neighbours). Bottom: the observed rows only, with the withheld cells left blank. That
panel is not the model's output for those rows, it is the same embedding with the unobserved cells
hidden, so the scattered points the model had to extrapolate stand out."""
        ),
        new_code_cell(
            """\
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

grid_index = np.empty(merged.num_nodes, dtype=np.int64)
grid_index[observed_mask] = np.arange(num_observed)
grid_index[withheld_mask] = np.arange(num_observed, merged.num_nodes)


def to_grid(graph_rows: torch.Tensor) -> np.ndarray:
    \"\"\"Graph-order embedding rows back into the original lat/lon raster order.\"\"\"
    values = np.full((grid_index.size, 3), np.nan, dtype=np.float32)
    values[grid_index] = graph_rows.detach().cpu().numpy()
    return values


full_grid = to_grid(all_locations)
observed_grid = np.full_like(full_grid, np.nan)
observed_grid[observed_mask] = full_grid[observed_mask]
print(
    "blank cells — full:",
    int(np.isnan(full_grid).any(axis=1).sum()),
    "| observed only:",
    int(np.isnan(observed_grid).any(axis=1).sum()),
)

# One scale from the full prediction, so a given node has the same colour in both panels.
scale = era5.colour_scale(full_grid, era5.PERCENTILE)

figure, axes = plt.subplots(
    2, 1, figsize=(12.0, 11.0), subplot_kw={"projection": ccrs.PlateCarree()}
)
for axis, (title, values, shown) in zip(
    axes,
    (
        ("all locations predicted", full_grid, merged.num_nodes),
        ("observed rows only", observed_grid, num_observed),
    ),
    strict=True,
):
    axis.set_global()
    era5.draw_globe(axis, latitudes, longitudes, values, scale=scale)
    axis.coastlines(linewidth=0.5)
    axis.set_title(f"{title} ({shown} of {merged.num_nodes} nodes)")
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Takeaways

|Fact|Why it matters|
|---|---|
|Merging unions locations|Added rows are unknown-featured either way|
|`y` holds the truth|The model never sees it as input; only the objective does|
|`input_mask` is provenance|Picks landmarks, senders and scorability; never masks the forward pass|
|`output_mask` chooses the prediction rows|Same graph, same inputs, different returned subset|

Next: **3-transfer-between-dates.ipynb** trains on this date and predicts on another."""
        ),
    ]
)

# 3 --------------------------------------------------------------------------------------
transfer = notebook(
    [
        new_markdown_cell(
            """\
# 3 — Train on one date, predict on another

A graph is just data: `train` and `embed` never assume the inference graph is the training one.
This notebook trains on **2018-01-01** and embeds **2018-01-02**, whose network is a different
`Data` object with its own edges.

Two inference inputs are shown:

1. **fully observed** — the new date is complete, so it embeds `num_nodes` rows;
2. **partly observed** — half its locations are withheld and supplied as `query_positions`, which
   is what deployment looks like when new locations arrive without features.

Both files are in `notebooks/data/`."""
        ),
        new_code_cell(SHARED),
        new_markdown_cell(
            """\
## Load both dates at the same resolution

Same `stride` gives identical lat/lon axes — convenient for plotting, and *not* required for the
transfer to work."""
        ),
        new_code_cell(
            """\
STRIDE = 8
train_features, train_latitudes, train_longitudes = era5.load_fields(
    DATA / "2018-01-01.nc", stride=STRIDE
)
test_features, test_latitudes, test_longitudes = era5.load_fields(
    DATA / "2018-01-02.nc", stride=STRIDE
)
assert np.array_equal(train_latitudes, test_latitudes)
assert np.array_equal(train_longitudes, test_longitudes)

positions_train = era5.unit_sphere(train_latitudes, train_longitudes)
positions_test = era5.unit_sphere(test_latitudes, test_longitudes)
flat_train = train_features.reshape(-1, train_features.shape[2])
flat_test = test_features.reshape(-1, test_features.shape[2])
print(
    "training date nodes:",
    positions_train.shape[0],
    "| new date nodes:",
    positions_test.shape[0],
)
"""
        ),
        new_markdown_cell("## Train on the first date\n\nNothing here knows a second date exists."),
        new_code_cell(
            CONFIG
            + """

train_graph = build_graph(positions_train, flat_train, neighbor="knn", k=era5.K)
model = NOFE(dim=3, config=CONFIG)
history = model.train(train_graph)
print(
    f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} "
    f"| mean KL {model.loss(train_graph):.6f}"
)
"""
        ),
        new_markdown_cell(
            """\
## Predict on the second date, fully observed

The new graph carries its own `x`, `pos` and edges. The only compatibility requirement is the
widths: 11 input features and `2P + 1 = 7` edge channels, which `build_graph` reproduces for any
point set of the same dimension."""
        ),
        new_code_cell(
            """\
test_graph = build_graph(positions_test, flat_test, neighbor="knn", k=era5.K)
built = model.model.config
print(
    "feature width matches:",
    test_graph.x.shape[1] == built["in_channels"],
    "| edge width matches:",
    test_graph.edge_attr.shape[1] == built["edge_channels"],
)

embedding_transfer = model.embed(test_graph)
assert embedding_transfer.shape == (test_graph.num_nodes, 3)
print("embedding on 2018-01-02:", tuple(embedding_transfer.shape))

# The new date is fully observed, so its KL is a real score, not a placeholder.
print(f"mean KL on the new date : {model.loss(test_graph):.6f}")
print(f"PCA-3 KL on the new date: {kl(test_graph, pca_baseline(test_graph, dims=3)):.6f}")
"""
        ),
        new_markdown_cell(
            """\
## Compare how each date maps into the shared embedding

The embedding is a *function* of local feature neighbourhoods, so two dates land in the same space:
similar weather regimes get similar colours even though the raw fields differ. The next cell
quantifies that instead of eyeballing it."""
        ),
        new_code_cell(
            """\
train_embedding = model.embed(train_graph)
print(f"KL on the training date: {kl(train_graph, train_embedding):.6f}")
print(f"KL on the new date     : {kl(test_graph, embedding_transfer):.6f}")
assert kl(test_graph, embedding_transfer) < kl(test_graph, pca_baseline(test_graph, dims=3))
print("transfer beats linear PCA on the unseen date: True")
"""
        ),
        new_markdown_cell(
            """\
## Partly observed second date

Withhold half the locations of 2018-01-02 and hand them to `build_graph` as `query_positions`.
Their features become the `fill` substitute, they receive messages without sending any, and their
truth — when supplied — goes to `y` for scoring only.

`test_blind` is the honest deployment case: no truth exists for the new locations."""
        ),
        new_code_cell(
            """\
rng = np.random.default_rng(0)
observed_test = rng.random(positions_test.shape[0]) < 0.5
print(f"observed {observed_test.sum()} of {observed_test.size} locations on the second date")

test_merged = build_graph(
    positions_test[observed_test],
    flat_test[observed_test],
    query_positions=positions_test[~observed_test],
    query_features=flat_test[~observed_test],
    neighbor="knn",
    k=era5.K,
)
test_blind = build_graph(
    positions_test[observed_test],
    flat_test[observed_test],
    query_positions=positions_test[~observed_test],
    neighbor="knn",
    k=era5.K,
)
print(
    "merged and blind agree on the graph:",
    torch.equal(test_merged.x, test_blind.x)
    and torch.equal(test_merged.edge_index, test_blind.edge_index),
)

predicted = model.embed(test_blind)
print("prediction for all locations:", tuple(predicted.shape))
print(f"mean KL where truth exists   : {model.loss(test_merged):.6f}")
"""
        ),
        new_markdown_cell(
            """\
## Two ways to predict the partly observed date

`output_mask` picks which rows come back. Restricting it to the observed rows reproduces a plain
observed-only prediction; leaving it wide returns the withheld locations too. The graph, its edges
and every `x` row are identical in both calls."""
        ),
        new_code_cell(
            """\
# `test_blind` reorders nodes (observed first, then added), and its own `input_mask` marks the
# observed prefix. A mask in grid order would not line up with the graph's rows.
observed_rows = test_blind.clone()
observed_rows.output_mask = test_blind.input_mask
observed_prediction = model.embed(observed_rows)
num_observed_test = int(observed_test.sum())

print("all locations :", tuple(predicted.shape))
print("observed only :", tuple(observed_prediction.shape))
max_diff = (observed_prediction - predicted[:num_observed_test]).abs().max()
print(f"max diff vs the matching prefix: {max_diff:.2e}")
assert close(observed_prediction, predicted[:num_observed_test])
print("observed-only rows are the matching prefix of the full prediction: True")
"""
        ),
        new_markdown_cell(
            """\
## Visualise the transfer

Top to bottom: the training date, the new date fully observed, and the new date with half its
locations withheld and predicted."""
        ),
        new_code_cell(
            """\
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

grid_index_test = np.empty(test_merged.num_nodes, dtype=np.int64)
grid_index_test[observed_test] = np.arange(num_observed_test)
grid_index_test[~observed_test] = np.arange(num_observed_test, test_merged.num_nodes)

predicted_grid = np.full((grid_index_test.size, 3), np.nan, dtype=np.float32)
predicted_grid[grid_index_test] = predicted.numpy()

panels = [
    ("2018-01-01 (training)", train_embedding.numpy()),
    ("2018-01-02 (fully observed)", embedding_transfer.numpy()),
    ("2018-01-02 (half withheld, predicted)", predicted_grid),
]

# All three panels share one colour scale, computed from the training embedding: without it each
# panel would rescale independently and the same weather regime would get a different colour.
scale = era5.colour_scale(panels[0][1], era5.PERCENTILE)

figure, axes = plt.subplots(
    3, 1, figsize=(12.0, 15.5), subplot_kw={"projection": ccrs.PlateCarree()}
)
for axis, (title, values) in zip(axes, panels, strict=True):
    axis.set_global()
    era5.draw_globe(axis, test_latitudes, test_longitudes, values, scale=scale)
    axis.coastlines(linewidth=0.5)
    axis.set_title(f"{title}: {values.shape[0]} nodes")
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Takeaways

|Fact|Why it matters|
|---|---|
|`train` and `embed` take different `Data`|A graph is data, not state; transfer needs no re-fitting|
|Only the widths must match|`in_channels` = features, `edge_channels` = `2P + 1`|
|`blind` graphs are embed-only|`train`/`loss` refuse scored rows holding substitutes with no `y`|
|`output_mask` chooses the rows|Same graph and inputs; only the returned subset changes|

To adapt to the new date rather than just score it, build a fresh `NOFE` and call `train` on the new
graph; nothing about the objects changes."""
        ),
    ]
)

NOTEBOOKS = (
    ("1-in-sample.ipynb", in_sample),
    ("2-held-out-training.ipynb", held_out),
    ("3-transfer-between-dates.ipynb", transfer),
)

if __name__ == "__main__":
    for name, book in NOTEBOOKS:
        target = HERE / name
        nbf.write(book, target)
        print(f"wrote {target.relative_to(HERE.parent)} ({len(book.cells)} cells)")
    # The data directory is data only; no stray artefacts should survive a build.
    for stray in HERE.glob("*.pt"):
        stray.unlink()
    del shutil
