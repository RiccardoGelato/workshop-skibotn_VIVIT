"""Generate the workflow notebooks.

Run from the repository root: ``uv run python examples/era5/utilities/_build.py``.
Notebook sources live here so the JSON is reproducible; the notebooks are written to
``examples/era5/notebooks/``, which is also where ``_execute.py`` starts its kernels.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS_DIR = HERE.parent / "notebooks"
SHARED = """\
%matplotlib inline

import sys
from pathlib import Path

import numpy as np
import torch

from nofe import NOFE
from nofe.affinity import affinity_kl_loss

# Resolve paths from the repository layout, not the process cwd: VS Code always starts kernels in
# the workspace folder (`jupyter.notebookFileRoot` is ignored since 2024.4), while `_execute.py`
# runs from this directory. Walk up from the cwd looking for the repo root, so both agree.
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
ERA5 = REPO / "examples" / "era5"
DATA = ERA5 / "data"

# The example data is not in Git (21 MB per date), so fail with the fetch command rather than the
# opaque `StopIteration`/`FileNotFoundError` the loader would otherwise raise.
if not (DATA / "2018-01-01.nc").exists():
    raise FileNotFoundError(
        f"missing {DATA / '2018-01-01.nc'}\\n"
        "The ERA5 inputs are not committed. Fetch them once with:\\n"
        f"  mkdir -p {DATA}\\n"
        f"  cp /Volumes/T7-storage/ERA5/data/2018-01-0{{1,2}}.nc {DATA}/"
    )

# The notebooks import the ERA5 helpers straight from the example scripts, so the loader, the
# graph generator and the globe plot have exactly one implementation in the repo. They are
# imported after `sys.path` is extended, which is why the noqa markers below are needed: the ERA5
# demo lives in `scripts/`, its graph generator here in `utilities/`, and the shared graph helpers
# one level up in `examples/`.
sys.path.insert(0, str(ERA5 / "scripts"))
sys.path.insert(0, str(ERA5 / "utilities"))
sys.path.insert(0, str(REPO / "examples"))
import era5  # noqa: E402
from era5_graph import Era5Graph  # noqa: E402


def kl(data, embedding) -> float:
    \"\"\"Mean KL between a graph's own feature affinities and an embedding of its rows.\"\"\"
    return float(affinity_kl_loss(data.x, embedding, data.edge_index, num_nodes=data.num_nodes))


def pca_baseline(data, dims: int = 3) -> torch.Tensor:
    \"\"\"Best linear projection to ``dims`` dimensions: the reference the model has to beat.\"\"\"
    centred = data.x - data.x.mean(dim=0, keepdim=True)
    _, _, basis = torch.linalg.svd(centred, full_matrices=False)
    return centred @ basis[:dims].T


def neighbour_sets(points: torch.Tensor, k: int = 10) -> np.ndarray:
    \"\"\"Exact ``k``-nearest-neighbour ids per row, self excluded.\"\"\"
    import faiss

    values = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    index = faiss.IndexFlatL2(values.shape[1])
    index.add(values)
    return index.search(values, k + 1)[1][:, 1:]


def knn_agreement(reference: torch.Tensor, candidate: torch.Tensor, k: int = 10) -> float:
    \"\"\"Mean share of exact k-nearest neighbours two point sets have in common.\"\"\"
    ref = neighbour_sets(reference, k)
    cand = neighbour_sets(candidate, k)
    shared = [len(set(a) & set(b)) / k for a, b in zip(ref, cand, strict=True)]
    return float(np.mean(shared))


def close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4) -> bool:
    \"\"\"Relative agreement in units of the embedding's own scale.\"\"\"
    return bool((a - b).abs().max() <= rtol * a.abs().max())


print("nofe ready")
"""

LOAD = """\
# TRAIN_STRIDE=8 is the whole globe at 89 x 180 = 16,020 grid points. Lower the stride for a finer
# grid (4 -> 64,440, 3 -> 114,720, 2 -> 258,480, 1 -> 719 x 1440 = 1,035,360).
TRAIN_STRIDE = 8
# Chord on the unit sphere: a 2.5 degree arc, giving a mean of ~12 neighbours per grid point.
TRAIN_RADIUS = 0.04363
# Inference grid and the chord that keeps the same mean degree there. The radius must shrink with
# the stride: 2.5 degrees at stride 1 would be ~917M edges (~40 GB), which cannot be built.
INFER_STRIDE = 1
INFER_RADIUS = 0.005454
NYSTROM = 0.25
graph = Era5Graph(radius=TRAIN_RADIUS, stride=TRAIN_STRIDE, data_dir=DATA)
features, latitudes, longitudes = graph.load("2018-01-01")
flat = features.reshape(-1, features.shape[2])
print(
    f"grid {features.shape[0]}x{features.shape[1]} -> {flat.shape[0]} nodes, "
    f"{flat.shape[1]} variables each"
)
"""

MODEL_DOC = """\
`width=8, ker_width=32` are the reference ERA5 settings: `NNConv` materialises an
`[num_edges, width]` message tensor, so width drives memory. 200 epochs at this resolution takes
well under a minute on an M-series GPU. `NOFE` is the trainer and holds the network as `.net`; the
input and edge widths are discovered on the first forward pass, so nothing has to be declared here.
"""

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
# 1 — One graph: train and embed every grid point

The base case. Every grid point has features and is also a location, so nothing is withheld and
nothing is queried:

- `Era5Graph.build_raster(features, lat, lon)` returns a `Data` with one node per grid point, its
  affinities in `data.affinity`, and no masks.
- `model.training(graph)` scores every node; the internal train/validation split chooses which rows
  enter the objective each epoch, it never hides neighbours.
- `model.embed(graph)` returns exactly `num_nodes` rows, in node order."""
        ),
        new_code_cell(SHARED),
        new_markdown_cell(
            """\
## Load one date at coarse resolution

`Era5Graph.load` standardises each variable per field and drops the two duplicated pole rows.
`Era5Graph.positions` turns lat/lon into 3-D unit vectors (in earth radii), so the ±180° seam and
the poles are ordinary neighbours for the radius graph."""
        ),
        new_code_cell(LOAD),
        new_markdown_cell(
            """\
## Build the graph

`Era5Graph` is radius-only: every grid point within `TRAIN_RADIUS` of a receiver sends it a message,
banded by latitude so even a stride-1 raster stays buildable. The graph also carries the target: the
student-t affinities of each node's feature rows against its own neighbourhood, which is what the
loss reproduces."""
        ),
        new_code_cell(
            """\
data = graph.build_raster(features, latitudes, longitudes)
print(data)
print(
    "edges", data.edge_index.shape[1],
    "| edge_attr width", data.edge_attr.shape[1],
    "| affinity entries", data.affinity.shape[0],
)
# Era5Graph passes normalize=False: `load` already standardised every variable, so `pos` and
# `pos_raw` are the same unit-sphere coordinates here. `PointGraph`'s default would z-score them.
print("pos is raw for this builder:", torch.equal(data.pos, data.pos_raw))
assert data.affinity.shape[0] == data.edge_index.shape[1]
"""
        ),
        new_markdown_cell("## Train\n\n" + MODEL_DOC),
        new_code_cell(
            """\
model = NOFE(3, width=8, ker_width=32, nystrom=NYSTROM)
history = model.training(data, epochs=200, seed=0, verbose=True, log_every=50)
print(
    f"loss {history[0]['train_loss']:.5f} -> {history[-1]['train_loss']:.6f} "
    f"| best val {history[-1]['best_val_loss']:.6f}"
)
print("resolved widths:", model.net.in_channels, model.net.edge_channels)
"""
        ),
        new_markdown_cell(
            """\
## Embed the same data

`embed` returns one row per node in node order (lat-major, matching the raster). `loss` is the mean
KL over the same rows, read from the graph's own `affinity`."""
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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

figure = era5.globe_figure(
    latitudes, longitudes, embedding.numpy(), f"2018-01-01, stride {TRAIN_STRIDE}: in-sample"
)
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Checkpointing

`NOFE` is a plain trainer, not an `nn.Module`, so the checkpoint is the network's own state dict:
`torch.save(model.net.state_dict(), path)`. Reloading reproduces the embedding to floating-point
tolerance.

On MPS the comparison is *not* bit-exact: repeating the same forward pass twice already differs by
~4e-6 (measured here), and reloading adds ~6e-6, on embeddings whose magnitude is ~20. That is
Metal kernel scheduling, not a checkpoint defect — `device="cpu"` reproduces bit-identically. The
assertion below uses a relative tolerance for that reason."""
        ),
        new_code_cell(
            """\
path = Path("in_sample.pt")
torch.save(model.net.state_dict(), path)
restored = NOFE(3, width=8, ker_width=32, nystrom=NYSTROM, in_channels=11, edge_channels=7)
restored.net.load_state_dict(torch.load(path, weights_only=True))
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
## Part B — train coarse, embed fine

Training needs the whole edge set resident at once, because the objective sees every neighbour on
every epoch. Inference does not: it is one forward pass, so it can cover a grid whose graph could
never be held alongside the optimiser.

The radius is fixed at construction, so a second `Era5Graph` at the finer stride carries its own
chord. The fine graph is never trained on."""
        ),
        new_code_cell(
            """\
fine = Era5Graph(radius=INFER_RADIUS, stride=INFER_STRIDE, data_dir=DATA)
fine_features, fine_latitudes, fine_longitudes = fine.load("2018-01-01")
fine_data = fine.build_raster(fine_features, fine_latitudes, fine_longitudes)
print(f"fine grid: {fine_data.num_nodes} nodes, {fine_data.edge_index.shape[1]} edges")
print(
    "mean in-degree:",
    float(torch.bincount(fine_data.edge_index[1], minlength=fine_data.num_nodes).float().mean()),
)
fine_embedding = model.embed(fine_data)
print("fine embedding:", tuple(fine_embedding.shape))
assert fine_embedding.shape == (fine_data.num_nodes, 3)
assert torch.isfinite(fine_embedding).all()
"""
        ),
        new_markdown_cell(
            """\
`embed` never consults the graph it was trained on: it reads `x`, `edge_index` and `edge_attr` from
whatever `Data` it is handed. The only requirement is matching widths — 11 input features and
`2P + 1 = 7` edge channels — which `Era5Graph` reproduces at any stride, so a coarse model embeds a
grid it has never seen."""
        ),
        new_code_cell(
            """\
# One scale from the coarse embedding so both panels are directly comparable: per-panel rescaling
# would give the same weather regime a different colour on the two grids.
scale = era5.colour_scale(embedding.numpy(), era5.PERCENTILE)
panels = [
    (
        f"stride {TRAIN_STRIDE} ({data.num_nodes} nodes, trained)",
        latitudes,
        longitudes,
        embedding.numpy(),
    ),
    (
        f"stride {INFER_STRIDE} ({fine_data.num_nodes} nodes, inferred)",
        fine_latitudes,
        fine_longitudes,
        fine_embedding.numpy(),
    ),
]
figure, axes = plt.subplots(
    2, 1, figsize=(12.0, 13.0), subplot_kw={"projection": ccrs.PlateCarree()}
)
for axis, (title, lat, lon, values) in zip(axes, panels, strict=True):
    axis.set_global()
    era5.draw_globe(axis, lat, lon, values, scale=scale)
    axis.coastlines(linewidth=0.5)
    axis.set_title(title)
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Takeaways

|Fact|Why it matters|
|---|---|
|One node per feature row|Every supplier of features is also a location; nothing is withheld|
|`data.affinity` is the target|The loss reads it off the graph; no truth lives anywhere else|
|`embed` returns `num_nodes` rows|One row per node, in node order|
|Widths resolve on the first forward|`net.in_channels` exists only after a graph was seen|

Next: **4-many-graphs.ipynb** trains one operator on several subgraphs of the same date."""
        ),
    ]
)

# 4 --------------------------------------------------------------------------------------
many_graphs = notebook(
    [
        new_markdown_cell(
            """\
# 4 — Several graphs from one date, trained together, applied to another

One date is one graph, but `training` also takes a list. Here the training set is **four random
subsets of the 2018-01-01 grid points at stride `STRIDE`**, each a `Data` with its own nodes and
its own edges; the trained model is then applied to the **complete 2019-08-01 field at native
stride 1** — 1,035,360 locations of a date it never saw, on a grid finer than anything it trained
on.

An epoch is one optimizer step per graph, so `epochs=200` over four graphs is 800 steps; the
reported `train_loss`/`val_loss` are the means over those graphs."""
        ),
        new_code_cell(SHARED),
        new_markdown_cell(
            """\
## Sample subgraphs from the first date

Each subgraph keeps a random `SAMPLE` of the grid points (`PointGraph(sample=...)`, drawn with its
own seed): same radius, fewer points, so a few percent of the kept nodes end up with no neighbour
inside their own subgraph. They are not broken — a node with no incoming edge is embedded from its
own features through the root weight — but they carry no neighbourhood information, which is why the
in-degree is printed rather than assumed."""
        ),
        new_code_cell(
            """\
STRIDE = 8
# Chord on the unit sphere: a 2.5 degree arc, the same neighbourhood as notebook 1.
RADIUS = 0.04363
SAMPLE = 8_010  # half of the 16,020 stride-8 grid points
NUM_GRAPHS = 4

graph = Era5Graph(radius=RADIUS, stride=STRIDE, data_dir=DATA)
features, latitudes, longitudes = graph.load("2018-01-01")
print(f"{features.shape[0] * features.shape[1]} grid points on 2018-01-01, "
      f"{features.shape[2]} variables each")


def sample_graph(seed: int):
    \"\"\"A random subset of the grid points, with a radius neighbourhood of its own.\"\"\"
    builder = Era5Graph(radius=RADIUS, stride=STRIDE, data_dir=DATA, sample=SAMPLE, seed=seed)
    return builder.build_raster(features, latitudes, longitudes)


graphs = [sample_graph(seed) for seed in range(NUM_GRAPHS)]
for index, sampled in enumerate(graphs):
    degrees = torch.bincount(sampled.edge_index[1], minlength=sampled.num_nodes)
    print(
        f"graph {index}: {sampled.num_nodes} nodes, {sampled.edge_index.shape[1]} edges, "
        f"mean in-degree {degrees.float().mean():.2f}, isolated {int((degrees == 0).sum())}"
    )
"""
        ),
        new_markdown_cell(
            """\
## Train on the list

`training` runs one optimizer step per graph per epoch, returns one history per graph, and averages
the losses it reports. `loss` on a list returns one float per graph, so the list is the per-graph
scores below in order."""
        ),
        new_code_cell(
            """\
model = NOFE(3, width=8, ker_width=32, nystrom=0.25)
histories = model.training(graphs, epochs=200, seed=0, verbose=True)
per_graph = model.loss(graphs)
print(f"{len(histories[0])} epochs, {len(histories[0]) * len(graphs)} optimizer steps")
print(f"graph 0 loss {histories[0][0]['train_loss']:.5f} -> {histories[0][-1]['train_loss']:.6f}")
print("mean KL per graph:", [round(value, 6) for value in per_graph])
assert len(histories) == len(per_graph) == NUM_GRAPHS
"""
        ),
        new_markdown_cell(
            """\
## Infer on the full second date, at native resolution

Training ran on 50% samples at `STRIDE`, because the objective needs every training edge resident
on every epoch. Inference does not: it is one forward pass, so it can cover the whole grid. The
model is therefore applied to **all 719 x 1440 = 1,035,360 grid points** of 2019-08-01 at stride 1,
a grid it has never seen.

The radius has to shrink with the stride — 2.5 degrees at stride 1 would be ~917M edges (~40 GB) —
and `0.005454` keeps the neighbourhood the fine grid can hold. `embed` stays single-graph and
returns one row per node."""
        ),
        new_code_cell(
            """\
INFER_STRIDE = 1
INFER_RADIUS = 0.005454
full_graph = Era5Graph(radius=INFER_RADIUS, stride=INFER_STRIDE, data_dir=DATA)
full_features, full_latitudes, full_longitudes = full_graph.load("2019-08-01")
full = full_graph.build_raster(full_features, full_latitudes, full_longitudes)
print(f"full grid: {full.num_nodes} nodes, {full.edge_index.shape[1]} edges")

embedding = model.embed(full)
print("embedding on 2019-08-01:", tuple(embedding.shape))
assert embedding.shape == (full.num_nodes, 3)
assert torch.isfinite(embedding).all()
"""
        ),
        new_markdown_cell(
            """\
## Plot the embedding of the unseen date

R/G/B are the three embedding dimensions, one `pcolormesh` quad per grid point, scaled by the
5th/95th percentile spread (`era5.PERCENTILE`)."""
        ),
        new_code_cell(
            """\
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

figure, axis = plt.subplots(
    1, 1, figsize=(12.0, 6.5), subplot_kw={"projection": ccrs.PlateCarree()}
)
axis.set_global()
era5.draw_globe(axis, full_latitudes, full_longitudes, embedding.numpy())
axis.coastlines(linewidth=0.5)
axis.set_title(f"2019-08-01, never trained on: {full.num_nodes} nodes")
plt.show()
"""
        ),
        new_markdown_cell(
            """\
## Takeaways

|Fact|Why it matters|
|---|---|
|`training` takes a list|One operator covers many graphs; an epoch is one step per graph|
|`sample=N` draws N rows|A subgraph is the same builder with fewer points, nothing else|
|`loss` on a list is per graph|Each graph counts once, whatever its node count|
|Inference needs no training grid|A coarse, sub-sampled model embeds the full-resolution field|
|`embed` takes one graph or a list|It returns the matching type: one tensor, or one per graph|"""
        ),
    ]
)


NOTEBOOKS = (
    ("1-in-sample.ipynb", in_sample),
    ("4-many-graphs.ipynb", many_graphs),
)

if __name__ == "__main__":
    for name, book in NOTEBOOKS:
        target = NOTEBOOKS_DIR / name
        nbf.write(book, target)
        print(f"wrote {target.relative_to(HERE.parent.parent)} ({len(book.cells)} cells)")
    # The notebooks directory is notebooks only; no stray artefacts should survive a build.
    for stray in NOTEBOOKS_DIR.glob("*.pt"):
        stray.unlink()
    del shutil
