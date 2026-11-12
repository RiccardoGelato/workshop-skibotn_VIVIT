# nofe

Neural operator function embedding: graph-kernel dimensionality reduction.

`nofe` takes a graph whose nodes carry feature vectors and learns a low-dimensional
embedding by training a graph-kernel operator (GKO). The operator is trained so that the
neighbourhood structure it predicts in embedding space reproduces the neighbourhood
structure of the input features. For a query node $i$ the objective is

$$\mathrm{KL}(p_i \,\|\, q_i) = \sum_{j \in \mathcal{N}(i)} p_j \log \frac{p_j}{q_j}$$

with the student-t kernel $q_j \propto 1 / (1 + \lVert y_i - y_j \rVert^2)$ over the learned
embedding $y$, and $p$ the same kernel evaluated on the input features $x$. Distances are
normalised per query node, so neighbourhoods of any degree are comparable, and the loss is
the mean over query nodes rather than a sum over edges.

Two axes are independent:

| Axis | Values | Fixed by |
|---|---|---|
| neighbourhood | `"radius"` (default) or `"knn"` | `build_graph(...)`; the graph you pass in |
| operator | dense (default) or Nyström landmarks | `config["nystrom"]`, once at construction |

`nofe.build_graph` turns arrays into the `Data` the operator consumes; `nofe.graphs.radius_graph`
and `nofe.graphs.knn_graph` are the edge builders underneath it.

## Install

```bash
uv add nofe        # or: pip install nofe
```

| Target | Command |
|---|---|
| any platform, CPU | `UV_TORCH_BACKEND=cpu uv sync` |
| Linux x86_64 + NVIDIA | `UV_TORCH_BACKEND=cu128 uv sync` (also `cu126`, `cu130`) |
| macOS arm64 | `uv sync` (Metal/MPS is enabled in the default macOS wheel) |
| Windows AMD64 | `UV_TORCH_BACKEND=cpu uv sync` |

`torch` is pinned as `torch>=2.6,<3`; the CUDA variant is selected by `UV_TORCH_BACKEND` at
install time and the shipped `uv.lock` is backend-independent. Requires Python ≥ 3.11.
`pip install nofe` also pulls `faiss-cpu`, which backs the neighbour search; when faiss is
missing or unusable the search silently falls back to `scipy.spatial.cKDTree` and prints one
`RuntimeWarning`.

### VS Code

`.vscode/settings.json` points VS Code at `.venv`, so the Python extension, Pylance, the
integrated terminal and the test runner all use this project's interpreter with no picker step.
It also enables pytest discovery over `tests/` and adds `src` to Pylance's search path.

This is a per-workspace override: the machine's user-level `python.defaultInterpreterPath` points
at a conda environment, and the workspace setting wins only for this folder. If the status bar
still shows another interpreter, run **Python: Select Interpreter** and choose
`.venv/bin/python` once — VS Code then records it in its own workspace state.

For the notebooks, install the extra and register a kernel once:

```bash
uv sync --extra notebooks
.venv/bin/python -m ipykernel install --user --name nofe-pkg --display-name "NOFE (repo .venv)"
```

Then open any notebook and pick **NOFE (repo .venv)** as the kernel. The name is deliberately
specific rather than `nofe`: a `nofe` kernelspec may already exist and point at a different
checkout, in which case selecting it silently runs the wrong environment. Check with
`jupyter kernelspec list` if a notebook imports an unexpected `nofe`.

The notebooks find their data from the repository layout rather than the process cwd, because VS
Code always starts kernels in the workspace folder (`jupyter.notebookFileRoot` has been ignored
since extension 2024.4). Running them from VS Code, from `notebooks/`, or through `_execute.py` all
resolve `notebooks/data` the same way.

## Graph contract

Every method consumes `torch_geometric.data.Data` with these fields:

|Field|Required|Shape|Meaning|
|---|---|---|---|
|`x`|always|`[N, F]` float|node features the model consumes; also the loss target when `y` is absent|
|`edge_index`|always|`[2, E]` int64|`[neighbor, query]`; messages flow `0 -> 1`|
|`edge_attr`|always|`[E, D]` float|edge features driving the kernel|
|`pos`|optional|`[N, P]` float|coordinates; 2-D when present|
|`y`|optional|`[N, F]` float|ground truth for rows whose `x` is only a substitute|
|`input_mask`|optional|`[N]` bool|provenance: `True` = observed row (sends messages)|
|`output_mask`|optional|`[N]` bool|rows to embed and score (default: all)|

Node count always comes from `data.num_nodes`. The convolution aggregates with `mean`, so
self loops are not required; the builders never emit them. `validate_graph` raises a
`ValueError` naming the offending field.

## Building a graph

```python
from nofe import build_graph

data = build_graph(positions, features, neighbor="knn", k=10)
```

|Argument|Meaning|
|---|---|
|`positions`|`[N, P]` observed coordinates|
|`features`|`[N, F]` observed features; `None` only with `feature_dim=` and added locations|
|`query_positions`|`[Q, P]` added locations, features unknown|
|`query_features`|`[Q, F]` ground truth for the added locations; goes to `y`, never to `x`|
|`feature_dim`|required when `features is None`|
|`neighbor`|`"radius"` (default) or `"knn"`|
|`radius`, `k`|neighbourhood size for the chosen `neighbor`|
|`fill`|`"interpolate"` (default), `"zeros"` or `"mean"` — how added rows are initialized|

Merging unions **locations, not features**. The node set becomes `positions ∪ query_positions`,
but the added locations have no known features: they are initialized by `fill` from the observed
rows and stay `input_mask=False`, so they receive messages and never send — the same graph at
training and at inference. `query_features`, when supplied, is those locations' **ground truth**
and lands in `y` for the loss only.

||Call|`x` on added rows|`y`|Loss covers|
|---|---|---|---|---|
|in-sample|`build_graph(p, f)`|— no added rows|absent|all rows|
|merged + truth (training)|`+ query_positions=q, query_features=fq`|`fill` substitute|`fq`|inputs + added|
|merged, no truth (inference)|`+ query_positions=q`|`fill` substitute|absent|not scorable, `embed` only|

Which locations you withhold is entirely your choice: pass them in `query_positions` and put
their truth in `query_features`. There is no holdout fraction; duplicates are dropped by exact
coordinate match with one `UserWarning`. `fill="interpolate"` takes the inverse-distance-weighted
mean of the `k` nearest observed rows, because zeros make an added row look like an observation at
the feature origin.

```python
import torch
from nofe import NOFE, build_graph

positions = torch.rand(900, 2) * 2 * torch.pi
features = torch.rand(900, 3)

# 1. in-sample: one node per observed row
data = build_graph(positions, features, neighbor="knn", k=10)
model = NOFE(dim=2, config={"epochs": 150})
history = model.train(data)
embedding = model.embed(data)  # [900, 2]

# 2. out-of-sample with truth: withheld rows get a substitute in x and their truth in y
merged = build_graph(
    positions[:800],
    features[:800],
    query_positions=positions[800:],
    query_features=features[800:],
    neighbor="knn",
    k=10,
)
model = NOFE(dim=2, config={"epochs": 150})
history = model.train(merged)  # scores all 900 rows
score = model.loss(merged)

# 3. true out-of-sample: same graph, no truth, embed only
blind = build_graph(
    positions[:800],
    features[:800],
    query_positions=positions[800:],
    neighbor="knn",
    k=10,
)
model.embed(blind)  # [900, 2]; train()/loss() refuse it
```

Two runnable scripts:

|Script|Shows|
|---|---|
|`examples/swiss_roll.py`|synthetic features on a 2-D domain: the dense path, a merged out-of-sample graph on the Nyström path, and a checkpoint round trip|
|`examples/era5.py`|a real single-timestep ERA5 field over the whole globe: 11 physical variables as the feature space, every grid point as the domain, `dim=3`, and an RGB map of the embedding|

The ERA5 example needs the optional dependencies and a netCDF file:

```bash
uv sync --extra examples
uv run python examples/era5.py                      # default: /Volumes/T7-storage/ERA5/data/2018-01-01.nc
NOFE_ERA5_DATA=/path/to/era5 uv run python examples/era5.py --date 2018-06-01
```

It prints graph size, loss history and mean KL, then writes `examples/era5_embedding.png`: the full globe in
a cartopy `PlateCarree` projection with coastlines and borders, where R/G/B are the three embedding
dimensions. The map shows the embedding, not the input features — a PCA of the 11 variables is not plotted.
Each channel is centred on its mean and divided by twice its 5th/95th-percentile deviation (the reference
`rgb_scale`), because a single extreme point would otherwise compress the whole field into a grey wash.

The plot is `pcolormesh`, one quad per grid point carrying its own RGB triple (`era5.draw_globe`). Unlike an
`imshow` raster it needs no repeated seam column: the quad edges run cell-to-cell, so the antimeridian closes
between the last and first column and non-finite cells stay transparent (alpha 0) instead of being painted.
The RGBA array is passed as `C`, not as `color=`, because `color` sets a single facecolour that matplotlib
then overrides with the scalar array's colormap.

`NOFE_ERA5_DATA` selects the directory, `NOFE_ERA5_DATE` the file stem, `--data`/`--date`/`--out` override both.

`STRIDE = 2` keeps every ERA5 grid point (258,480 nodes, 2,584,800 edges); raise it to shrink the graph.
`CONFIG` pins `width=8, ker_width=32`, the reference ERA5 model configs. That matters on MPS: `NNConv`
materialises an `[num_edges, width]` message tensor, so at 2.58M edges the nofe defaults (`width=32`) need
~9.9 GiB and abort with `MPS backend out of memory` (observed: 20.11 GiB allocated), while `width=8` needs
0.62 GiB and the whole run peaks at 0.94 GiB. `edge_batch_size` does not help — `EdgeConv.message` concatenates
every chunk before returning, so the full stack is allocated regardless.

## Notebooks

`notebooks/` holds three executed walkthroughs on real ERA5 data. They need two dates,
2018-01-01 and 2018-01-02, in `notebooks/data/`; that directory is **not** in Git (each file is
21 MB), so fetch them once before running the notebooks. Nothing in `nofe` itself depends on them.

```bash
mkdir -p notebooks/data
cp /Volumes/T7-storage/ERA5/data/2018-01-0{1,2}.nc notebooks/data/   # any ERA5 source works
```

They are the fastest way to see which workflow matches a problem:

|Notebook|Shows|
|---|---|
|`1-in-sample.ipynb`|every location observed: `build_graph` → `train` → `embed` → `loss`, a PCA baseline, the per-point RGB globe, and a checkpoint round trip|
|`2-held-out-training.ipynb`|a random 20% of locations withheld: the `observed` / `merged` / `blind` graphs, `fill` substitution, training against `y`, and two predictions of the same coarse graph — all locations vs observed rows only|
|`3-transfer-between-dates.ipynb`|train on 2018-01-01, embed 2018-01-02 fully observed and half withheld; `output_mask` selects the returned rows|

```bash
uv sync --extra notebooks
cd notebooks && uv run python _execute.py      # run and store outputs in place
uv run python _build.py                        # regenerate the .ipynb sources from _build.py
```

`_build.py` is the source of truth for the cell content; edit it and rebuild rather than editing the
JSON. `_execute.py` doubles as the notebooks' test: any cell that raises fails the run.

Each notebook scores against `pca_baseline`, the best *linear* projection to three dimensions. On
this data the 11 variables are nearly isotropic (the top three components carry 54% of the
variance), so the trained operator reaches KL ≈ 0.014 where PCA-3 reaches ≈ 0.055; exact 10-NN set
overlap is a much weaker proxy and is reported alongside for honesty. On MPS, repeated forward
passes differ by ~4e-6 on embeddings of magnitude ~20, so the round-trip check uses a relative
tolerance; `device="cpu"` is bit-exact.

## API

```python
NOFE(dim=2, *, config=None, **overrides)
```

`config` is a mapping and the keyword arguments are flat overrides; merge order is
`DEFAULTS` < `config` < keyword overrides. Unknown keys fail at construction.

|Member|Returns / raises|Notes|
|---|---|---|
|`.config`|`dict`|defaults + mapping + overrides + `dim`|
|`.model`|`KernelNN`|`RuntimeError` before build/train/embed/loss|
|`.device`|`torch.device`|resolved from `config["device"]` at construction|
|`.history`|`list[dict]`|`epoch`, `train_loss`, `val_loss`, `best_val_loss`|
|`.best_val_loss`|`float \| None`|best validation loss seen; `None` before training|
|`.build(data)`|`KernelNN`|infers widths from `data`; later calls reuse the weights|
|`.train(data)`|`list[dict]`|trains in place and returns the history|
|`.embed(data)`|CPU `torch.Tensor`|`[num_output_nodes, dim]`, rows in `output_mask` order|
|`.loss(data)`|`float`|mean KL over the scored rows of the trained operator|
|`.save(path)`|`Path`|writes a `nofe-checkpoint-2` payload|
|`NOFE.load(path, device=None)`|`NOFE`|`weights_only=True` load, model in `eval()` mode; a `nofe-checkpoint-1` file (which had a `method` argument) is rejected with a retrain hint|

### Config keys

|Key|Default|Meaning|
|---|---|---|
|`width`|`32`|model width|
|`ker_width`|`64`|kernel MLP width|
|`depth`|`3`|number of convolution layers|
|`edge_batch_size`|`32768`|edge chunking; `None` disables it|
|`neighbor`|`"radius"`|`"radius"` or `"knn"`; passed to `build_graph` when you call it|
|`radius`|`1.0`|radius for `neighbor="radius"`|
|`k`|`10`|neighbours for `neighbor="knn"`|
|`epochs`|`200`|training epochs|
|`lr`|`1e-2`|AdamW learning rate|
|`weight_decay`|`5e-4`|AdamW weight decay|
|`grad_clip`|`1.0`|gradient-norm clip; `None` disables it|
|`patience`|`20`|early-stopping patience|
|`min_delta`|`1e-6`|minimum validation improvement|
|`val_fraction`|`0.1`|share of the scored rows held out for validation|
|`early_stopping`|`True`|stop when validation stalls|
|`seed`|`42`|seed for python, numpy, torch and the landmark draws|
|`device`|`"auto"`|`"auto"`, `"cpu"`, `"cuda"` or `"mps"`|
|`verbose`|`False`|print loss progress|
|`log_every`|`1000`|print every N epochs (the final epoch always prints)|
|`nystrom`|`None`|`None` = dense operator; a fraction in `(0, 1]` enables the Nyström approximation with that landmark share|
|`chunk_size`|`8192`|query chunk size for the Nyström path|

Early stopping is on by default: training breaks once validation stops
improving for `patience` epochs, and the best weights are restored. With
`early_stopping=False` every epoch runs and the best epoch is **still** restored, so the
final model is deterministic given the seed.

`neighbor`/`radius`/`k` are echoed in `config` so a resolved config can be passed straight to
`build_graph(**{key: config[key]})`; `nofe` itself never builds edges for you.

## Neighbourhoods and Nyström

**Neighbourhood.** `build_graph(..., neighbor=...)` decides the edges, once, before the model
sees anything: `radius` (default) connects every row within `radius`, `knn` connects each row to
its `k` nearest senders, and either way the senders are the observed rows. The operator is
neighbourhood-agnostic — it consumes `edge_index`/`edge_attr` — so the two choices are
interchangeable and produce identical `edge_index` bytes whichever search backend (faiss or the
scipy fallback) is installed.

**Dense operator** (`nystrom=None`, the default) is the exact path. One forward pass covers the
whole graph; the train/validation split selects which scored rows enter the objective
(`val_fraction` of `output_mask`, drawn from `config["seed"]`), not which neighbours are visible,
so gradients flow through the entire edge set.

**Nyström operator** (`nystrom=fraction`) approximates the operator with landmarks:

- Landmarks are the **senders** of messages, drawn from `input_mask`, and there are
  `round(fraction × |input_mask|)` of them (at least one). Queries are the receivers, taken from
  `output_mask ∩ input_mask` for supervision and from `output_mask` for embedding (a row without
  features has no target distribution).
- The active subgraph keeps, from the graph's own edges, every edge whose sender is a landmark
  and whose receiver is active, so the approximation inherits whatever neighbourhood
  `build_graph` chose and peak memory is `O(queries × mean_degree)` per step.
- Subgraph features are copied unchanged. Withheld rows already hold the `fill` substitute the
  dense path sees; nothing zeroes them at forward time, so the two paths agree on the model input.
- Queries are walked in ascending node order in chunks of `chunk_size`; chunk size changes peak
  memory, not the result.
- The mode is fixed at construction. There is deliberately no per-call override: `EdgeConv`
  aggregates with `mean` and `geometry_edge_attr` rescales the per-receiver `relative` distance
  column, so switching modes between training and inference changes the network's input
  statistics without any error surfacing.

**`embed` and `loss`.** `embed` always returns one row per `output_mask` node, in that order, for
both operators. `loss` scores the `output_mask` rows. Both take the target from `y` when present
and from `x` otherwise; a graph whose scored rows include added locations but carry no `y` is
refused by `train` and `loss` (`embed` still works) because those rows hold a substitute, not
truth.

## Platform notes

- Intel macOS is unsupported: no `torch` wheel ships for `macosx_*_x86_64` at any version
  ≥ 2.6.
- macOS arm64 requires macOS ≥ 14, because torch 2.15 publishes
  `macosx_14_0_arm64` wheels.
- CPU/CUDA choice happens at install time (`UV_TORCH_BACKEND`), device choice at runtime
  (`config={"device": ...}`). `"auto"` prefers CUDA, then MPS, then CPU.
- The caller's `Data` is cloned before it moves to the accelerator, so `data.x.numpy()`
  keeps working on the CPU copy.
- Checkpoints from the research repository use `NNConv_old` parameter names
  (`conv1.root`); map `root.T -> lin.weight` by hand — no compatibility loader ships.
- `faiss-cpu` is pinned to `>=1.15,<1.16`: 1.16.0 publishes only `macosx_15_0_*` wheels and
  breaks macOS 14, while 1.15.1 ships wheels for every platform in `required-environments`. Its
  macOS wheel bundles its own `libomp.dylib`, so a process that has already loaded torch's copy
  would abort on the first parallel search (`OMP: Error #15`); `nofe` detects the duplicate
  runtime and uses the scipy fallback with one `RuntimeWarning` instead of crashing.
- The `examples` extra pins `netCDF4<1.7.5`: 1.7.5+ publishes only macOS 15 wheels, so macOS 14 cannot
  install it. It also pulls `cartopy`, whose `coastlines()`/`BORDERS` download Natural Earth shapefiles on
  first use and cache them in `~/.local/share/cartopy`. Everything else in `nofe` remains torch-only;
  `pip install nofe` pulls neither matplotlib nor cartopy.

## Development

```bash
uv sync --extra examples --extra notebooks
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python examples/swiss_roll.py
uv run python examples/era5.py
cd notebooks && uv run python _execute.py
uv build
```
