# nofe

Neural operator function embedding: graph-kernel dimensionality reduction.

`nofe` takes a graph whose nodes carry feature vectors, learns a low-dimensional embedding with a
graph-kernel operator (GKO), and hands that embedding back. Training makes the *neighbourhood
structure of the embedding* reproduce the *neighbourhood structure of the input features*. For a
query node $i$, with $p_j$ the student-t affinity between feature vectors $x_i$ and $x_j$, and $q_j$
the same kernel on the learned embedding $z$,

$$\mathrm{KL}(p_i \,\|\, q_i) = \sum_{j \in \mathcal{N}(i)} p_j \log \frac{p_j}{q_j}, \qquad
q_j \propto \frac{1}{1 + \lVert z_i - z_j \rVert^2}$$

($z$ is the embedding, returned by `embed`; `data.y` is a different thing — the ground-truth feature
row of a node whose `x` is only a substitute, described under
[The row contract for out-of-sample rows](#the-row-contract-for-out-of-sample-rows).)

Distances are normalised per query node, so neighbourhoods of any degree are comparable, and the
loss is the mean over query nodes rather than a sum over edges.

Two things follow from that objective, and they explain most of the package's surface:

- **The graph decides what "neighbour" means.** `nofe` never builds neighbourhoods and never reads
  `pos` to derive edges. A radius, a k-nearest set, a mesh, a physical model — whatever `edge_index`
  says is the neighbourhood is the one the embedding is trained to preserve.
- **Only the operator is the package's business.** Everything else — which rows exist, which rows
  send, which rows are scored — is configuration the caller expresses through the graph's fields.

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

`torch` is pinned as `torch>=2.6,<3`; the CUDA variant is selected by `UV_TORCH_BACKEND` at install
time and the shipped `uv.lock` is backend-independent. Requires Python ≥ 3.11. Runtime dependencies
are only `torch`, `torch-geometric` and `numpy`.

## Providing data

`nofe` consumes graphs as `torch_geometric.data.Data`. `train`, `loss` and `build` accept one graph
or an iterable of them; `embed` takes exactly one. There is no dataset class, no transform pipeline
and no train/test file format: assemble each graph yourself, from whatever your data source is, and
hand it over.

### Required fields

| Field | Shape / dtype | Meaning |
|---|---|---|
| `x` | `[N, F]` float | the features whose neighbourhoods define the target affinities. Also the loss target when there is no `y`. |
| `edge_index` | `[2, E]` **int64** (`torch.long`) | `[sender, receiver]`; messages flow `row 0 -> row 1`. Self loops are allowed but unnecessary. |
| `edge_attr` | `[E, D]` float | one feature row per edge; this is what the kernel MLP sees, so it decides the operator's geometry. |

`N` is `data.num_nodes`. `F` and `D` are inferred from the tensors — they fix the model's input and
edge widths at first build and cannot change afterwards. Reusing a trained model on a graph with a
different `F` or `D` fails inside the first linear layer (`RuntimeError: addmm: mat1 and mat2 shapes
cannot be multiplied`); graphs that must share a model have to agree on both widths.

### Optional fields

| Field | Shape / dtype | Meaning |
|---|---|---|
| `pos` | `[N, P]` float | node coordinates. Never used to derive edges. **Required by the Nyström path**, which slices it when building subgraphs, so a Nyström graph without it fails with `TypeError: 'NoneType' object is not subscriptable`; the dense path never reads it. |
| `y` | `[N, F]` float | ground-truth features for rows whose `x` is only a substitute. Reaches the objective only — never `x`, never the edges. |
| `input_mask` | `[N]` bool | provenance: `True` = an observed row. On the Nyström path this is the landmark pool. |
| `output_mask` | `[N]` bool | the rows to embed and score. Default: all rows. `input_mask` must be a subset of it. |
| `edge_dist` | `[E]` float | carried through untouched if you set it; `nofe` never needs it. |

Any other attribute you attach survives cloning and moves with the graph, so metadata (ids,
timestamps, station names) can ride along.

### What validation enforces

`validate_graph` runs on every `train`/`embed`/`loss`/`build` call and raises `ValueError` naming the
offending field:
- `x` is 2-D; `edge_index` is `[2, E]`, `torch.long`, and every id is in `[0, N)`.
- `edge_attr` has exactly `E` rows.
- `pos`, `input_mask`, `output_mask` are 1-D with `N` entries.
- `input_mask ⊆ output_mask` when both are present.
- `y`, when present, has exactly the shape of `x`.

Float fields are cast to `float32` on the way to the device, so `float64` inputs are accepted; the
embedding comes back on CPU.

### Semantics worth knowing before you assemble edges

- **Aggregation is `mean` with a root weight.** Each layer computes
  `h_i' = W_root h_i + mean_{j -> i} MLP(edge_attr_ji) h_j`, then ReLU. Two consequences:
  self loops are unnecessary (`nofe` supplies the self term), and a node's own features reach its
  embedding even when it has no incoming edges.
- **A directed edge `i -> j` feeds `x_i` into `y_j` and nothing else.** There is no reverse edge
  unless you add one. A radius or k-NN graph is usually made symmetric by the builder; a graph with
  observed rows as the only senders is deliberately not.
- **The loss only scores receivers.** KL is defined over each query node's incoming neighbourhood, so
  a row with zero in-degree contributes nothing (its `x` still drives its own embedding through the
  root weight). If a scored row's neighbourhood is empty, that row is silently absent from the
  objective — check `torch.bincount(edge_index[1], minlength=N) == 0` for the rows you expect to be
  scored.

### Building a neighbourhood

The intended neighbourhood is a **radius** graph. `nofe` trains a graph-kernel operator, and a
radius graph is what makes that operator a genuine neural-operator construction: the edge set is a
discretisation of a continuum kernel rather than a per-point lookup table. A kernel restricted to
each row's `k` nearest neighbours is not a fixed operator — it changes with how densely that region
happens to be sampled, so two rows in different density regimes get kernels with different support,
and moving one point can flip an edge on the far side of the graph. The radius form keeps the kernel
fixed: the same physical distance means the same neighbourhood everywhere, and refining the sampling
converges to the same operator. That is the property the Nyström path relies on to sub-sample
landmarks and stay faithful to the full-graph operator.

In practice this means `edge_attr` should carry a **continuous function of relative position** —
typically the receiver's coordinates plus a unit direction, or a distance — so the kernel has
something smooth to interpolate across. If `edge_attr` only says "this is the 3rd neighbour", the
operator degenerates into an attention-like mechanism over a learned lookup, and the continuum
argument no longer applies.

```python
import faiss
import numpy as np
import torch
from torch_geometric.data import Data


def radius_graph(pos, x, radius):
    """Each row receives from every row within ``radius`` (self excluded)."""
    points = np.ascontiguousarray(pos, dtype=np.float32)
    index = faiss.IndexFlatL2(pos.shape[1])
    index.add(points)
    # IndexFlatL2 reports squared distances, so the search radius is squared.
    limits, _, found = index.range_search(points, float(radius) ** 2)
    receiver = np.repeat(np.arange(len(points), dtype=np.int64), np.diff(limits.astype(np.int64)))
    keep = found != receiver
    edge_index = torch.from_numpy(np.stack((found[keep], receiver[keep])).astype(np.int64))
    delta = pos[edge_index[0]] - pos[edge_index[1]]
    direction = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)
    edge_attr = torch.cat((pos[edge_index[1]], direction), dim=1)
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
```

Choosing `radius` is a modelling decision, not a tuning detail: it sets the operator's support. Too
small and rows genuinely have no neighbours; a uniform-random 200-point square has 172 of 200 rows
isolated at radius 2 and none at radius 10. A row with no incoming edge is embedded from its own
features alone through the root weight, so it is not broken — but it carries no neighbourhood
information, and on the merged workflow a queried row that no observed row reaches is predicted from
its `fill` substitute only. Check the in-degree distribution before trusting those rows.

### A k-NN builder, for easier implementation

A `k`-nearest-neighbour graph is **not** the intended architecture, but it is a legitimate shortcut
and the right choice in some cases:

- the sampling density is wildly uneven, so no single radius serves the whole domain;
- a plain index search is simpler to wire up than a radius sweep you then have to calibrate;
- you want a guaranteed in-degree of exactly `k` with no isolated rows, which is what makes a demo
  or a first prototype predictable.

Its costs are the ones listed above: the support varies with local density, the edge set is
discontinuous in the coordinates, and on the merged workflow every queried row necessarily attaches
to `k` senders however far away the nearest observed row is — which hides the "nothing observed
nearby" case a radius graph reports honestly.

```python
def knn_graph(pos, x, k=10):
    """Each row receives from its k nearest neighbours (self excluded)."""
    points = np.ascontiguousarray(pos, dtype=np.float32)
    index = faiss.IndexFlatL2(pos.shape[1])
    index.add(points)
    pairs = index.search(points, k + 1)[1][:, 1:]
    edge_index = torch.from_numpy(
        np.stack([pairs.reshape(-1), np.repeat(np.arange(pos.shape[0]), k)])
    )
    delta = pos[edge_index[0]] - pos[edge_index[1]]
    direction = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)
    edge_attr = torch.cat((pos[edge_index[1]], direction), dim=1)
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
```

Both builders are deliberately minimal; `examples/graph_utils.py` has the general versions
(`radius_edges`, `knn_edges`), including restricting senders to observed rows for the merged
workflow.

Either way `edge_attr` is entirely your choice — it is the kernel's input, not a distance. Position
of the receiver plus a unit direction is the convention the reference implementations use; a bare
distance, a physical quantity, or a one-hot edge type works just as well.

## Graph structure

A graph handed to `nofe` is usually in one of three shapes. The differences are which rows exist and
who is allowed to send, and they map directly onto which methods are legal.

| Shape | Rows | `edge_index` senders | Masks | `y` |
|---|---|---|---|---|
| **in-sample** | one per observed row | every row (builder's choice) | none | none |
| **merged** (out-of-sample with truth) | observed rows first, then queried rows | observed rows only | `input_mask`, `output_mask` | required for the queried rows |
| **blind** (out-of-sample, no truth) | observed rows first, then queried rows | observed rows only | `input_mask`, `output_mask` | absent |

The *merged* and *blind* shapes are the same graph; they differ only in whether the queried rows
carry `y`. That is deliberate — the edges never depend on the truth, so one graph serves training
and inference, and dropping `y` cannot change an embedding.

The `senders` column is the builder's job on every path. This is the single most important thing to
get right: **the dense operator never consults `input_mask`.** It aggregates over exactly the edges
you supply, so a withheld row that appears as a sender still sends. `input_mask` is provenance — it
marks which rows are observed — and only the Nyström path uses it structurally, to pick landmarks.
If withheld rows must not send, do not create edges out of them.

### The row contract for out-of-sample rows

| Row kind | `x` | `y` | `input_mask` | `output_mask` | sends | receives |
|---|---|---|---|---|---|---|
| observed | its features | its features (or omitted) | `True` | `True` | yes | yes |
| queried | a **substitute** | its truth when known | `False` | `True` | no | yes |

A queried row has no features of its own, so `x` must hold something. `nofe` never invents that
substitute: it consumes whatever you put there, and only validates that a graph whose scored rows
lie outside `input_mask` also carries `y` (otherwise those rows would be scored against a fabricated
target). The usual choices are zeros, the observed mean, or an inverse-distance-weighted mean of the
nearest observed rows — the last is what the reference builders use, because zeros make a withheld
row look like a genuine observation at the feature origin.

Because queried rows never send, a queried location with no observed row inside its neighbourhood
gets **no incoming edges at all**. Its embedding is then only what its substitute and the root weight
produce, so check in-degree before trusting such rows. The k-NN builder always attaches `k` senders
however far away the nearest observed row is; a fixed radius does not.

## Workflows

Four workflows, differing in whether rows are withheld and which operator runs. The operator is
fixed at construction by `config["nystrom"]` and is never a per-call switch — mixing modes between
training and inference silently changes the network's input statistics.

### 1. In-sample (the exact path)

Every row is observed. Build any neighbourhood, train, embed.

```python
from nofe import NOFE

model = NOFE(dim=2, config={"epochs": 150})
history = model.train(data)  # mean KL over output_mask rows
embedding = model.embed(data)  # [N, dim], CPU tensor
score = model.loss(data)  # same objective, no gradient
```

One forward pass covers the whole graph. The train/validation split selects *which scored rows*
enter the objective (`val_fraction`, drawn from `config["seed"]`), never which neighbours are
visible, so gradients flow through the entire edge set. This is the reference path: it is exact, and
it is what to reach for whenever the graph fits in memory.

### 2. Merged: out-of-sample with truth

Observed rows and queried rows in one graph, with `y` on the queried rows. Training scores **all**
scored rows, out-of-sample included, because each of them has a target distribution to match.

```python
merged = build_merged(...)  # input_mask/output_mask set, y set
model = NOFE(dim=2, config={"epochs": 150})
model.train(merged)  # scores observed + queried rows
score = model.loss(merged)
prediction = model.embed(merged)  # one row per output_mask row
```

This is how the operator learns to place locations it has never seen features for: the withheld rows
contribute to the objective through `y` while receiving messages from observed rows only.

### 3. Blind: true out-of-sample, embed only

Identical graph, no `y`. `embed` works unchanged; `train` and `loss` refuse it with an explicit
error, because the scored rows hold substitutes rather than truth.

```python
blind = build_merged(...)  # same rows and edges, no y
model.embed(blind)  # fine
model.train(blind)  # ValueError: ... no ground-truth features ...
```

A model trained on the merged graph is what you apply here: same weights, same edges, so the only
change between training and inference is the absence of the target.

### 4. Nyström: large graphs

`config={"nystrom": fraction}` swaps the dense operator for a landmark approximation. Landmarks are
drawn from `input_mask`, `round(fraction * |input_mask|)` of them, and are the **only senders**;
every `output_mask` row is a receiver. Each step builds an active subgraph holding the queries first
and then the landmarks not already among them, keeping your own edges whenever the sender is a
landmark and the receiver is in that set, with each `edge_attr` row copied unchanged. Landmarks
therefore receive as well as send when your graph has landmark-to-landmark edges — only sending is
restricted. Peak memory is `O(queries × mean_degree)`.

On the Nyström path, `pos` is required. Supervision needs a target distribution per query row, taken
from `y` when present and from `x` otherwise, which is why `y` is carried into the subgraph. The
same truth rule as the dense path applies: a graph whose scored rows lie outside `input_mask` and
which carries no `y` is refused outright (`ValueError: cannot score a graph with no ground-truth
features`) rather than silently scored against substitutes. When `y` is present, every
`output_mask` row is scored, out-of-sample included; without masks, the scored rows are the ones
owning their features.

`chunk_size` walks the queries in fixed-size blocks and changes peak memory only, never the result.
It matters most here: a small chunk rebuilds the active subgraph many times per epoch.


### Training on several graphs

Pass a list (or any iterable) of graphs to `train`, `loss` or `build` when one operator has to cover
several of them — one graph per date, per station set or per time window:

```python
model = NOFE(dim=2, config={"epochs": 150})
history = model.train([january, august])
score = model.loss([january, august])
embedding = model.embed(august)  # embed stays single-graph
```

- **One optimizer step per graph per epoch.** `N` graphs make an epoch `N` steps, so `epochs`,
  `patience` and `early_stopping` keep counting epochs rather than steps.
- **`history["train_loss"]` and `history["val_loss"]` are means over the graphs** of that epoch.
- **`loss` returns the unweighted mean over graphs**: each graph counts once, whatever its node
  count.
- **Every graph must share the `x` and `edge_attr` widths.** They are checked up front, before any
  optimizer step, and the error names the offending graph index.
- **`embed` takes one graph.** Loop over the list if you need every embedding: its return type would
  otherwise be one tensor or many depending on the argument.

The train/validation split and the Nyström landmark draws are seeded per graph, so a list is
reproducible run to run.

A single `Data` behaves exactly as before: one graph reduces to today's seeds and one step per epoch.

### Choosing between them

| Situation | Workflow |
|---|---|
| every location observed, graph fits in memory | in-sample, dense |
| every location observed, graph too large | in-sample, Nyström |
| some locations have no features but their truth is known | merged, dense or Nyström |
| deploying on locations with no features and no truth | blind (embed with a merged-trained model) |
| you need bit-exact, reproducible geometry | dense |

Dense and Nyström agree on the model input — withheld rows already hold your substitute in both — and
differ only in which rows send messages. Expect the Nyström embedding to be close, not identical.

## API

```python
NOFE(dim=2, *, config=None, **overrides)
```

`config` is a mapping and the keyword arguments are flat overrides; merge order is
`DEFAULTS` < `config` < keyword overrides. Unknown keys fail at construction.

| Member | Returns / raises | Notes |
|---|---|---|
| `.config` | `dict` | defaults + mapping + overrides + `dim` |
| `.model` | `KernelNN` | `RuntimeError` before build/train/embed/loss |
| `.device` | `torch.device` | resolved from `config["device"]` at construction |
| `.history` | `list[dict]` | `epoch`, `train_loss`, `val_loss`, `best_val_loss` |
| `.best_val_loss` | `float \| None` | best validation loss seen; `None` before training |
| `.build(data \| graphs)` | `KernelNN` | infers widths from the first graph; later calls reuse the weights |
| `.train(data \| graphs)` | `list[dict]` | trains in place; one optimizer step per graph per epoch; returns the history |
| `.embed(data)` | CPU `torch.Tensor` | `[num_output_nodes, dim]`, rows in `output_mask` order |
| `.loss(data \| graphs)` | `float` | mean KL over the scored rows of the trained operator; the mean over graphs for a list |
| `.save(path)` | `Path` | writes a `nofe-checkpoint-2` payload |
| `NOFE.load(path, device=None)` | `NOFE` | `weights_only=True` load, model in `eval()` mode; a `nofe-checkpoint-1` file is rejected with a retrain hint |

`embed` and `loss` both take their target from `y` when present and from `x` otherwise. `embed`
always returns one row per `output_mask` node, in that order, for both operators — so the same call
signature works for all four workflows, and a graph carrying masks needs no post-hoc filtering.

### Config keys

| Key | Default | Meaning |
|---|---|---|
| `width` | `32` | model width |
| `ker_width` | `64` | kernel MLP width |
| `depth` | `3` | number of convolution layers |
| `edge_batch_size` | `65536` | edge chunking during message passing; `None` disables it |
| `epochs` | `200` | training epochs |
| `lr` | `1e-2` | AdamW learning rate |
| `weight_decay` | `5e-4` | AdamW weight decay |
| `grad_clip` | `1.0` | gradient-norm clip; `None` disables it |
| `patience` | `20` | early-stopping patience |
| `min_delta` | `1e-6` | minimum validation improvement |
| `val_fraction` | `0.1` | share of the scored rows held out for validation |
| `early_stopping` | `True` | stop when validation stalls |
| `seed` | `42` | seed for python, numpy, torch and the landmark draws |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` or `"mps"` |
| `verbose` | `False` | print loss progress |
| `log_every` | `1000` | print every N epochs (the final epoch always prints) |
| `nystrom` | `None` | `None` = dense operator; a fraction in `(0, 1]` enables the Nyström approximation |
| `chunk_size` | `8192` | row chunk size for report and query walking; memory, not result |

All keys are validated at construction. Early stopping is on by default; with
`early_stopping=False` every epoch runs and the best epoch is **still** restored, so the final model
is deterministic given the seed. `edge_batch_size` and `chunk_size` bound peak memory without
changing the output.

## Platform notes

- Intel macOS is unsupported: no `torch` wheel ships for `macosx_*_x86_64` at any version ≥ 2.6.
- macOS arm64 requires macOS ≥ 14, because torch 2.15 publishes `macosx_14_0_arm64` wheels.
- CPU/CUDA choice happens at install time (`UV_TORCH_BACKEND`), device choice at runtime
  (`config={"device": ...}`). `"auto"` prefers CUDA, then MPS, then CPU.
- The caller's `Data` is cloned before it moves to the accelerator, so `data.x.numpy()` keeps
  working on the CPU copy.
- Checkpoints from the research repository use `NNConv_old` parameter names (`conv1.root`); map
  `root.T -> lin.weight` by hand — no compatibility loader ships.

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

`.vscode/settings.json` points VS Code at `.venv` and adds `src` to Pylance's search path, so the
integrated terminal, the test runner and IntelliSense all use this project's interpreter with no
picker step. It is a per-workspace override: if the status bar shows another interpreter, run
**Python: Select Interpreter** and choose `.venv/bin/python` once.

`examples/` holds a full worked pipeline — the ERA5 demo, its graph generator and the notebooks that
use them — with `examples/graph_utils.py` as the shared neighbourhood builder worth copying from.
The examples need the `examples` extra, which pulls `cartopy`, `matplotlib`, `scipy`, `netCDF4` and
`faiss-cpu`; `nofe` itself needs none of them.

- The `examples` extra pins `netCDF4<1.7.5`: 1.7.5+ publishes only macOS 15 wheels, so macOS 14
  cannot install it. Its `cartopy` dependency downloads Natural Earth shapefiles on first use into
  `~/.local/share/cartopy`.
- `faiss-cpu` is capped below 1.16: 1.16.0 publishes only a `macosx_15_0_arm64` wheel and no sdist,
  while torch 2.15 supports macOS 14.
- On macOS the `faiss-cpu` wheel vendors its own `libomp.dylib` and torch ships another; dyld loads
  both and the second initialisation aborts with `OMP: Error #15`. Run the repair after every
  `uv sync` (idempotent, a no-op off macOS): `uv run python tools/repair_faiss_omp.py`.
