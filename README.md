# nofe

Neural operator function embedding: graph-kernel dimensionality reduction.

`nofe` takes point features supplied at point locations, learns a low-dimensional embedding with a
graph-kernel operator (GKO), and hands that embedding back. Training makes the *neighbourhood
structure of the embedding* reproduce the *neighbourhood structure of the input features*. For a
node $i$, with $p_j$ the student-t affinity between feature vectors $x_i$ and $x_j$, and $q_j$ the
same kernel on the learned embedding $z$,

$$\mathrm{KL}(p_i \,\|\, q_i) = \sum_{j \in \mathcal{N}(i)} p_j \log \frac{p_j}{q_j}, \qquad
q_j \propto \frac{1}{1 + \lVert z_i - z_j \rVert^2}$$

Distances are normalised per node, so neighbourhoods of any degree are comparable, and the loss is
the mean over nodes rather than a sum over edges.

One case, one operator:

- **Every supplier of features is also a location.** Nothing is withheld, nothing is queried, and the
  node set is exactly the rows you supplied. A `Data` therefore carries no masks and no substitute
  feature rows.
- **The graph decides what "neighbour" means.** A radius, a k-nearest set, a mesh, a physical model —
  whatever the builder produces is the neighbourhood the embedding preserves.
- **The graph carries its own target.** `data.affinity` holds the per-edge student-t weights of `x`;
  the loss reads them from the graph, so training and scoring need no truth anywhere else.

## Install

`nofe` is not on PyPI, so `uv add nofe` fails with `nofe was not found in the package registry`.
Three installs, depending on what you need.

### Just the library

Add it to your own project from the repository:

```bash
uv add "nofe @ git+ssh://git@github.com/laueb4818/workshop-skibotn.git"
```

Then `import nofe` works and nothing else is needed. The repository is private, so this needs your
SSH key; with a token use `git+https://<token>@github.com/laueb4818/workshop-skibotn.git`.

`uv` records the resolved commit in your `uv.lock`, so installs are reproducible until you run
`uv lock --upgrade`. Put `@<commit-sha>` in the URL when you want that pin in `pyproject.toml`.

This install has no `examples/` — that directory lives outside `src/`, so the ERA5 and F3 demos,
their data and `examples/graph_utils.py` are not included. If you only want the package, that is the
whole story.

### The repository, examples included

Clone and sync if you want the demos, their data, or a copy of the code to read:

```bash
git clone git@github.com:laueb4818/workshop-skibotn.git
cd workshop-skibotn
uv sync --all-groups --all-extras
```

Substitute the sync line for your platform:

| Target | Command |
|---|---|
| any platform, CPU | `UV_TORCH_BACKEND=cpu uv sync --all-groups --all-extras` |
| Linux x86_64 + NVIDIA | `UV_TORCH_BACKEND=cu128 uv sync --all-groups --all-extras` (also `cu126`, `cu130`) |
| macOS arm64 | `uv sync --all-groups --all-extras` (Metal/MPS is enabled in the default macOS wheel) |
| Windows AMD64 | `UV_TORCH_BACKEND=cpu uv sync --all-groups --all-extras` |

`--all-extras` is what pulls `cartopy`/`matplotlib`/`netCDF4` for the examples; the library itself
needs none of them. Both ERA5 dates ship in the clone, so the ERA5 pipeline runs with no download
step; the 1.2 GB F3 volume is not tracked and `examples/seismic/README.md` says where to put it.

### Editing `nofe` from another project

If you are changing the package and want your project to see the changes live, install the clone as
an editable dependency:

```bash
uv add --editable /path/to/workshop-skibotn
```

Edits under `src/nofe/` take effect on the next run; no reinstall, no version bump.

`torch` is pinned as `torch>=2.6,<3`; the CUDA variant is selected by `UV_TORCH_BACKEND` at install
time and the shipped `uv.lock` is backend-independent. Requires Python ≥ 3.11. Runtime dependencies
are `torch`, `torch-geometric`, `numpy`, `scipy`, and `faiss-cpu` everywhere except macOS — see
[Neighbour backend](#neighbour-backend).

## Neighbour backend

The neighbourhood search runs on one of two exact backends, chosen at runtime:

| Platform | Default | Why |
|---|---|---|
| macOS | `scipy.spatial.cKDTree` | works out of the box |
| Linux, Windows | `faiss` | faster index |

faiss is not installed on macOS by default, because its wheel vendors its own `libomp.dylib` and
dyld aborts on the second OpenMP initialisation next to torch's (`OMP: Error #15`). That abort
happens at **import** time, so a fallback inside the import could never run — the platform is
inspected first and faiss is simply never imported. A plain `uv sync` on macOS therefore needs no
repair step, and `nofe` never touches faiss there.

Both backends return **byte-identical edge sets** for the same input: radius queries are compared as
squared float32 distances with `<` (a point at exactly `radius` is excluded, as faiss does), and
nearest-neighbour ties resolve by sender id rather than by whichever order the library happened to
emit. `tests/test_graph_utils.py` cross-checks them on random points, a raster band, and a lattice
full of exact ties.

### Using faiss on macOS

Optional, and only a speed choice — the graphs are the same either way.

```bash
uv sync --extra faiss          # installs faiss-cpu alongside the project
uv run python tools/repair_faiss_omp.py   # link faiss's libomp at torch's (idempotent)
NOFE_BACKEND=faiss uv run python examples/era5/scripts/era5.py
```

`tools/repair_faiss_omp.py` replaces `faiss/.dylibs/libomp.dylib` with a symlink to
`torch/lib/libomp.dylib`, so one OpenMP runtime is loaded instead of two. It locates both through
`importlib.util.find_spec` and never imports faiss, so it is safe to run before the repair has
anything to fix. It is a no-op off macOS and is idempotent.

`auto` still picks cKDTree on macOS after the repair, because it cannot know the repair was applied.
Select faiss explicitly with `NOFE_BACKEND=faiss` or `backend("faiss")`; both raise if faiss is
missing rather than silently falling back.

## Providing data

`PointGraph` is the single entry point: one node per feature row, at that row's location.

```python
import numpy as np

from nofe import PointGraph
from nofe.search import RadiusIndex, sorted_pairs


class RadiusGraph(PointGraph):
    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def edge_parameters(self) -> tuple[float, int]:
        return self.radius, 0  # (radius, k) handed to neighbour_pairs

    def neighbour_pairs(self, points, radius, k):
        """(sender, receiver) ids: each row receives from every row within ``radius``."""
        rows = np.arange(points.shape[0])
        sender, receiver = RadiusIndex(points, rows).within(rows, radius)
        return sorted_pairs(sender, receiver)


graph = RadiusGraph(radius=0.5).build(features, locations)
```

`RadiusIndex`/`NearestIndex` (in `nofe.search`) are the backend-neutral searches: they pick faiss or
cKDTree for you and return the same pairs on both. `examples/graph_utils.py` wraps them as
`radius_edges`/`knn_edges`, which is what the two example subclasses call.

`features` and `locations` are any array-likes with one row per point — the same number of rows,
`locations` in any dimension. `examples/` holds two full subclasses to copy from.

The edge features are `[2P + 1]`-wide for `P` location dimensions. `build` returns a
`torch_geometric.data.Data` with:

| Field | Shape / dtype | Meaning |
|---|---|---|
| `x` | `[N, F]` float32 | the features, normalised when `normalize=True` |
| `pos` | `[N, P]` float32 | the locations used to search, normalised when `normalize=True` |
| `pos_raw` | `[N, P]` float32 | the locations you passed, unchanged and row-aligned with `pos` |
| `edge_index` | `[2, E]` int64 | `[sender, receiver]`, sorted by `(receiver, sender)` |
| `edge_attr` | `[E, 2P + 1]` float32 | `[pos[receiver], unit direction, relative distance]` |
| `edge_dist` | `[E]` float32 | length of each edge in the scaled space |
| `affinity` | `[E]` float32 | the loss target: student-t weights of `x` per edge |

### Normalization

`normalize=True` (the default) z-scores **features and locations per column over all supplied
rows**, before any sampling; `normalize=False` leaves both arrays exactly as you passed them and the
caller owns scaling. The statistics always come from the whole array, never from the sample, because
a subsample's mean/std would make the feature space depend on `sample`.

The consequence worth knowing: the model sees the normalised numbers. A graph built with different
constants than the model was trained on — a different date whose fields were standardised
differently, a subset rescaled on its own — lands in a different space and embeds differently. Pass
`normalize=False` and scale consistently yourself when several graphs have to share one space.

### Sampling

`sample=None` uses every supplied row. `sample=N` draws `N` rows at random (`seed`) and keeps them in
ascending row order, so the graph is reproducible and `edge_index` stays comparable across builds.
Sampling bounds build time and memory on large point sets: a 541,501-point graph costs a nearest-
neighbour search over everything, which is exactly the step a subsample shrinks.

## API

```python
KernelNN(out_dim=3, *, in_channels=None, edge_channels=None, width=32, ker_width=64, depth=3,
         edge_batch_size=65_536)
```

The `nn.Module` operator: a lift, `depth` graph-kernel convolutions and a projection. `in_channels`
(node features) and `edge_channels` (edge features) may be omitted — they are discovered on the
first forward pass and readable afterwards as `net.in_channels` / `net.edge_channels`, which raise
`RuntimeError` until then. Self terms are implicit, so a node needs no self loop, and a node with no
incoming edge is still embedded from its own features through the root weight.

```python
NOFE(out_dim=3, *, in_channels=None, edge_channels=None, width=32, ker_width=64, depth=3,
     edge_batch_size=65_536, nystrom=None, chunk_size=8192, device="auto")
```

The trainer. It holds the network as `.net` and is not itself an `nn.Module` — `nn.Module` owns the
attribute `training` (a bool), so a driver method with that name has to live elsewhere.

| Member | Returns | Notes |
|---|---|---|
| `.net` | `KernelNN` | the torch module; `torch.save(model.net.state_dict(), path)` is the checkpoint |
| `.device` | `torch.device` | resolved from `device` at construction |
| `.training(graphs, **params)` | `list[dict]` or `list[list[dict]]` | trains in place; one epoch keyed `epoch`, `train_loss`, `val_loss`, `best_val_loss` per graph |
| `.embed(graphs)` | `torch.Tensor` or `list[torch.Tensor]` | CPU tensor(s), `[num_nodes, out_dim]`, in node order |
| `.fit(graph, **params)` | `torch.Tensor` | train on that one graph, then embed it |
| `.loss(graphs)` | `float` or `list[float]` | mean KL read from each graph's `affinity` |

Each of them accepts a single `Data` or an iterable of them and returns the matching type:
one history versus one history per graph, one tensor versus a list, a float versus a list of floats.
`training` takes `epochs=200, lr=1e-2, weight_decay=5e-4, grad_clip=1.0, patience=20, min_delta=1e-6,
val_fraction=0.1, early_stopping=True, verbose=False, log_every=1000, seed=42`.

With `N` graphs an epoch is `N` optimizer steps, one per graph; the reported `train_loss` and
`val_loss` are the means over graphs, and early stopping restores the weights of the best epoch on
that mean. The validation split and the landmark draws are seeded per graph, so a run is
reproducible.

Because the trainer is a plain object, checkpoints are the network's state dict:
`torch.save(model.net.state_dict(), path)`, and `NOFE(..., in_channels=F, edge_channels=D).net
.load_state_dict(...)` reads it back. Pass the widths on reload: an uninitialised lazy parameter
cannot be restored with `weights_only=True`.

### Nyström: large graphs

`NOFE(..., nystrom=fraction)` swaps the dense operator for a landmark approximation.
`round(fraction * num_nodes)` landmarks are drawn from all nodes and are the **only senders**; every
node is a receiver. Each step builds an active subgraph holding the queries first and then the
landmarks not already among them, keeping the graph's own edges whenever the sender is a landmark and
the receiver is in that set, with each `edge_attr` row copied unchanged. Landmarks therefore receive
as well as send when the graph has landmark-to-landmark edges — only sending is restricted. Peak
memory is `O(queries × mean_degree)`.

`chunk_size` walks the queries in fixed-size blocks and changes peak memory only, never the result.
It matters most here: a small chunk rebuilds the active subgraph many times per epoch. `loss` uses
the same path training did, so on this approximation the reported KL describes the operator that
was actually fit, not a dense pass it never ran.

The approximation assumes the radius form. A kernel restricted to each row's `k` nearest neighbours
changes support with local sampling density, so two rows in different density regimes get different
kernels and landmark sub-sampling is no longer faithful to the full-graph operator.

## Platform notes

- Intel macOS is unsupported: no `torch` wheel ships for `macosx_*_x86_64` at any version ≥ 2.6.
- macOS arm64 requires macOS ≥ 14, because torch 2.15 publishes `macosx_14_0_arm64` wheels. The
  neighbourhood backend there is `scipy.spatial.cKDTree`; faiss is an opt-in speed choice, see
  [Neighbour backend](#neighbour-backend).
- CPU/CUDA choice happens at install time (`UV_TORCH_BACKEND`), device choice at runtime
  (`NOFE(..., device=...)`). `"auto"` prefers CUDA, then MPS, then CPU.
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

`examples/` holds worked pipelines — the ERA5 demo and the F3 seismic cube, each with a
`PointGraph` subclass in its `utilities/` directory and its notebooks. The examples need the
`examples` extra, which pulls `cartopy`, `matplotlib` and `netCDF4`.

The two ERA5 dates the demo and its notebooks use (`2018-01-01`, `2019-08-01`) are tracked in
`examples/era5/data/`, 21 MB each, and that directory is the default data path, so the ERA5 pipeline
runs from a fresh clone with no download step. `--data` (env `NOFE_ERA5_DATA`) points it at a
different directory. The 1.2 GB F3 volume is not tracked; `examples/seismic/README.md` says where to
put it.

- The `examples` extra pins `netCDF4<1.7.5`: 1.7.5+ publishes only macOS 15 wheels, so macOS 14
  cannot install it. Its `cartopy` dependency downloads Natural Earth shapefiles on first use into
  `~/.local/share/cartopy`.
- `faiss-cpu` is capped below 1.16: 1.16.0 publishes only a `macosx_15_0_arm64` wheel and no sdist,
  while torch 2.15 supports macOS 14. It is not installed on macOS by default; see
  [Neighbour backend](#neighbour-backend) to opt in, and run
  `uv run python tools/repair_faiss_omp.py` after installing it there.
