# Seismic example: embedding F3 traces

The Netherlands F3 facies benchmark as a NOFE demonstration. `scripts/seismic.py` trains one
operator over the cube and draws its 3-D embedding as an RGB map.

## The setup

The cube is 601 inlines x 901 crosslines x 255 vertical samples on a 25 m bin grid.

- **Domain — the horizontal plane.** One graph node per (inline, crossline) location, so a node is a
  whole seismic trace and distances in the graph are physical (the survey is 15 km x 22.5 km).
- **Feature space — the vertical axis.** Each node's `x` is its 253 vertical amplitude samples, so
  the embedding compresses a trace from 253 numbers to 3.

A radius neighbourhood over a regular raster is exactly a fixed set of (row, column) offsets, so
`utilities/f3_graph.py` derives the edges analytically from the bin size instead of searching an
index: exact, and cheap even for 541,501 nodes. A 50 m radius is 2 bins, i.e. 12 neighbours per
trace, at every interior node.

Training and inference use the **same graph — all 541,501 traces**, so the reported KL is not an
artefact of scoring a different neighbourhood than the one trained on. `--stride 2` halves the
graph for a quick look.

## Data

`data/f3-full-ready/` holds the vendored f3-full package (upstream `loader.py`, `visualize.py`,
reports, provenance and license, MIT — see `data/f3-full-ready/f3-full/LICENSE` and its README).
The three `.npy` arrays (1.2 GB) are git-ignored; copy the package in before running:

```
examples/seismic/data/f3-full-ready/f3-full/data/processed/f3/full/{seismic,seismic_raw,labels}.npy
```

Two facts from the shipped metadata drive the code:

- **Vertical planes 0 and 254 are padding** — constant amplitude, class-0 labels throughout. They
  are dropped (`f3_graph.PADDING_PLANES`); a trace built from them would encode the padding.
- `seismic.npy` is already `(seismic_raw - 0.0009997) / 0.2097655`. The loader standardises each
  *vertical sample* across traces with global constants, fitted once and reused at any stride, so
  shallow and deep amplitude stay comparable.

## Run

```bash
uv run python examples/seismic/scripts/seismic.py                  # full cube, ~11 min on CPU
uv run python examples/seismic/scripts/seismic.py --stride 2 --epochs 10   # quick look
```

`--radius-km` sets the lateral support, `--epochs` the training budget. The graph, the model and the
embedding all cover the whole cube at `--stride 1`.

Outputs two files beside the script:

- `f3_embedding.png`: one RGB colour per trace, R/G/B the three embedding dimensions, one
  `pcolormesh` cell per 25 m bin;
- `f3_embedding.npz`: the embedding plus its raster axes, for the clustering notebook.

## Cluster the embedding

`notebooks/cluster-maps.py` is a [marimo](https://marimo.io) notebook that k-means the embedding at
several `k` and draws each labelling back onto the survey:

```bash
uv run marimo edit examples/seismic/notebooks/cluster-maps.py
```

It reads the cache above, so it never retrains (~10 s). `k`, the k-means seed and the number of
points used to fit centres are interactive; every fit is deterministic in the seed, so comparing
`k` against `k` is fair. `utilities/clusters.py` holds k-means, the sweep and the palette, so the
notebook and any script share one implementation.
