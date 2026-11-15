# Full Netherlands F3 benchmark volume

This package contains the complete, unsplit F3 benchmark volume: 601 × 901 × 255 = 138,082,755 voxels. It is the cropped, depth-converted benchmark release, not the entire original acquisition survey. All six facies classes are present. No data download or preparation is required after extracting this archive.

## Load

Use Python 3.10 or later. NumPy is required for loading; Matplotlib is needed only for visualization. From the extracted folder:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

```python
from loader import Volume

v = Volume()  # defaults to f3/full in this standalone package
seismic, labels = v.seismic, v.labels
print(seismic.shape)  # (601, 901, 255)
image, mask = v.slice('inline', 300)     # (255, 901)
image, mask = v.slice('crossline', 450)  # (255, 601)
raw = Volume(normalized=False).seismic
```

The arrays are read-only NumPy memory maps. Loading them does not require reading the entire volume into RAM. Methods that copy the complete arrays will allocate that memory. All paths resolve relative to loader.py, so the package can be moved as a unit.

For direct loading:

```python
import numpy as np
seismic = np.load('data/processed/f3/full/seismic.npy', mmap_mode='r', allow_pickle=False)
labels = np.load('data/processed/f3/full/labels.npy', mmap_mode='r', allow_pickle=False)
```

## Geometry and arrays

Axes are `(inline, crossline, vertical_sample)` in the source array order. Inline slices use `volume[i,:,:].T`; crosslines use `volume[:,j,:].T`. Amplitudes and labels are sliced identically, with vertical samples increasing downwards. All indices are zero-based array indices. The vertical coordinate is a depth-domain sample index, not milliseconds. The NPY files do not establish physical coordinates, CRS, or spacing.

Under `data/processed/f3/full/`:

- `seismic_raw.npy`: supplied benchmark amplitudes converted to float32; 552,331,020 data bytes.
- `seismic.npy`: normalized float32 amplitudes; 552,331,020 data bytes.
- `labels.npy`: uint8 facies labels; 138,082,755 data bytes.
- `metadata.json`: shapes, classes, normalization, statistics, source comparisons, and caveats.

Each NPY file also has a small header. Allow about 1.25 GB of free space for extraction, plus space for the ZIP itself. The original float64 source arrays are not duplicated in this package; they can be downloaded from the original release or reproduced with the companion preparation repository. Maximum float32 amplitude conversion error is below 3e-8.

## Labels and normalization

Labels form a dense interpreted 3D geological model, not sparse 2D annotations or independently hand-labeled voxels. Source IDs are preserved:

| ID | Facies |
|---|---|
| 0 | Upper North Sea |
| 1 | Middle North Sea |
| 2 | Lower North Sea |
| 3 | Rijnland/Chalk |
| 4 | Scruff |
| 5 | Zechstein |

Class 0 is valid; it is not a generic missing-label or background sentinel.

Raw here means the supplied benchmark values, already scaled to [-1,1], not unprocessed acquisition amplitudes. Normalization uses `(source - mean) / population_std`, fitted on the original benchmark training subvolume: mean = 0.000999671080886209, std = 0.20976548783479312. These same parameters are applied throughout the full volume. Computation uses float64 and storage uses float32. There is no per-slice scaling or clipping; this full volume need not itself have exactly zero mean/unit standard deviation.

**Source padding caveat:** plane 0 contains zero amplitudes with supplied class-0 labels. The final vertical plane, index 254, also contains zeros/class-0 labels in the unsplit source. The separate original train/test files instead repeat plane 253 at index 254. This final plane is suspect padding, preserved without repair and flagged by `metadata['suspect_vertical_planes']`. All samples 0–253 match the corresponding source split regions exactly. Numeric label coverage is 100%, which does not establish a usable annotation on the suspect final plane. Whether to exclude it is left explicit for the user; no pixels have been silently removed or relabeled.

## Visualize

Six amplitude/label/alpha-0.35-overlay PNGs are already in `reports/f3/full/`, covering three inline and three crossline positions. To regenerate them and export paired NumPy slices:

```bash
python visualize.py --dataset f3 --volume full
```

Slice arrays will be written beneath `data/processed/f3/full/slices/`. `slices.json` records their exact indices. Only figure display clips normalized amplitudes to ±3; stored amplitudes are unchanged.

Validation scanned every voxel for finite amplitudes, matching dimensions, integral class IDs, label coverage, and normalization round trips. Source geometry was checked against all three benchmark splits, with the documented final-plane discrepancy. Example slices in both directions were visually inspected for obvious transposition or reversal. Validation is not independent verification of the geological interpretation. See `reports/full-validation.json` and `reports/full-roundtrip.json`.

## Provenance and licensing

Source: Alaudah et al., Facies Classification Benchmark, https://zenodo.org/records/3755060, `raw.zip`. The record declares MIT in its machine-readable license metadata. The original repository's MIT notice and the record snapshot are included in `provenance/`; the software license for these preparation utilities is in LICENSE. Changes are float32 conversion, documented normalization, and slice/figure exports. Native volume geometry and numeric label IDs are preserved.

Please cite: Alaudah, Y., Michałowicz, P., Alfarraj, M., and AlRegib, G. (2019), A machine-learning benchmark for facies classification, Interpretation 7(3), SE175–SE187, https://doi.org/10.1190/INT-2018-0249.1.

Archive source MD5: afe9183d2d454573f8a132e3c97d10ee. Calculated SHA256 values are in provenance/downloaded-files.json. This package is separate from the source-only preparation repository so large data need not be committed to Git.
