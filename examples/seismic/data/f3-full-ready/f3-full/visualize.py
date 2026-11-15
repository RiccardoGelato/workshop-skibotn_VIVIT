"""Export native-resolution paired slices and amplitude/label/overlay figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from loader import ROOT, Volume

COLORS = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']


def export(dataset, volume, root=ROOT, output=None):
    data = Volume(dataset, volume, root)
    raw = Volume(dataset, volume, root, normalized=False)
    output = Path(output) if output else Path(__file__).resolve().parent / 'reports' / dataset / volume
    slices = data.path / 'slices'
    slices.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), 6)
    manifest = []
    for axis, direction in enumerate(('inline', 'crossline')):
        for fraction in (0.1, 0.5, 0.9):
            index = round((data.seismic.shape[axis] - 1) * fraction)
            a, y = data.slice(direction, index)
            r, _ = raw.slice(direction, index)
            stem = f'{direction}_{index:04d}'
            for name, values in [('seismic', a), ('seismic_raw', r), ('labels', y)]:
                np.save(slices / f'{stem}_{name}.npy', values, allow_pickle=False)
            fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout='constrained')
            for ax in (axes[0], axes[2]):
                ax.imshow(a, cmap='gray', vmin=-3, vmax=3, aspect='auto', interpolation='nearest')
            facies = axes[1].imshow(y, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            overlay = axes[2].imshow(y, cmap=cmap, norm=norm, alpha=0.35, aspect='auto', interpolation='nearest')
            for ax, title in zip(axes, ['Amplitude (z score; display clipped ±3)', 'Facies IDs', 'Overlay (alpha 0.35)']):
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Crossline array index' if axis == 0 else 'Inline array index')
                ax.set_ylabel('Vertical sample index (increasing down)')
            cb = fig.colorbar(facies, ax=axes, ticks=range(6), shrink=0.7)
            cb.set_label('Source class ID (0 is valid)')
            fig.suptitle(f'{dataset} / {volume} / {direction} index {index} — native pixels')
            fig.savefig(output / f'{stem}.png', dpi=130)
            plt.close(fig)
            manifest.append(dict(orientation=direction, index=index, shape=list(a.shape),
                                 transpose='volume[index,:,:].T' if axis == 0 else 'volume[:,index,:].T',
                                 class_ids=np.unique(y).tolist(), stem=stem))
    (output / 'slices.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (slices / 'slices.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest)} paired slices to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['f3', 'parihaka'], default='f3')
    parser.add_argument('--volume', default='full')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export(args.dataset, args.volume, args.data_root, args.output)
