"""Read prepared volumes as read-only memory maps. No model dependencies."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / 'data' / 'processed'


class Volume:
    def __init__(self, dataset='f3', volume='full', root=ROOT, normalized=True):
        self.path = Path(root) / dataset / volume
        self.metadata = json.loads((self.path / 'metadata.json').read_text())
        filename = 'seismic.npy' if normalized else 'seismic_raw.npy'
        self.seismic = np.load(self.path / filename, mmap_mode='r', allow_pickle=False)
        self.labels = np.load(self.path / 'labels.npy', mmap_mode='r', allow_pickle=False)
        if self.seismic.shape != self.labels.shape or self.seismic.ndim != 3:
            raise ValueError('Expected matching three-dimensional seismic and labels')

    def slice(self, orientation, index):
        """Return paired (amplitude, label) views with vertical samples in rows.

        Indices are zero-based array indices, not survey line numbers.
        inline: volume[index, :, :].T; crossline: volume[:, index, :].T.
        The inline/crossline names for Parihaka follow the redistribution;
        their physical survey interpretation is unverified (see README).
        """
        if orientation not in ('inline', 'crossline'):
            raise ValueError('orientation must be inline or crossline')
        axis = 0 if orientation == 'inline' else 1
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.seismic.shape[axis]:
            raise IndexError(f'{orientation} index must be in [0, {self.seismic.shape[axis]})')
        key = (index, slice(None), slice(None)) if axis == 0 else (slice(None), index, slice(None))
        return self.seismic[key].T, self.labels[key].T
