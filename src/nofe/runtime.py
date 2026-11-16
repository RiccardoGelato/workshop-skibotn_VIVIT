"""Device selection, seeding and device transfer."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Data

DEVICES = ("auto", "cpu", "cuda", "mps")
_FLOAT_FIELDS = ("x", "pos", "pos_raw", "edge_attr", "edge_dist", "affinity")


def get_device(spec: str = "auto") -> torch.device:
    """Resolve a device specifier to a concrete ``torch.device``."""
    if spec not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {spec!r}")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if spec == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("device='mps' requested but MPS is not available")
        return torch.device("mps")
    if spec == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all available backends)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def to_device(data: Data, device: torch.device) -> Data:
    """Clone ``data`` onto ``device`` with its array attributes cast to ``float32``.

    ``affinity`` is a target rather than a model input, so it stays ``float32`` too; ``edge_index``
    and the metadata attributes are copied unchanged.
    """
    moved = data.clone().to(device)
    for name in _FLOAT_FIELDS:
        value = getattr(moved, name, None)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(moved, name, value.float())
    return moved
