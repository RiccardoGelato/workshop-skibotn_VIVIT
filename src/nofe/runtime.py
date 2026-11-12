"""Device selection, seeding, graph validation and device transfer."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Data

DEVICES = ("auto", "cpu", "cuda", "mps")
_FLOAT_FIELDS = ("x", "pos", "edge_attr", "edge_dist")


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


def _check_edge_index(data: Data) -> int:
    """Validate ``edge_index`` shape/dtype/range; return the edge count."""
    edge_index = data.edge_index
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
        )
    if edge_index.dtype != torch.long:
        raise ValueError(f"edge_index must be torch.long, got {edge_index.dtype}")
    if edge_index.numel() > 0 and int(edge_index.max()) >= data.num_nodes:
        raise ValueError("edge_index references node ids outside [0, num_nodes)")
    return int(edge_index.shape[1])


def validate_graph(data: Data) -> None:
    """Check that ``data`` satisfies the documented graph contract."""
    if not isinstance(data, Data):
        raise TypeError(f"expected torch_geometric.data.Data, got {type(data).__name__}")

    required = ("x", "edge_index", "edge_attr")
    missing = tuple(name for name in required if getattr(data, name, None) is None)
    if missing:
        raise ValueError(
            f"graph requires fields {missing}; build them with nofe.build_graph, "
            "nofe.graphs.radius_graph or nofe.graphs.knn_graph"
        )

    if data.x.dim() != 2:
        raise ValueError(f"x must be 2-D [num_nodes, features], got shape {tuple(data.x.shape)}")

    num_nodes = data.num_nodes
    num_edges = _check_edge_index(data)
    edge_attr = data.edge_attr
    if edge_attr.dim() != 2 or edge_attr.shape[0] != num_edges:
        raise ValueError(
            f"edge_attr must have shape [num_edges, edge_features], got {tuple(edge_attr.shape)}"
        )

    pos = getattr(data, "pos", None)
    if pos is not None and pos.dim() != 2:
        raise ValueError(
            f"pos must be 2-D [num_nodes, position_dims], got shape {tuple(pos.shape)}"
        )

    for name in ("input_mask", "output_mask"):
        mask = getattr(data, name, None)
        if mask is None:
            continue
        if mask.dim() != 1 or mask.shape[0] != num_nodes:
            raise ValueError(f"{name} must have {num_nodes} entries, got {tuple(mask.shape)}")

    input_mask = getattr(data, "input_mask", None)
    output_mask = getattr(data, "output_mask", None)
    if input_mask is not None and output_mask is not None:
        selected = input_mask.to(dtype=torch.bool) & torch.logical_not(
            output_mask.to(dtype=torch.bool)
        )
        if bool(selected.any()):
            raise ValueError("input_mask must be a subset of output_mask")

    target = getattr(data, "y", None)
    if target is not None and (target.dim() != 2 or tuple(target.shape) != tuple(data.x.shape)):
        raise ValueError(
            f"y must have the same shape as x {tuple(data.x.shape)}, got {tuple(target.shape)}"
        )


def to_device(data: Data, device: torch.device) -> Data:
    """Clone ``data`` onto ``device`` with float fields cast to ``float32``."""
    moved = data.clone().to(device)
    for name in _FLOAT_FIELDS:
        value = getattr(moved, name, None)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(moved, name, value.float())
    return moved
