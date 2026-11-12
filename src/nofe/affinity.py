"""Student-t affinities and the per-query KL objective."""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_EPS = 1e-6


def affinities(
    values: Tensor,
    edge_index: Tensor,
    num_nodes: int | None = None,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor]:
    """Normalised student-t affinities ``p[j | i]`` per directed edge.

    Returns the per-edge weights and a per-node mask of nodes with at least one
    incoming edge. Weights for each target node sum to one.
    """
    if num_nodes is None:
        num_nodes = int(values.shape[0])
    if edge_index.numel() == 0:
        return values.new_zeros(0), torch.zeros(num_nodes, dtype=torch.bool, device=values.device)

    src, dst = edge_index[0], edge_index[1]
    weights = 1.0 / (1.0 + (values[src] - values[dst]).square().sum(dim=1))
    normalizer = values.new_zeros(num_nodes)
    normalizer.index_add_(0, dst, weights)
    connected = torch.zeros(num_nodes, dtype=torch.bool, device=values.device)
    connected[dst] = True
    return weights / (normalizer[dst] + eps), connected


def per_query_kl(
    target: Tensor,
    prediction: Tensor,
    edge_index: Tensor,
    num_nodes: int | None = None,
    query_mask: Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor]:
    """Per-query-node KL divergence between target and predicted affinities."""
    if num_nodes is None:
        num_nodes = int(target.shape[0])
    values = target.new_zeros(num_nodes)
    p, connected = affinities(target, edge_index, num_nodes, eps)
    valid = connected
    if query_mask is not None:
        valid = connected & query_mask.to(connected.device)
    if not bool(valid.any()):
        return values, valid

    q, _ = affinities(prediction, edge_index, num_nodes, eps)
    edges = valid[edge_index[1]]
    kl = p[edges] * (torch.log(p[edges] + eps) - torch.log(q[edges] + eps))
    values.index_add_(0, edge_index[1, edges], kl)
    return values, valid


def affinity_kl_loss(
    target: Tensor,
    prediction: Tensor,
    edge_index: Tensor,
    num_nodes: int | None = None,
    query_mask: Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Mean KL divergence over the query nodes that have neighbourhoods."""
    values, valid = per_query_kl(target, prediction, edge_index, num_nodes, query_mask, eps)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return values[valid].mean()
