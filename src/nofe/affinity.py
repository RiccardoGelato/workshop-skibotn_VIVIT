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
    *,
    chunk_size: int | None = 1_048_576,
) -> tuple[Tensor, Tensor]:
    """Normalised student-t affinities ``p[j | i]`` per directed edge.

    Returns the per-edge weights and a per-node mask of nodes with at least one
    incoming edge. Weights for each target node sum to one.

    The squared difference is accumulated in edge chunks of ``chunk_size`` (``None`` = one
    pass): a dense graph on wide features would otherwise materialise two ``[E, F]`` tensors at
    once — 3.3 GB apiece on the reference F3 graph — and the result is identical either way.
    """
    if num_nodes is None:
        num_nodes = int(values.shape[0])
    if edge_index.numel() == 0:
        return values.new_zeros(0), torch.zeros(num_nodes, dtype=torch.bool, device=values.device)

    src, dst = edge_index[0], edge_index[1]
    num_edges = src.shape[0]
    if chunk_size is None or num_edges <= chunk_size:
        weights = 1.0 / (1.0 + (values[src] - values[dst]).square().sum(dim=1))
    else:
        weights = values.new_empty(num_edges)
        for start in range(0, num_edges, chunk_size):
            stop = start + chunk_size
            squared = (values[src[start:stop]] - values[dst[start:stop]]).square()
            weights[start:stop] = 1.0 / (1.0 + squared.sum(dim=1))
    normalizer = values.new_zeros(num_nodes)
    normalizer.index_add_(0, dst, weights)
    connected = torch.zeros(num_nodes, dtype=torch.bool, device=values.device)
    connected[dst] = True
    return weights / (normalizer[dst] + eps), connected


def per_query_kl(
    target: Tensor | None,
    prediction: Tensor,
    edge_index: Tensor,
    num_nodes: int | None = None,
    query_mask: Tensor | None = None,
    eps: float = DEFAULT_EPS,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-query-node KL divergence between target and predicted affinities.

    ``weights`` is a precomputed ``affinities(...)[0]`` for ``target``: supplying it skips that call
    and lets ``target`` be ``None``, which is how a graph that carries its own affinities is scored.
    """
    if weights is None and target is None:
        raise ValueError("target is required unless weights are supplied")
    if num_nodes is None:
        num_nodes = int(prediction.shape[0] if target is None else target.shape[0])
    values = prediction.new_zeros(num_nodes)
    if weights is None:
        p, connected = affinities(target, edge_index, num_nodes, eps)
    else:
        p = weights
        # The valid-query set comes from the edges alone, so it is the same mask either way.
        connected = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        connected[edge_index[1]] = True
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
    target: Tensor | None,
    prediction: Tensor,
    edge_index: Tensor,
    num_nodes: int | None = None,
    query_mask: Tensor | None = None,
    eps: float = DEFAULT_EPS,
    weights: Tensor | None = None,
) -> Tensor:
    """Mean KL divergence over the query nodes that have neighbourhoods."""
    values, valid = per_query_kl(
        target, prediction, edge_index, num_nodes, query_mask, eps, weights
    )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return values[valid].mean()
