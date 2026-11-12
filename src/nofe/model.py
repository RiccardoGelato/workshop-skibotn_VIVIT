"""Graph-kernel convolution and the embedding network."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import NNConv


class EdgeConv(NNConv):
    """``NNConv`` with edge-chunked kernel evaluation.

    A single dense ``width x width`` kernel over all edges is the peak-memory
    point of the model on MPS/CUDA; evaluating it in fixed-size edge slices keeps
    peak memory bounded without changing the result.
    """

    def __init__(
        self, channels: int, kernel: nn.Module, *, edge_batch_size: int | None = 32_768
    ) -> None:
        super().__init__(channels, channels, kernel, aggr="mean")
        self.edge_batch_size = edge_batch_size

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        limit = self.edge_batch_size
        if limit is None or edge_attr.shape[0] <= limit:
            return NNConv.message(self, x_j, edge_attr)
        return torch.cat(
            [
                NNConv.message(self, x_j[i : i + limit], edge_attr[i : i + limit])
                for i in range(0, edge_attr.shape[0], limit)
            ]
        )


def _edge_kernel(edge_channels: int, ker_width: int, width: int) -> nn.Module:
    """MLP mapping edge features to a flattened ``width x width`` kernel."""
    return nn.Sequential(
        nn.Linear(edge_channels, ker_width),
        nn.ReLU(),
        nn.Linear(ker_width, ker_width),
        nn.ReLU(),
        nn.Linear(ker_width, width * width),
    )


class KernelNN(nn.Module):
    """Graph-kernel operator trained to reproduce affinity neighbourhoods."""

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        out_dim: int = 2,
        *,
        width: int = 32,
        ker_width: int = 64,
        depth: int = 3,
        edge_batch_size: int | None = 32_768,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.lift = nn.Linear(in_channels, width)
        self.conv = EdgeConv(
            width,
            _edge_kernel(edge_channels, ker_width, width),
            edge_batch_size=edge_batch_size,
        )
        self.project = nn.Linear(width, out_dim)
        self._config = {
            "in_channels": in_channels,
            "edge_channels": edge_channels,
            "out_dim": out_dim,
            "width": width,
            "ker_width": ker_width,
            "depth": depth,
            "edge_batch_size": edge_batch_size,
        }

    @property
    def config(self) -> dict:
        """Constructor keyword arguments that rebuild this exact model."""
        return dict(self._config)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.lift(x)
        for _ in range(self.depth):
            h = self.conv(h, edge_index, edge_attr).relu()
        return self.project(h)
