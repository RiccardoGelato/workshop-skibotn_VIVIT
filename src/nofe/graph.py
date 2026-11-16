"""The graph builder base class: one node per feature row, at that row's location.

A ``PointGraph`` turns ``(features, locations)`` into the ``torch_geometric.data.Data`` the
operator consumes: normalised features, normalised and original locations, radius or k-NN edges,
geometric edge features, and the pairwise affinities the local KL objective targets. Subclasses own
exactly two rules — how neighbours are found, and which radius/k to use — and inherit the rest.
"""

from __future__ import annotations

import abc

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from .affinity import affinities

EDGE_DIST_EPS = 1e-8


def _as_matrix(values: Tensor | np.ndarray, name: str) -> np.ndarray:
    """``values`` as a float32 2-D array, or ``ValueError`` naming the expected shape."""
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D [{name}_rows, dims], got shape {array.shape}")
    return array


def _geometry_edge_attr(
    pos: Tensor,
    edge_index: Tensor,
    edge_dist: Tensor | None = None,
    eps: float = EDGE_DIST_EPS,
) -> Tensor:
    """Edge features ``[pos[dst], direction, relative distance]``, shape ``[E, 2P + 1]``."""
    if pos.dim() != 2:
        raise ValueError(
            f"pos must be 2-D [num_nodes, position_dims], got shape {tuple(pos.shape)}"
        )
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got shape {tuple(edge_index.shape)}"
        )
    if edge_index.numel() == 0:
        return pos.new_empty((0, 2 * pos.shape[1] + 1))

    src, dst = edge_index[0], edge_index[1]
    delta = pos[src] - pos[dst]
    norm = torch.linalg.vector_norm(delta, dim=1)
    if edge_dist is None:
        edge_dist = norm
    direction = delta / (norm + eps).unsqueeze(1)
    max_dist = pos.new_full((pos.shape[0],), -torch.inf)
    max_dist.scatter_reduce_(0, dst, edge_dist, reduce="amax", include_self=True)
    relative = (edge_dist / (max_dist[dst] + eps)).unsqueeze(1)
    return torch.cat((pos[dst], direction, relative), dim=1)


class PointGraph(abc.ABC):
    """Build a graph from ``(features, locations)``: one node per row, at any location dimension.

    Every row is also a location: nothing is withheld and nothing is queried, so there is no mask,
    no substitute feature vector and no separate truth. A subclass supplies the two neighbourhood
    rules (:meth:`neighbour_pairs`, :meth:`edge_parameters`); :meth:`build` does the rest.

    Normalisation statistics are computed from the **whole supplied array, never from the sample**:
    a subsample's mean/std would make the feature space depend on ``sample``, so two graphs built
    from the same data with different sample sizes would no longer share one space.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        sample: int | None = None,
        edge_dist_eps: float = EDGE_DIST_EPS,
        seed: int = 0,
    ) -> None:
        if sample is not None and (
            not isinstance(sample, int) or isinstance(sample, bool) or sample < 1
        ):
            raise ValueError(f"sample must be None or a positive integer, got {sample!r}")
        if not isinstance(edge_dist_eps, (int, float)) or edge_dist_eps <= 0:
            raise ValueError(f"edge_dist_eps must be a positive float, got {edge_dist_eps!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self.normalize = bool(normalize)
        self.sample = sample
        self.edge_dist_eps = float(edge_dist_eps)
        self.seed = int(seed)

    # -- rules the subclass owns ------------------------------------------

    @abc.abstractmethod
    def neighbour_pairs(
        self, points: np.ndarray, radius: float, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(sender, receiver)`` node-id pairs for the whole point set, self pairs excluded."""

    @abc.abstractmethod
    def edge_parameters(self) -> tuple[float, int]:
        """``(radius, k)`` to hand :meth:`neighbour_pairs`; the subclass owns the defaults."""

    # -- assembly ---------------------------------------------------------

    def _scale(self, values: np.ndarray) -> np.ndarray:
        """Per-column z-score over every row of ``values``, or ``values`` itself."""
        if not self.normalize:
            return values
        return (values - values.mean(0)) / (values.std(0) + self.edge_dist_eps)

    def build(self, features, locations) -> Data:
        """The graph for ``(features, locations)``: features at locations, one node per row.

        ``sample=None`` takes every row; ``sample=N`` draws ``N`` rows at random (``seed``) and
        keeps them in ascending row order, so edge order stays deterministic. Scaling is fitted on
        all supplied rows before sampling, so a sampled graph lives in the same feature space as the
        full one.
        """
        x = _as_matrix(features, "features")
        locations = _as_matrix(locations, "locations")
        if x.shape[0] != locations.shape[0]:
            raise ValueError(
                f"features has {x.shape[0]} rows but locations has {locations.shape[0]}"
            )

        scaled_x = self._scale(x)
        scaled_locations = self._scale(locations)
        num_rows = x.shape[0]
        if self.sample is not None and self.sample < num_rows:
            keep = np.sort(
                np.random.default_rng(self.seed).choice(num_rows, size=self.sample, replace=False)
            )
        else:
            keep = np.arange(num_rows)
        x_rows = scaled_x[keep]
        loc_rows = scaled_locations[keep]

        senders, receivers = self.neighbour_pairs(loc_rows, *self.edge_parameters())
        senders = np.asarray(senders).reshape(-1).astype(np.int64)
        receivers = np.asarray(receivers).reshape(-1).astype(np.int64)
        if senders.size != receivers.size:
            raise ValueError(
                f"neighbour_pairs returned {senders.size} senders and {receivers.size} receivers"
            )
        # Sorted by (receiver, sender): deterministic edge order, and the order the Nyström
        # kept-edge indexing assumes.
        order = np.lexsort((senders, receivers))
        senders, receivers = senders[order], receivers[order]
        edge_index = torch.from_numpy(np.stack((senders, receivers))).to(torch.long)

        pos = torch.from_numpy(np.ascontiguousarray(loc_rows))
        edge_dist = torch.linalg.vector_norm(pos[senders] - pos[receivers], dim=1)
        data = Data(
            x=torch.from_numpy(np.ascontiguousarray(x_rows)),
            pos=pos,
            edge_index=edge_index,
            edge_attr=_geometry_edge_attr(pos, edge_index, edge_dist),
            edge_dist=edge_dist,
        )
        # The caller's own locations for the same rows: ``Data`` has no slot for them, so they ride
        # along as an attribute, row-aligned with ``pos`` whatever ``normalize`` and ``sample`` say.
        data.pos_raw = torch.from_numpy(np.ascontiguousarray(locations[keep]))
        data.affinity = self.affinities(data)[0]
        return data

    def affinities(self, data: Data) -> tuple[Tensor, Tensor]:
        """Per-edge student-t affinities of ``data.x`` plus the has-a-neighbour mask."""
        return affinities(data.x, data.edge_index, data.num_nodes)
