"""Landmark (Nyström) sampling and active-graph construction."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .config import DEFAULTS
from .graphs import geometry_edge_attr, radius_graph


def node_indices(data: Data, name: str) -> np.ndarray:
    """Node ids selected by a mask attribute, or every node when absent."""
    mask = getattr(data, name, None)
    if mask is None:
        return np.arange(data.num_nodes)
    flags = torch.as_tensor(mask).flatten().to(device="cpu", dtype=torch.bool)
    return flags.nonzero().numpy().reshape(-1)


class NystromSampler:
    """Builds the landmark-plus-query subgraphs the Nyström path trains on."""

    def __init__(self, transition: float, chunk_size: int) -> None:
        if (
            not isinstance(transition, (int, float))
            or isinstance(transition, bool)
            or not 0 < transition <= 1
        ):
            raise ValueError("landmark fraction must be in (0, 1]")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.transition = float(transition)
        self.chunk_size = int(chunk_size)

    @property
    def landmark_fraction(self) -> float:
        """The fraction of the input rows used as landmarks."""
        return self.transition

    def landmark_count(self, num_inputs: int) -> int:
        """Landmarks drawn from ``num_inputs`` rows: the fraction becomes a count here."""
        return min(int(num_inputs), max(1, round(int(num_inputs) * self.transition)))

    @staticmethod
    def query_count(num_outputs: int) -> int:
        """Queries used per step; every output row, chunked by ``chunk_size``."""
        return int(num_outputs)

    @staticmethod
    def sample_nodes(candidates: np.ndarray, count: int, generator: torch.Generator) -> np.ndarray:
        """Draw at most ``count`` distinct node ids from ``candidates``."""
        candidates = np.asarray(candidates)
        if candidates.size == 0:
            raise ValueError("cannot sample from an empty node set")
        count = min(int(count), len(candidates))
        order = torch.randperm(len(candidates), generator=generator)[:count].numpy()
        return candidates[order]

    @staticmethod
    def _subgraph_pairs(
        data: Data, active: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Landmark-sender, active-receiver pairs taken from ``data``'s own neighbourhood."""
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            radius = float(DEFAULTS["radius"])
            pairs, _ = radius_graph(data.pos, radius, sources=landmarks, targets=active)
            return pairs[0].numpy(), pairs[1].numpy()
        senders = edge_index[0].detach().cpu().numpy()
        receivers = edge_index[1].detach().cpu().numpy()
        keep = np.isin(senders, landmarks) & np.isin(receivers, active)
        return senders[keep], receivers[keep]

    def active_graph(
        self,
        data: Data,
        queries: np.ndarray,
        landmarks: np.ndarray,
        device: torch.device | None = None,
    ) -> Data:
        """Subgraph of ``queries`` first, then the remaining ``landmarks``.

        Edges come from ``data`` when it carries them, filtered to landmark senders and active
        receivers, so the subgraph inherits whatever neighbourhood ``build_graph`` chose. A
        ``Data`` without edges falls back to ``radius_graph``.
        """
        queries = np.asarray(queries, dtype=np.int64).reshape(-1)
        landmarks = np.asarray(landmarks, dtype=np.int64).reshape(-1)

        if not np.isin(queries, node_indices(data, "output_mask")).all():
            raise ValueError("queries must be selected by output_mask")
        if not np.isin(landmarks, node_indices(data, "input_mask")).all():
            raise ValueError("landmarks must be selected by input_mask")

        active = np.concatenate((queries, landmarks[~np.isin(landmarks, queries)]))
        lookup = {int(node): position for position, node in enumerate(active)}
        senders, receivers = self._subgraph_pairs(data, active, landmarks)
        mapped = np.asarray(
            (
                [lookup[int(node)] for node in senders],
                [lookup[int(node)] for node in receivers],
            ),
            dtype=np.int64,
        ).reshape(2, -1)

        index = torch.as_tensor(active, dtype=torch.long)
        pos = data.pos[index].to(device).float()
        # Features are copied unchanged: added rows already hold the substitute the dense path
        # sees, so zeroing them here would make the two paths disagree about the model input.
        x = data.x[index].to(device).float()

        edge_index = torch.as_tensor(mapped, dtype=torch.long, device=device)
        edge_dist = torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = geometry_edge_attr(pos, edge_index, edge_dist)

        active_graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        active_graph.edge_dist = edge_dist
        active_graph.query_mask = torch.zeros(len(active), dtype=torch.bool, device=device)
        active_graph.query_mask[: len(queries)] = True
        active_graph.global_indices = index
        output_mask = getattr(data, "output_mask", None)
        if output_mask is not None:
            active_graph.output_mask = (
                torch.as_tensor(output_mask)
                .flatten()
                .to(device="cpu", dtype=torch.bool)[index]
                .to(device)
            )
        return active_graph
