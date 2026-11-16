"""Landmark (Nyström) sampling and active-graph construction."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .affinity import affinities


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
        """The fraction of the nodes used as landmarks."""
        return self.transition

    def landmark_count(self, num_nodes: int) -> int:
        """Landmarks drawn from ``num_nodes`` nodes: the fraction becomes a count here."""
        return min(int(num_nodes), max(1, round(int(num_nodes) * self.transition)))

    @staticmethod
    def query_count(num_outputs: int) -> int:
        """Queries used per step; every node, chunked by ``chunk_size``."""
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Landmark-sender, active-receiver edges of ``data``: ids plus the edge mask."""
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("the Nyström path needs data.edge_index")
        senders = edge_index[0]
        receivers = edge_index[1]
        # Marking tables beat ``np.isin`` here: the mask runs once per chunk over every edge of the
        # graph, and ``isin`` rebuilds its lookup table from the second argument each time.
        sender_flags = torch.zeros(data.num_nodes, dtype=torch.bool, device=senders.device)
        sender_flags[torch.as_tensor(landmarks, device=senders.device)] = True
        receiver_flags = torch.zeros(data.num_nodes, dtype=torch.bool, device=receivers.device)
        receiver_flags[torch.as_tensor(active, device=receivers.device)] = True
        kept = (sender_flags[senders] & receiver_flags[receivers]).cpu().numpy()
        return senders.cpu().numpy()[kept], receivers.cpu().numpy()[kept], kept

    def active_graph(
        self,
        data: Data,
        queries: np.ndarray,
        landmarks: np.ndarray,
        device: torch.device | None = None,
    ) -> Data:
        """Subgraph of ``queries`` first, then the remaining ``landmarks``.

        Edges are ``data``'s own, filtered to landmark senders and active receivers, so the
        subgraph inherits the neighbourhood of the graph it was handed. Their features are copied
        from ``data.edge_attr`` too, because the edge kernel was trained on them. The target
        affinities are recomputed on the subgraph, from the features of the nodes it holds.
        """
        queries = np.asarray(queries, dtype=np.int64).reshape(-1)
        landmarks = np.asarray(landmarks, dtype=np.int64).reshape(-1)

        active = np.concatenate((queries, landmarks[~np.isin(landmarks, queries)]))
        senders, receivers, kept = self._subgraph_pairs(data, active, landmarks)
        # Senders and receivers are both in ``active``, so one lookup table over ``data.num_nodes``
        # remaps them: indexing per edge in Python would cost a dict hit for every message.
        lookup = np.full(data.num_nodes, -1, dtype=np.int64)
        lookup[active] = np.arange(active.size, dtype=np.int64)
        mapped = np.stack((lookup[senders], lookup[receivers]))

        index = torch.as_tensor(active, dtype=torch.long)
        pos = data.pos[index].to(device).float()
        x = data.x[index].to(device).float()

        edge_index = torch.as_tensor(mapped, dtype=torch.long, device=device)
        kept_index = torch.as_tensor(np.flatnonzero(kept), dtype=torch.long)
        edge_dist = (
            data.edge_dist.to(device)[kept_index]
            if getattr(data, "edge_dist", None) is not None
            else torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        )
        # ``kept_index`` indexes edges, not nodes: ``edge_attr`` has one row per edge. Edge order is
        # unchanged, so the kept rows still line up with ``edge_index``.
        edge_attr = data.edge_attr.to(device).float()[kept_index]

        active_graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        active_graph.edge_dist = edge_dist
        active_graph.affinity = affinities(x, edge_index, active.size)[0]
        active_graph.query_mask = torch.zeros(len(active), dtype=torch.bool, device=device)
        active_graph.query_mask[: len(queries)] = True
        active_graph.global_indices = index
        return active_graph
