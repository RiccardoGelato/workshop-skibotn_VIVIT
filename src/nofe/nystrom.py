"""Landmark (Nyström) sampling and active-graph construction."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


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
        from ``data.edge_attr`` too, because the edge kernel was trained on them.
        """
        queries = np.asarray(queries, dtype=np.int64).reshape(-1)
        landmarks = np.asarray(landmarks, dtype=np.int64).reshape(-1)

        every_output = node_indices(data, "output_mask")
        every_input = node_indices(data, "input_mask")
        if not np.isin(queries, every_output).all():
            raise ValueError("queries must be selected by output_mask")
        if not np.isin(landmarks, every_input).all():
            raise ValueError("landmarks must be selected by input_mask")

        active = np.concatenate((queries, landmarks[~np.isin(landmarks, queries)]))
        senders, receivers, kept = self._subgraph_pairs(data, active, landmarks)
        # Senders and receivers are both in ``active``, so one lookup table over ``data.num_nodes``
        # remaps them: indexing per edge in Python would cost a dict hit for every message.
        lookup = np.full(data.num_nodes, -1, dtype=np.int64)
        lookup[active] = np.arange(active.size, dtype=np.int64)
        mapped = np.stack((lookup[senders], lookup[receivers]))

        index = torch.as_tensor(active, dtype=torch.long)
        pos = data.pos[index].to(device).float()
        # Features are copied unchanged: added rows already hold the substitute the dense path
        # sees, so zeroing them here would make the two paths disagree about the model input.
        x = data.x[index].to(device).float()
        target = getattr(data, "y", None)

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
        # ``y`` travels with ``x``: a withheld query row has no features of its own, so its only
        # target distribution is the truth the caller put in ``y``. Without this the Nyström
        # objective falls back to ``x`` and silently skips every out-of-sample row.
        if target is not None:
            active_graph.y = target[index].to(device).float()
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
