"""The graph-kernel operator and the trainer that drives it."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv

from .affinity import affinity_kl_loss
from .nystrom import NystromSampler
from .runtime import get_device, set_seed, to_device


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


def _edge_kernel(ker_width: int, width: int, *, edge_channels: int | None = None) -> nn.Module:
    """MLP mapping edge features to a flattened ``width x width`` kernel.

    ``edge_channels=None`` makes the first layer lazy: it takes the edge width from the first
    ``edge_attr`` it sees, inside ``NNConv``'s own call.
    """
    first = (
        nn.LazyLinear(ker_width) if edge_channels is None else nn.Linear(edge_channels, ker_width)
    )
    return nn.Sequential(
        first,
        nn.ReLU(),
        nn.Linear(ker_width, ker_width),
        nn.ReLU(),
        nn.Linear(ker_width, width * width),
    )


def _resolve(layer: nn.Module) -> int:
    """Input width of ``layer``, or ``RuntimeError`` while it is still uninitialised."""
    if isinstance(layer.weight, nn.parameter.UninitializedParameter):
        raise RuntimeError(
            "call forward() once (or pass in_channels/edge_channels) before inspecting widths"
        )
    return int(layer.in_features)


class KernelNN(nn.Module):
    """Graph-kernel operator trained to reproduce affinity neighbourhoods.

    ``in_channels`` (node features) and ``edge_channels`` (edge features) may be omitted; they are
    then discovered on the first forward pass and readable afterwards through the properties of the
    same names.
    """

    def __init__(
        self,
        out_dim: int = 3,
        *,
        in_channels: int | None = None,
        edge_channels: int | None = None,
        width: int = 32,
        ker_width: int = 64,
        depth: int = 3,
        edge_batch_size: int | None = 65_536,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.lift = nn.LazyLinear(width) if in_channels is None else nn.Linear(in_channels, width)
        self.conv = EdgeConv(
            width,
            _edge_kernel(ker_width, width, edge_channels=edge_channels),
            edge_batch_size=edge_batch_size,
        )
        self.project = nn.Linear(width, out_dim)

    @property
    def in_channels(self) -> int:
        """Node-feature width; raises until the first forward pass resolves it."""
        return _resolve(self.lift)

    @property
    def edge_channels(self) -> int:
        """Edge-feature width; raises until the first forward pass resolves it."""
        return _resolve(self.conv.nn[0])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.lift(x)
        for _ in range(self.depth):
            h = self.conv(h, edge_index, edge_attr).relu()
        return self.project(h)


def _as_graphs(data: Data | Iterable[Data]) -> list[Data]:
    """Normalise a single graph or an iterable of them to a non-empty list.

    ``Data`` is checked first on purpose: it implements ``__iter__`` (yielding ``(key, value)``
    pairs), so an ``isinstance(..., Iterable)`` test would shred a single graph into its attribute
    tuples.
    """
    if isinstance(data, Data):
        graphs = [data]
    elif isinstance(data, (str, bytes)) or not isinstance(data, Iterable):
        raise TypeError(
            f"expected torch_geometric.data.Data or an iterable of them, got {type(data).__name__}"
        )
    else:
        graphs = list(data)
    if not graphs:
        raise ValueError("at least one graph is required")

    first = graphs[0]
    for index, graph in enumerate(graphs[1:], start=1):
        if graph.x.shape[1] != first.x.shape[1] or (
            graph.edge_attr.shape[1] != first.edge_attr.shape[1]
        ):
            raise ValueError(
                f"graph {index} has widths x={graph.x.shape[1]}, "
                f"edge_attr={graph.edge_attr.shape[1]} but graph 0 has "
                f"x={first.x.shape[1]}, edge_attr={first.edge_attr.shape[1]}; every graph in one "
                "call must share both widths"
            )
    return graphs


class _PreparedGraph(NamedTuple):
    """One training graph plus its validation split or Nyström sampling state."""

    data: Data
    train_mask: Tensor | None = None
    val_mask: Tensor | None = None
    inputs: np.ndarray | None = None
    outputs: np.ndarray | None = None
    val_queries: np.ndarray | None = None
    val_landmarks: np.ndarray | None = None
    sampler: NystromSampler | None = None
    train_rng: torch.Generator | None = None


class NOFE:
    """Train and apply a graph-kernel embedding: ``.net`` is the torch module, this is the driver.

    ``NOFE`` is not an ``nn.Module`` on purpose: ``nn.Module`` owns the attribute ``training`` (a
    bool), so a trainer method with that name cannot live on one.
    """

    def __init__(
        self,
        out_dim: int = 3,
        *,
        in_channels: int | None = None,
        edge_channels: int | None = None,
        width: int = 32,
        ker_width: int = 64,
        depth: int = 3,
        edge_batch_size: int | None = 65_536,
        nystrom: float | None = None,
        chunk_size: int = 8_192,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if nystrom is not None:
            if (
                not isinstance(nystrom, (int, float))
                or isinstance(nystrom, bool)
                or not 0 < nystrom <= 1
            ):
                raise ValueError(f"nystrom must be None or a fraction in (0, 1], got {nystrom!r}")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        self.device = get_device(device)
        self.nystrom = None if nystrom is None else float(nystrom)
        self.chunk_size = int(chunk_size)
        # The weights are drawn here, so the seed has to be applied here: a seed set only inside
        # ``training`` would come after construction and leave the initialisation up to whatever
        # global RNG state the caller happened to be in.
        set_seed(seed)
        self.net = KernelNN(
            out_dim,
            in_channels=in_channels,
            edge_channels=edge_channels,
            width=width,
            ker_width=ker_width,
            depth=depth,
            edge_batch_size=edge_batch_size,
        ).to(self.device)

    # -- helpers ----------------------------------------------------------

    def _sampler(self) -> NystromSampler:
        if self.nystrom is None:
            raise RuntimeError("nystrom is not enabled; pass nystrom=<fraction in (0, 1]>")
        return NystromSampler(self.nystrom, self.chunk_size)

    def _active(self, data: Data, queries: np.ndarray, landmarks: np.ndarray) -> Data:
        return self._sampler().active_graph(data, queries, landmarks, device=self.device)

    def _graph_loss(self, data: Data, prediction: Tensor) -> Tensor:
        """The graph's own mean KL: its stored affinities are the target."""
        return affinity_kl_loss(
            None,
            prediction,
            data.edge_index,
            num_nodes=data.num_nodes,
            weights=data.affinity,
        )

    def _split(self, data: Data, *, seed: int, val_fraction: float) -> tuple[Tensor, Tensor]:
        """Boolean train/validation masks over every node."""
        num_nodes = data.num_nodes
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(num_nodes, generator=generator).numpy()
        n_val = max(1, round(num_nodes * val_fraction))
        if n_val >= num_nodes:
            raise ValueError(
                f"val_fraction={val_fraction} with N={num_nodes} leaves no training nodes; "
                "lower val_fraction or use more nodes"
            )
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[torch.as_tensor(order[:n_val])] = True
        train_mask[torch.as_tensor(order[n_val:])] = True
        return train_mask.to(self.device), val_mask.to(self.device)

    def _chunked_loss(self, data: Data, landmarks: np.ndarray, queries: np.ndarray) -> Tensor:
        """Nyström loss over ``queries``: chunk-weighted mean of per-chunk goals."""
        chunk = max(1, min(self.chunk_size, len(queries)))
        total = torch.zeros((), device=self.device)
        weight = 0
        for start in range(0, len(queries), chunk):
            piece = queries[start : start + chunk]
            active = self._active(data, piece, landmarks)
            prediction = self.net(active.x, active.edge_index, active.edge_attr)
            total = total + self._graph_loss(active, prediction) * len(piece)
            weight += len(piece)
        return total / weight

    def _prepare(self, graphs: list[Data], seed: int, val_fraction: float) -> list[_PreparedGraph]:
        """Per-graph validation split or Nyström state, built once with a per-graph seed."""
        approximation = self.nystrom is not None
        prepared: list[_PreparedGraph] = []
        for index, graph in enumerate(graphs):
            graph = to_device(graph, self.device)
            if not approximation:
                train_mask, val_mask = self._split(
                    graph, seed=seed + index, val_fraction=val_fraction
                )
                prepared.append(_PreparedGraph(graph, train_mask, val_mask))
                continue
            sampler = self._sampler()
            every_node = np.arange(graph.num_nodes)
            prepared.append(
                _PreparedGraph(
                    graph,
                    inputs=every_node,
                    outputs=every_node,
                    val_queries=every_node[: sampler.query_count(graph.num_nodes)],
                    val_landmarks=sampler.sample_nodes(
                        every_node,
                        sampler.landmark_count(graph.num_nodes),
                        torch.Generator().manual_seed(seed + len(graphs) + index),
                    ),
                    sampler=sampler,
                    train_rng=torch.Generator().manual_seed(seed + index),
                )
            )
        return prepared

    def _train_step(self, prepared: _PreparedGraph) -> Tensor:
        data = prepared.data
        if prepared.sampler is None:
            prediction = self.net(data.x, data.edge_index, data.edge_attr)
            return self._masked_loss(data, prediction, prepared.train_mask)
        landmarks = prepared.sampler.sample_nodes(
            prepared.inputs,
            prepared.sampler.landmark_count(len(prepared.inputs)),
            prepared.train_rng,
        )
        return self._chunked_loss(data, landmarks, prepared.outputs)

    def _masked_loss(self, data: Data, prediction: Tensor, mask: Tensor | None) -> Tensor:
        return affinity_kl_loss(
            None,
            prediction,
            data.edge_index,
            num_nodes=data.num_nodes,
            query_mask=mask,
            weights=data.affinity,
        )

    def _validation_loss(self, prepared: _PreparedGraph) -> Tensor:
        data = prepared.data
        if prepared.sampler is None:
            prediction = self.net(data.x, data.edge_index, data.edge_attr)
            return self._masked_loss(data, prediction, prepared.val_mask)
        return self._chunked_loss(data, prepared.val_landmarks, prepared.val_queries)

    # -- training ---------------------------------------------------------

    def training(
        self,
        graphs,
        *,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        grad_clip: float | None = 1.0,
        patience: int = 20,
        min_delta: float = 1e-6,
        val_fraction: float = 0.1,
        early_stopping: bool = True,
        verbose: bool = False,
        log_every: int = 1_000,
        seed: int = 42,
    ) -> list[dict[str, float]] | list[list[dict[str, float]]]:
        """Fit the operator and return the per-epoch history.

        One graph returns one history; an iterable of graphs returns one history per graph. With
        ``N`` graphs an epoch is ``N`` optimizer steps, one per graph — early stopping and the
        restored best state are decided on the mean validation loss over the graphs.
        """
        single = isinstance(graphs, Data)
        data = _as_graphs(graphs)
        set_seed(seed)
        prepared = self._prepare(data, seed, float(val_fraction))

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        histories: list[list[dict[str, float]]] = [[] for _ in prepared]
        per_graph_best = [float("inf")] * len(prepared)
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale = 0

        for epoch in range(1, int(epochs) + 1):
            train_values = []
            for index, prepared_graph in enumerate(prepared):
                self.net.train()
                optimizer.zero_grad()
                loss = self._train_step(prepared_graph)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                optimizer.step()
                value = float(loss.detach())
                train_values.append(value)
                histories[index].append({"epoch": epoch, "train_loss": value})

            self.net.eval()
            with torch.no_grad():
                val_values = [float(self._validation_loss(p)) for p in prepared]

            for index, val_value in enumerate(val_values):
                per_graph_best[index] = min(per_graph_best[index], val_value)
                histories[index][-1]["val_loss"] = val_value
                histories[index][-1]["best_val_loss"] = per_graph_best[index]

            train_value = float(np.mean(train_values))
            val_value = float(np.mean(val_values))
            if not np.isfinite(train_value) or not np.isfinite(val_value):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch} (train={train_value}, val={val_value}); "
                    "lower the learning rate or check the input features"
                )

            if val_value < best_loss - min_delta:
                best_loss = val_value
                best_state = copy.deepcopy(self.net.state_dict())
                stale = 0
            else:
                stale += 1

            stopping = early_stopping and stale >= patience
            if verbose and (epoch % log_every == 0 or epoch == int(epochs) or stopping):
                print(
                    f"epoch {epoch:4d}/{int(epochs)}  train {train_value:.6f}  val {val_value:.6f}"
                )
            if stopping:
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return histories[0] if single else histories

    # -- inference --------------------------------------------------------

    def _embed_one(self, data: Data) -> Tensor:
        data = to_device(data, self.device)
        with torch.no_grad():
            if self.nystrom is None:
                return self.net(data.x, data.edge_index, data.edge_attr).detach().cpu()

            sampler = self._sampler()
            every_node = np.arange(data.num_nodes)
            landmarks = sampler.sample_nodes(
                every_node,
                sampler.landmark_count(data.num_nodes),
                torch.Generator().manual_seed(0),
            )
            chunk = max(1, min(self.chunk_size, data.num_nodes))
            pieces: list[Tensor] = []
            for start in range(0, data.num_nodes, chunk):
                piece = every_node[start : start + chunk]
                active = self._active(data, piece, landmarks)
                prediction = self.net(active.x, active.edge_index, active.edge_attr)
                # Queries come first in the active subgraph, so the reported rows are its prefix.
                pieces.append(prediction[: len(piece)].detach().cpu())
        if not pieces:
            return torch.empty((0, self.net.project.out_features), dtype=torch.float32)
        return torch.cat(pieces)

    def embed(self, graphs) -> Tensor | list[Tensor]:
        """Embedding of every node, in node order: one CPU tensor, or one per graph."""
        single = isinstance(graphs, Data)
        data = _as_graphs(graphs)
        self.net.eval()
        embeddings = [self._embed_one(graph) for graph in data]
        return embeddings[0] if single else embeddings

    def fit(self, graph: Data, **training_kwargs) -> Tensor:
        """Train on one graph, then return its embedding."""
        self.training(graph, **training_kwargs)
        return self.embed(graph)

    def loss(self, graphs) -> float | list[float]:
        """Mean KL of the trained operator: a float for one graph, one per graph for a dataset.

        Uses the same forward path training did, so on the Nyström approximation the number
        describes the operator that was actually fit. The landmark draw is seeded with 0 for every
        graph on purpose: ``loss`` is a per-graph score, so scoring ``[a, b]`` has to agree with
        scoring each alone.
        """
        single = isinstance(graphs, Data)
        data = _as_graphs(graphs)
        self.net.eval()
        values = []
        with torch.no_grad():
            for graph in data:
                graph = to_device(graph, self.device)
                if self.nystrom is None:
                    prediction = self.net(graph.x, graph.edge_index, graph.edge_attr)
                    values.append(float(self._graph_loss(graph, prediction)))
                    continue
                every_node = np.arange(graph.num_nodes)
                landmarks = self._sampler().sample_nodes(
                    every_node,
                    self._sampler().landmark_count(graph.num_nodes),
                    torch.Generator().manual_seed(0),
                )
                values.append(float(self._chunked_loss(graph, landmarks, every_node)))
        return values[0] if single else values
