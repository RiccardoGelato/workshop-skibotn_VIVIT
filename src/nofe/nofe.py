"""The ``NOFE`` facade: train, embed, score, save and load."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .affinity import affinity_kl_loss
from .config import resolve_config
from .model import KernelNN
from .nystrom import NystromSampler, node_indices
from .runtime import get_device, set_seed, to_device, validate_graph

CHECKPOINT_FORMAT = "nofe-checkpoint-2"


class NOFE:
    """Graph-kernel dimensionality reduction over a ``torch_geometric`` graph."""

    def __init__(
        self,
        dim: int = 2,
        *,
        config: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self._config = resolve_config(config, overrides, dim)
        self._dim = self._config["dim"]
        self.device = get_device(self._config["device"])
        self._model: KernelNN | None = None
        self.history: list[dict[str, float]] = []
        self.best_val_loss: float | None = None

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """Resolved configuration (defaults, mapping, overrides and dim)."""
        return dict(self._config)

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def model(self) -> KernelNN:
        """The underlying network; raises until it has been built."""
        if self._model is None:
            raise RuntimeError("the model is not built yet; call train(), embed() or build(graph)")
        return self._model

    def __repr__(self) -> str:
        return (
            f"NOFE(dim={self.dim}, width={self._config['width']}, "
            f"depth={self._config['depth']}, device={str(self.device)!r})"
        )

    # -- construction -----------------------------------------------------

    def build(self, data: Data) -> KernelNN:
        """Create (or reuse) the network for the feature width of ``data``."""
        if self._model is not None:
            return self._model
        validate_graph(data)
        in_channels = int(data.x.shape[1])
        edge_channels = int(data.edge_attr.shape[1])
        self._model = KernelNN(
            in_channels,
            edge_channels,
            self.dim,
            width=self._config["width"],
            ker_width=self._config["ker_width"],
            depth=self._config["depth"],
            edge_batch_size=self._config["edge_batch_size"],
        ).to(self.device)
        return self._model

    def _parameters(self):
        return self.model.parameters()

    def _nystrom(self) -> float | None:
        """The landmark fraction fixed at construction, or ``None`` for the dense path."""
        return self._config["nystrom"]

    def _sampler(self) -> NystromSampler:
        fraction = self._nystrom()
        if fraction is None:
            raise RuntimeError(
                "nystrom is not enabled; set config['nystrom'] to a landmark fraction in (0, 1]"
            )
        return NystromSampler(fraction, self._config["chunk_size"])

    def _active(self, data: Data, queries: np.ndarray, landmarks: np.ndarray) -> Data:
        return self._sampler().active_graph(data, queries, landmarks, device=self.device)

    @staticmethod
    def _query_candidates(data: Data) -> np.ndarray:
        """Rows to supervise: ``output_mask`` rows that have a target distribution.

        That is every row when ``y`` supplies truth for all of them — the out-of-sample rows
        included — and otherwise only the rows with features of their own, because a value without
        features has no distribution to match. Landmarks still come from ``input_mask``, so the
        rows being scored never act as senders.
        """
        outputs = node_indices(data, "output_mask")
        if getattr(data, "y", None) is not None:
            return outputs
        candidates = np.intersect1d(outputs, node_indices(data, "input_mask"))
        if candidates.size == 0:
            raise ValueError(
                "the Nyström path needs scored rows with features: supply y for the output_mask "
                "rows, or include rows selected by input_mask in output_mask"
            )
        return candidates

    @staticmethod
    def _truth(data: Data) -> torch.Tensor:
        """Ground truth of every row: ``data.y`` when present, else the input features."""
        target = getattr(data, "y", None)
        return data.x if target is None else target

    @staticmethod
    def _require_truth(data: Data, action: str) -> None:
        """Refuse to score a graph whose scored rows hold a substitute instead of truth."""
        if getattr(data, "y", None) is not None:
            return
        input_mask = getattr(data, "input_mask", None)
        output_mask = getattr(data, "output_mask", None)
        if input_mask is None or output_mask is None:
            return
        observed = torch.as_tensor(input_mask).flatten().to(dtype=torch.bool)
        scored = torch.as_tensor(output_mask).flatten().to(dtype=torch.bool)
        if bool((scored & ~observed).any()):
            raise ValueError(
                f"cannot {action} a graph with no ground-truth features; "
                "build it with query_features=... via your graph builder, or call embed() instead"
            )

    def _graph_loss(self, data: Data, prediction: torch.Tensor, mask: torch.Tensor | None):
        return affinity_kl_loss(
            target=self._truth(data),
            prediction=prediction,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            query_mask=mask,
        )

    def _split(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean train/validation masks over the scored (``output_mask``) rows."""
        num_nodes = data.num_nodes
        candidates = node_indices(data, "output_mask")
        generator = torch.Generator().manual_seed(int(self._config["seed"]))
        order = candidates[torch.randperm(len(candidates), generator=generator).numpy()]
        n_val = max(1, round(len(candidates) * self._config["val_fraction"]))
        if n_val >= len(candidates):
            raise ValueError(
                f"val_fraction={self._config['val_fraction']} with N={num_nodes} leaves no "
                "training nodes; lower val_fraction or use more nodes"
            )
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[torch.as_tensor(order[:n_val])] = True
        train_mask[torch.as_tensor(order[n_val:])] = True
        return train_mask.to(self.device), val_mask.to(self.device)

    def _chunked_loss(self, data: Data, landmarks: np.ndarray, queries: np.ndarray) -> torch.Tensor:
        """Nyström loss over ``queries``: chunk-weighted mean of per-chunk goals."""
        chunk = max(1, min(int(self._config["chunk_size"]), len(queries)))
        total = torch.zeros((), device=self.device)
        weight = 0
        for start in range(0, len(queries), chunk):
            piece = queries[start : start + chunk]
            active = self._active(data, piece, landmarks)
            prediction = self.model(active.x, active.edge_index, active.edge_attr)
            total = total + self._graph_loss(active, prediction, active.query_mask) * len(piece)
            weight += len(piece)
        return total / weight

    # -- training ---------------------------------------------------------

    def train(self, data: Data) -> list[dict[str, float]]:
        """Fit the operator on ``data`` and return the per-epoch history."""
        validate_graph(data)
        self._require_truth(data, "train")
        set_seed(int(self._config["seed"]))
        if self._model is None:
            self.build(data)
        data = to_device(data, self.device)

        optimizer = torch.optim.AdamW(
            self._parameters(),
            lr=self._config["lr"],
            weight_decay=self._config["weight_decay"],
        )
        grad_clip = self._config["grad_clip"]
        early_stopping = bool(self._config["early_stopping"])
        patience = int(self._config["patience"])
        min_delta = float(self._config["min_delta"])

        seed = int(self._config["seed"])
        approximation = self._nystrom() is not None
        total_epochs = int(self._config["epochs"])
        log_every = int(self._config["log_every"])

        if approximation:
            sampler = self._sampler()
            train_rng = torch.Generator().manual_seed(seed)
            val_rng = torch.Generator().manual_seed(seed + 1)
            inputs = node_indices(data, "input_mask")
            outputs = self._query_candidates(data)
            val_queries = outputs[: sampler.query_count(len(outputs))]
            val_landmarks = sampler.sample_nodes(
                inputs, sampler.landmark_count(len(inputs)), val_rng
            )
        else:
            train_mask, val_mask = self._split(data)

        self.history = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(1, total_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            if approximation:
                landmarks = sampler.sample_nodes(
                    inputs, sampler.landmark_count(len(inputs)), train_rng
                )
                loss = self._chunked_loss(data, landmarks, outputs)
            else:
                prediction = self.model(data.x, data.edge_index, data.edge_attr)
                loss = self._graph_loss(data, prediction, train_mask)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._parameters(), grad_clip)
            optimizer.step()
            train_value = float(loss.detach())

            self.model.eval()
            with torch.no_grad():
                if approximation:
                    val_loss = self._chunked_loss(data, val_landmarks, val_queries)
                else:
                    val_prediction = self.model(data.x, data.edge_index, data.edge_attr)
                    val_loss = self._graph_loss(data, val_prediction, val_mask)
                val_value = float(val_loss)

            if not np.isfinite(train_value) or not np.isfinite(val_value):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch} (train={train_value}, val={val_value}); "
                    "lower the learning rate or check the input features"
                )

            if val_value < best_loss - min_delta:
                best_loss = val_value
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_value,
                    "val_loss": val_value,
                    "best_val_loss": best_loss,
                }
            )
            stopping = early_stopping and stale >= patience
            if self._config["verbose"] and (
                epoch % log_every == 0 or epoch == total_epochs or stopping
            ):
                print(
                    f"epoch {epoch:4d}/{total_epochs}  train {train_value:.6f}  val {val_value:.6f}"
                )

            if stopping:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_val_loss = best_loss
        else:
            self.best_val_loss = None
        self.model.eval()
        return self.history

    # -- inference --------------------------------------------------------

    def _forward_chunked(self, data: Data, rows: np.ndarray) -> torch.Tensor:
        """Embed ``rows`` in query chunks, assembling the full graph's predictions.

        Every chunk runs the **full graph** forward pass, because a node's embedding depends on
        messages from all its neighbours, not on which nodes are being reported. Only the returned
        rows are chunked, so the output is bit-identical to one pass while peak resident
        predictions are ``chunk_size x dim``.
        """
        chunk = int(self._config["chunk_size"])
        if chunk >= len(rows):
            return self.model(data.x, data.edge_index, data.edge_attr)[torch.as_tensor(rows)]
        pieces = []
        for start in range(0, len(rows), chunk):
            piece = self.model(data.x, data.edge_index, data.edge_attr)
            pieces.append(piece[torch.as_tensor(rows[start : start + chunk])])
        return torch.cat(pieces)

    def _ensure_built(self, data: Data) -> None:
        if self._model is None:
            self.build(data)

    def embed(self, data: Data) -> torch.Tensor:
        """Embedding for the nodes selected by ``output_mask``, as a CPU tensor."""
        validate_graph(data)
        self._ensure_built(data)
        self.model.eval()
        data = to_device(data, self.device)
        queries = node_indices(data, "output_mask")

        if self._nystrom() is None:
            with torch.no_grad():
                return self._forward_chunked(data, queries).detach().cpu()

        sampler = self._sampler()
        inputs = node_indices(data, "input_mask")
        landmarks = sampler.sample_nodes(
            inputs,
            sampler.landmark_count(len(inputs)),
            torch.Generator().manual_seed(int(self._config["seed"])),
        )
        chunk_size = max(1, min(int(self._config["chunk_size"]), len(queries)))
        pieces: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(queries), chunk_size):
                chunk = queries[start : start + chunk_size]
                active = self._active(data, chunk, landmarks)
                prediction = self.model(active.x, active.edge_index, active.edge_attr)
                pieces.append(prediction[: len(chunk)].detach().cpu())
        self._empty_cache()
        if not pieces:
            return torch.empty((0, self.dim), dtype=torch.float32)
        return torch.cat(pieces)

    def loss(self, data: Data) -> float:
        """Mean KL divergence of the trained operator on ``data``."""
        validate_graph(data)
        self._require_truth(data, "score")
        self._ensure_built(data)
        self.model.eval()
        data = to_device(data, self.device)

        with torch.no_grad():
            if self._nystrom() is None:
                prediction = self.model(data.x, data.edge_index, data.edge_attr)
                mask = getattr(data, "output_mask", None)
                if mask is not None:
                    mask = mask.to(dtype=torch.bool)
                return float(self._graph_loss(data, prediction, mask))

            sampler = self._sampler()
            inputs = node_indices(data, "input_mask")
            outputs = self._query_candidates(data)
            rng = torch.Generator().manual_seed(int(self._config["seed"]))
            landmarks = sampler.sample_nodes(inputs, sampler.landmark_count(len(inputs)), rng)
            queries = outputs[: sampler.query_count(len(outputs))]
            return float(self._chunked_loss(data, landmarks, queries))

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write a self-describing checkpoint and return its path."""
        target = Path(path)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "dim": self.dim,
            "config": self.config,
            "model_config": self.model.config,
            "state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> NOFE:
        """Rebuild a trained model from a checkpoint written by :meth:`save`."""
        target = Path(path)
        payload = torch.load(target, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and payload.get("format") == "nofe-checkpoint-1":
            raise ValueError(
                f"{target} is a nofe-checkpoint-1 file, which had a 'method' argument; "
                "retrain or convert it"
            )
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            found = payload.get("format") if isinstance(payload, dict) else None
            raise ValueError(f"{target} is not a NOFE checkpoint (format={found!r})")

        config = dict(payload["config"])
        dim = config.pop("dim")
        if device is not None:
            config["device"] = device
        model = cls(dim, config=config)
        model._model = KernelNN(**payload["model_config"]).to(model.device)
        model._model.load_state_dict(payload["state_dict"])
        model._model.eval()
        model.best_val_loss = payload.get("best_val_loss")
        return model

    @staticmethod
    def _empty_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
