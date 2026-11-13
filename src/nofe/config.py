"""Configuration defaults and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULTS: dict[str, Any] = {
    # model
    "width": 32,
    "ker_width": 64,
    "depth": 3,
    "edge_batch_size": 65_536,
    # training
    "epochs": 200,
    "lr": 1e-2,
    "weight_decay": 5e-4,
    "grad_clip": 1.0,
    "patience": 20,
    "min_delta": 1e-6,
    "val_fraction": 0.1,
    "early_stopping": True,
    "seed": 42,
    "device": "auto",
    "verbose": False,
    "log_every": 1_000,
    # nystrom
    "nystrom": None,
    "chunk_size": 8192,
}

_RESERVED = ("dim",)
_DEVICES = ("auto", "cpu", "cuda", "mps")

_POSITIVE_INTS = (
    "width",
    "ker_width",
    "depth",
    "epochs",
    "log_every",
    "chunk_size",
)
_POSITIVE_FLOATS = ("lr", "weight_decay", "min_delta")


def _validate(key: str, value: Any) -> None:
    if key in _POSITIVE_INTS:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
    elif key in _POSITIVE_FLOATS:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive float, got {value!r}")
    elif key == "val_fraction":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {value!r}")
    elif key == "nystrom":
        if value is None:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
            raise ValueError(f"nystrom must be None or a fraction in (0, 1], got {value!r}")
    elif key == "patience":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"patience must be an integer >= 1, got {value!r}")
    elif key == "grad_clip":
        if value is None:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"grad_clip must be a positive float or None, got {value!r}")
    elif key == "edge_batch_size":
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"edge_batch_size must be a positive integer or None, got {value!r}")
    elif key == "device":
        if value not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {value!r}")


def resolve_config(
    config: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
    dim: int,
) -> dict[str, Any]:
    """Merge ``DEFAULTS`` < ``config`` < ``overrides`` and validate the result."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")

    resolved = dict(DEFAULTS)
    for source in (config or {}, overrides):
        for key, value in source.items():
            if key in _RESERVED:
                raise ValueError(f"{key!r} is a constructor argument, not a config key")
            if key not in DEFAULTS:
                raise ValueError(f"unknown config key {key!r}; valid keys: {sorted(DEFAULTS)}")
            _validate(key, value)
            resolved[key] = value

    if resolved["nystrom"] is not None:
        resolved["nystrom"] = float(resolved["nystrom"])
    resolved["dim"] = dim
    return resolved
