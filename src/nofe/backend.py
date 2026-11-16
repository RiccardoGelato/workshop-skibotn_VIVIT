"""The neighbourhood backend: faiss where it works, ``scipy.spatial.cKDTree`` where it does not.

faiss is the faster index and the one the non-macOS platforms use, but the macOS arm64 wheel vendors
its own ``libomp.dylib`` next to torch's and dyld aborts on the second initialisation with
``OMP: Error #15`` — a crash at import time, before any fallback could run. So the choice is made
from the platform *before* importing either, which is what makes a macOS checkout work out of the
box: no faiss import, no abort, no repair step.

``tools/repair_faiss_omp.py`` links the two OpenMP runtimes together, after which macOS can use
faiss too; :func:`backend` and ``NOFE_BACKEND=faiss`` honour that. Both backends return **exactly
the same edge pairs** for the same input — see ``tests/test_graph_utils.py``, which cross-checks
them.

``NOFE_BACKEND`` (``auto``, ``faiss`` or ``ckdtree``) overrides the platform default.
"""

from __future__ import annotations

import importlib.util
import os
import sys

BACKENDS = ("auto", "faiss", "ckdtree")
ENV_VAR = "NOFE_BACKEND"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _faiss_available() -> bool:
    """Whether faiss is importable without importing it: the import can abort on macOS."""
    try:
        return importlib.util.find_spec("faiss") is not None
    except (ImportError, ValueError):
        return False


def requested_backend(spec: str | None = None) -> str:
    """``spec`` when given, else ``NOFE_BACKEND``, else ``"auto"``; validated."""
    value = spec if spec is not None else os.environ.get(ENV_VAR, "auto")
    value = value.strip().lower()
    if value not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {value!r}")
    return value


def backend(spec: str | None = None) -> str:
    """The concrete backend name: ``"faiss"`` or ``"ckdtree"``.

    ``"auto"`` picks cKDTree on macOS even when faiss is installed, because importing the macOS
    faiss wheel aborts unless ``tools/repair_faiss_omp.py`` has been run. Select ``faiss`` (or set
    ``NOFE_BACKEND=faiss``) once it has.
    """
    choice = requested_backend(spec)
    if choice == "ckdtree":
        return "ckdtree"
    if choice == "faiss":
        if not _faiss_available():
            raise RuntimeError(
                f"{ENV_VAR}=faiss requested but faiss is not installed; "
                "install it with `uv sync --all-groups`"
            )
        return "faiss"
    # auto: cKDTree on macOS, faiss everywhere else it is installed.
    if _is_macos():
        return "ckdtree"
    return "faiss" if _faiss_available() else "ckdtree"


def describe() -> str:
    """One line naming the backend and why, for the examples' startup print."""
    if backend() == "faiss":
        return "faiss (exact L2 index)"
    if _is_macos() and requested_backend() == "auto":
        return "scipy cKDTree (macOS: faiss's vendored libomp clashes with torch's; see README)"
    return "scipy cKDTree"


__all__ = ["BACKENDS", "ENV_VAR", "backend", "describe", "requested_backend"]
