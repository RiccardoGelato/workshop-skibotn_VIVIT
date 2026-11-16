"""Point faiss's bundled libomp at torch's copy so one OpenMP runtime loads, not two.

The macOS faiss-cpu wheel vendors ``faiss/.dylibs/libomp.dylib`` and links it through
``@loader_path``; torch ships its own ``torch/lib/libomp.dylib``. Their install names differ, so
dyld loads both and the second initialisation aborts with ``OMP: Error #15``. Replacing the
vendored file with a symlink to torch's copy leaves one runtime, which is exactly how conda-forge
lays it out. Safe to run repeatedly; a no-op off macOS.

**Run this before importing faiss.** The abort happens at import, so this script locates the two
libraries through ``importlib.util.find_spec`` and never imports faiss itself. After it, opt in with
``NOFE_BACKEND=faiss`` (or ``nofe.backend.backend("faiss")``) — ``auto`` stays on cKDTree on macOS
because it cannot know the repair has been applied.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _package_dir(name: str) -> Path | None:
    """Where ``name`` is installed, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(spec.submodule_search_locations[0])


def main() -> int:
    if sys.platform != "darwin":
        return 0

    faiss_dir = _package_dir("faiss")
    if faiss_dir is None:
        print("faiss is not installed: nothing to repair (install it with `uv sync --all-groups`)")
        return 1
    torch_dir = _package_dir("torch")
    if torch_dir is None:
        print("torch is not importable: install the project dependencies")
        return 1

    torch_omp = torch_dir / "lib" / "libomp.dylib"
    bundled = faiss_dir / ".dylibs" / "libomp.dylib"

    if bundled.is_symlink() and bundled.resolve() == torch_omp.resolve():
        print("faiss OpenMP already points at torch's runtime")
        return 0
    if not bundled.exists():
        print(f"{bundled} is missing: the faiss-cpu wheel layout changed, repair it by hand")
        return 1
    if not torch_omp.exists():
        print(f"{torch_omp} is missing: this torch build ships no libomp, repair it by hand")
        return 1

    bundled.unlink()
    bundled.symlink_to(torch_omp)
    print(f"linked {bundled} -> {torch_omp}")
    print("faiss now shares torch's OpenMP runtime; opt in with NOFE_BACKEND=faiss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
