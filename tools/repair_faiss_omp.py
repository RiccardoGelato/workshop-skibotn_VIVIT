"""Point faiss's bundled libomp at torch's copy so one OpenMP runtime loads, not two.

The macOS faiss-cpu wheel vendors ``faiss/.dylibs/libomp.dylib`` and links it through
``@loader_path``; torch ships its own ``torch/lib/libomp.dylib``. Their install names differ, so
dyld loads both and the second initialisation aborts with ``OMP: Error #15``. Replacing the
vendored file with a symlink to torch's copy leaves one runtime, which is exactly how conda-forge
lays it out. Safe to run repeatedly; a no-op off macOS.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin":
        return 0

    try:
        import faiss
    except ImportError:
        print("faiss is not importable: install the examples extra or the dev group")
        return 1
    try:
        import torch
    except ImportError:
        print("torch is not importable: install the project dependencies")
        return 1

    torch_omp = Path(torch.__file__).parent / "lib" / "libomp.dylib"
    bundled = Path(faiss.__file__).parent / ".dylibs" / "libomp.dylib"

    if bundled.is_symlink() and bundled.resolve() == torch_omp.resolve():
        print("faiss OpenMP already points at torch's runtime")
        return 0
    if not bundled.exists():
        print(f"{bundled} is missing: the faiss-cpu wheel layout changed, repair it by hand")
        return 1

    bundled.unlink()
    bundled.symlink_to(torch_omp)
    print(f"linked {bundled} -> {torch_omp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
