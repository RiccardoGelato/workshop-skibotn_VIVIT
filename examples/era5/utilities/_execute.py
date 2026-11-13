"""Execute every notebook in place and store its outputs.

Run from any directory (paths are resolved from this file):

    uv run python examples/era5/utilities/_execute.py            # all notebooks
    uv run python examples/era5/utilities/_execute.py 1-in-sample.ipynb

The kernels start in ``examples/era5/notebooks/``, not here: the cells resolve their data from the
repo layout, and that is where the notebooks themselves live.

A notebook that raises fails with the traceback, so this doubles as the notebooks' test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
TIMEOUT = 1_800


def execute(path: Path) -> None:
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(notebook, path)
    print(f"executed {path.name}")


if __name__ == "__main__":
    targets = [NOTEBOOKS / name for name in sys.argv[1:]] or sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))
    for target in targets:
        execute(target)
