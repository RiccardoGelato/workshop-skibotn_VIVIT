"""Execute every notebook in place and store its outputs.

Run from ``notebooks/`` (or any directory; paths are resolved from this file):

    uv run python notebooks/_execute.py            # all notebooks
    uv run python notebooks/_execute.py 1-in-sample.ipynb

A notebook that raises fails with the traceback, so this doubles as the notebooks' test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
TIMEOUT = 1_800


def execute(path: Path) -> None:
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(notebook, path)
    print(f"executed {path.name}")


if __name__ == "__main__":
    targets = [HERE / name for name in sys.argv[1:]] or sorted(HERE.glob("[0-9]*.ipynb"))
    for target in targets:
        execute(target)
