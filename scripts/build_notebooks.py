"""Build the methodology notebooks from their percent-format sources.

Sources live in ``notebooks/methodology/src/*.py`` (diff-friendly), notebooks
are written next to ``src/``. Usage::

    python scripts/build_notebooks.py            # every source
    python scripts/build_notebooks.py 00 02      # sources whose name starts with these

'# %% [markdown]' starts a markdown cell (lines stripped of a leading '# ');
'# %% [parameters]' starts the papermill parameters cell; '# %%' a code cell.
"""

import json
import sys
from pathlib import Path


def convert(src: Path, dst: Path) -> None:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        source = [l + "\n" for l in lines]
        if source:
            source[-1] = source[-1].rstrip("\n")
        cell = {"cell_type": "markdown" if kind == "markdown" else "code", "metadata": {}, "source": source}
        if kind != "markdown":
            cell["execution_count"] = None
            cell["outputs"] = []
        if kind == "parameters":
            cell["metadata"]["tags"] = ["parameters"]
        cells.append(cell)

    for line in src.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            buf = []
            tag = line[4:].strip()
            kind = "markdown" if tag == "[markdown]" else "parameters" if tag == "[parameters]" else "code"
            continue
        buf.append(line)
    flush()
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    dst.write_text(json.dumps(nb, indent=1) + "\n")


SRC = Path(__file__).resolve().parent.parent / "notebooks" / "methodology" / "src"


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for src in sorted(SRC.glob("*.py")):
        if prefixes and not any(src.name.startswith(p) for p in prefixes):
            continue
        dst = src.parent.parent / (src.stem + ".ipynb")
        convert(src, dst)
        print("wrote", dst)
