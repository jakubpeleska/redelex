r"""Execute the methodology notebooks headless and record which gates passed.

Each notebook ends in a GATE cell of assertions, so a notebook that runs to the
end has passed its gate. Executed copies are written to
``notebooks/methodology/executed/<name>__<tag>.ipynb``, and each run's outcome to
its own ``executed/status/<name>__<tag>.json``, so parallel jobs never write the
same file. ``--summary`` prints every recorded outcome.

Usage::

    .venv/bin/python scripts/run_methodology_notebooks.py 00 01 02 --tag cpu-smoke -p SCALE smoke
    .venv/bin/python scripts/run_methodology_notebooks.py 03 --tag a100-full -p SCALE full -p DEVICE cuda
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "methodology"
EXEC_DIR = NB_DIR / "executed"


def _value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefixes", nargs="*", help="notebook name prefixes, e.g. 00 01 (default: all)")
    ap.add_argument("--tag", required=True, help="label of this run, e.g. cpu-smoke or a100-full")
    ap.add_argument("-p", "--param", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"))
    ap.add_argument("--timeout", type=int, default=None, help="per-cell timeout in seconds")
    ap.add_argument("--summary", action="store_true", help="print recorded outcomes and exit")
    args = ap.parse_args(argv)

    status_dir = EXEC_DIR / "status"
    if args.summary:
        for path in sorted(status_dir.glob("*.json")):
            st = json.loads(path.read_text())
            print(f"{'PASS' if st['gate_passed'] else 'FAIL'}  {path.stem}  {st['seconds']} s  {st['finished']}"
                  + (f"  {st['error']}" if st["error"] else ""))
        return 0

    import papermill as pm

    params = {k: _value(v) for k, v in args.param}
    notebooks = sorted(p for p in NB_DIR.glob("*.ipynb"))
    if args.prefixes:
        notebooks = [p for p in notebooks if any(p.name.startswith(x) for x in args.prefixes)]
    if not notebooks:
        print("no notebooks matched", args.prefixes, file=sys.stderr)
        return 2

    status_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for nb in notebooks:
        target = EXEC_DIR / f"{nb.stem}__{args.tag}.ipynb"
        start = time.time()
        error = None
        try:
            pm.execute_notebook(str(nb), str(target), parameters=params, cwd=str(ROOT),
                                execution_timeout=args.timeout, progress_bar=False)
        except Exception as exc:  # the executed copy keeps the traceback
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:500]
            failed += 1
        seconds = round(time.time() - start, 1)
        record = {
            "notebook": nb.name,
            "tag": args.tag,
            "parameters": params,
            "gate_passed": error is None,
            "error": error,
            "seconds": seconds,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (status_dir / f"{nb.stem}__{args.tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"{'PASS' if error is None else 'FAIL'}  {nb.name}  ({seconds} s)  -> {target.relative_to(ROOT)}")
        if error:
            print("      ", error)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
