r"""Plan and submit rerun chains on RCI as lanes of short, resuming jobs.

Every chain (dataset, task, regime, seed) is assigned to one of ``--lanes`` lanes,
longest first onto the least-loaded lane, with per-chain hours taken from the
budget (notebook 08) when given. A lane is a list file plus a dependency chain of
jobs (``afterany``): each job runs ``slurm/rci/run_chain2.sh`` over the lane's list,
skips finished chains in seconds, resumes the unfinished one, and stops before
its deadline. Slurm's account cap enforces the GPU limit; lanes keep the queue
short.

Dry run by default: prints the lanes and the sbatch lines. ``--submit`` submits.

    .venv/bin/python scripts/submit_chains.py --grid pilot --variant pilot --budget notebooks/methodology/out/budget.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_LANES = 6  # RCI account cap (Jakub, 2026-10-14)
MAIN_REGIMES = ["scratch_all", "scratch_last", "cumulative", "replay_p50", "incremental"]
TASKS = {
    "rel-f1": ["driver-position", "driver-dnf", "driver-top3"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success"],
    "rel-stack": ["user-engagement", "post-votes", "user-badge"],
    "rel-hm": ["user-churn", "item-sales"],
    "rel-amazon": ["user-churn", "user-ltv", "item-churn", "item-ltv"],
    "rel-ratebeer": ["beer-churn", "user-churn", "user-count", "brewer-dormant"],
    "rel-event": ["user-repeat", "user-ignore", "user-attendance"],
    "rel-avito": ["ad-ctr", "user-visits", "user-clicks"],
}
GRIDS = {
    "pilot": ["rel-f1", "rel-trial"],
    "main": ["rel-f1", "rel-trial", "rel-stack", "rel-hm", "rel-amazon", "rel-ratebeer", "rel-event", "rel-avito"],
}
MEM = {"rel-f1": 16, "rel-trial": 32, "rel-stack": 64, "rel-hm": 64, "rel-amazon": 224, "rel-ratebeer": 80,
       "rel-event": 32, "rel-avito": 64}


def chain_list(datasets, regimes, seeds):
    return [(d, t, m, s) for d in datasets for t in TASKS[d] for m in regimes for s in seeds]


def estimate_hours(chains, budget_path=None, default=1.0):
    est = {}
    if budget_path and Path(budget_path).exists():
        for c in json.loads(Path(budget_path).read_text())["chains"]:
            est[(c["dataset"], c["task"], c["mode"])] = float(c["gpu_h_per_chain"])
    return [est.get((d, t, m), default) for d, t, m, _ in chains]


def assign_lanes(chains, hours, lanes):
    """Longest chain first onto the least-loaded lane. Returns per-lane lists and loads."""
    if not 1 <= lanes <= MAX_LANES:
        raise ValueError(f"lanes must be between 1 and {MAX_LANES}, got {lanes}")
    heap = [(0.0, i) for i in range(lanes)]
    heapq.heapify(heap)
    out = [[] for _ in range(lanes)]
    loads = [0.0] * lanes
    for h, c in sorted(zip(hours, chains), key=lambda x: -x[0]):
        load, i = heapq.heappop(heap)
        out[i].append(c)
        loads[i] = load + h
        heapq.heappush(heap, (loads[i], i))
    return out, loads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", choices=sorted(GRIDS), required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--regimes", nargs="+", default=MAIN_REGIMES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lanes", type=int, default=MAX_LANES)
    ap.add_argument("--budget", default=None, help="budget.json from notebook 08")
    ap.add_argument("--job_hours", type=float, default=3.5, help="useful hours per 4 h amdgpufast job")
    ap.add_argument("--spare_jobs", type=int, default=1)
    ap.add_argument("--extra", default="", help="flags passed to chain_runner, e.g. '--holdout_frac 0.2'")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args(argv)

    chains = chain_list(GRIDS[args.grid], args.regimes, args.seeds)
    hours = estimate_hours(chains, args.budget)
    lanes, loads = assign_lanes(chains, hours, args.lanes)
    lane_dir = ROOT / "logs" / "rci" / "cl2" / "lanes" / args.variant
    lane_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(chains)} chains, {sum(hours):.1f} GPU-h estimated, makespan {max(loads):.1f} h on {args.lanes} lanes")
    for i, (lane, load) in enumerate(zip(lanes, loads)):
        path = lane_dir / f"lane{i}.txt"
        path.write_text("".join(f"{d} {t} {m} {s} {args.variant}\n" for d, t, m, s in lane))
        mem = max(MEM[d] for d, *_ in lane) if lane else 16
        n_jobs = math.ceil(load / args.job_hours) + args.spare_jobs
        prev = None
        print(f"lane {i}: {len(lane)} chains, {load:.1f} h, {n_jobs} jobs, {mem}G  -> {path.relative_to(ROOT)}")
        for k in range(n_jobs):
            cmd = ["sbatch", "--parsable", f"--job-name=cl2-{args.variant}-L{i}-{k}", f"--mem={mem}G"]
            if prev:
                cmd.append(f"--dependency=afterany:{prev}")
            cmd += ["slurm/rci/run_chain2.sh", str(path.relative_to(ROOT))] + args.extra.split()
            if args.submit:
                prev = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
                print("   submitted", prev)
            else:
                print("   ", " ".join(cmd))
                prev = f"<job{k}>"
    return 0


if __name__ == "__main__":
    sys.exit(main())
