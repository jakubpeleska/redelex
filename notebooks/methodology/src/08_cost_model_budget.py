# %% [markdown]
# # 08 — Cost model and budget for the full rerun
#
# **Purpose.** Before the grid launches, estimate what it costs. Three inputs are
# combined:
#
# 1. **Steps to convergence per update** (notebook 03 probes). These are modelled as
#    a function of the regime and the number of training rows, then replayed with the
#    chosen stopping rule.
# 2. **Per-dataset costs** (`scripts/benchmark_step_cost.py` on RCI A100s): seconds
#    per training step, scored rows per second, and the plan's rows per update.
# 3. **The grid**: 5 main regimes × 5 seeds × every main-grid task.
#
# Output: GPU-hours per chain and in total, the makespan on 6 GPUs (longest chain
# first), and which RCI partition each chain fits. This is the estimate behind
# checkpoint D; the rel-f1 + rel-trial pilot (checkpoint E) then checks it.
#
# **Extrapolation warning.** The probes cover rel-f1's row counts (hundreds to ~15k
# rows per update). Larger datasets reach millions, so their step counts are
# extrapolated, and each one is flagged.
#
# **Gate.**
# 1. Every benchmarked task has an estimate.
# 2. The total is below `BUDGET_GPU_H` (infinite by default: informational).
# 3. `budget.json` is written.

# %% [parameters]
OUT = "notebooks/methodology/out"
CALIBRATION_DATASET = "rel-f1"
SCALE = "full"
GPUS = 6
SEEDS = 5
REGIMES = ["scratch_all", "scratch_last", "cumulative", "replay_p50", "incremental"]
BUDGET_GPU_H = float("inf")
OVERHEAD = 1.15  # validation checks, setup, saving: on top of training and scoring
PARTITIONS = {"amdgpufast": 4.0, "amdgpu": 24.0, "amdgpulong": 72.0}  # hours
CHAIN_JOB_LIMIT_H = 3.5  # a job stops before its deadline and the chain resumes

# %%
import json
import os
import sys
from pathlib import Path

root = Path.cwd()
while not (root / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
sys.path.insert(0, str(root))

import heapq

import numpy as np
import pandas as pd

from redelex.continual.analysis import replay_stopping

out = Path(OUT)
bench = [json.loads(p.read_text()) for p in sorted((out / "bench").glob("*__gpu.json"))]
rule_path = out / f"03_stopping_rule_{CALIBRATION_DATASET}_{SCALE}.json"
rule = json.loads(rule_path.read_text())["choice"] if rule_path.exists() else None
print(len(bench), "benchmarked tasks; stopping rule:", rule)

# %% [markdown]
# ## Steps to convergence vs. training rows
#
# For every probe update: the rows it trained on, and where the chosen rule stops.
# Fit `log(steps) = a + b log(rows)` per regime family. Cold-start regimes need more
# steps than warm ones.

# %%
probe = []
for path in sorted((out / "03_probe").glob("*__probe/seed*/ep*/episode.json")):
    rec = json.loads(path.read_text())
    task, mode = path.parts[-4].split("__")[1:3]
    rows = sum(rec["train_rows"].values())
    if rule is not None:
        n = rule["check_every"]
        n = max(10, int(round(0.25 * rec["steps_per_epoch"] / 10.0)) * 10) if n == "epoch/4" else int(n)
        hist = rec["convergence"]["history"] if rule["monitor"] == "val_loss" else rec["convergence"]["history_extra"][f"val_{rec['metric']}"]
        mon = "min" if (rule["monitor"] == "val_loss" or rec["metric"] == "mae") else "max"
        stop = replay_stopping(hist, every=n, patience=rule["P1"], mode=mon)["stop_step"]
    else:
        stop = rec["cost"]["steps"]
    probe.append({"task": task, "mode": mode, "warm": mode != "scratch_all" and rec["episode"] > 1,
                  "rows": rows, "stop": stop, "s_per_step_probe": rec["cost"]["train_seconds"] / max(rec["cost"]["steps"], 1)})
probe = pd.DataFrame(probe)
fits = {}
for warm, g in probe.groupby("warm"):
    g = g[(g.rows > 0) & (g.stop > 0)]
    b, a = np.polyfit(np.log(g.rows), np.log(g.stop), 1) if len(g) >= 3 else (0.0, np.log(max(g.stop.median(), 1)))
    fits[bool(warm)] = (a, b, float(g.rows.min()), float(g.rows.max()))
pd.DataFrame(fits, index=["a", "b", "rows min", "rows max"]).T.rename_axis("warm start")


def steps_for(rows, warm):
    a, b, lo, hi = fits.get(bool(warm), next(iter(fits.values())))
    return float(np.exp(a + b * np.log(max(rows, 1)))), not (lo <= rows <= hi)

# %% [markdown]
# ## GPU-hours per chain

# %%
WARM = {"scratch_all": False, "scratch_last": False, "cumulative": True, "replay_p50": True, "incremental": True}
chains = []
for b in bench:
    for mode in REGIMES:
        hours, extrapolated = 0.0, False
        for u in b["per_update"]:
            rows = u["labelled"] if mode in ("scratch_all", "cumulative", "replay_p50") else u["new"]
            steps, ext = steps_for(rows, WARM[mode] and u["update"] > 1)
            extrapolated |= ext
            scored = u["deploy"] * (b["updates"] - u["update"] + 1)  # next and later windows
            hours += (steps * b["s_per_step"] + scored / max(b["eval_rows_per_s"] or 1, 1)) / 3600
        chains.append({"dataset": b["dataset"], "task": b["task"], "mode": mode, "updates": b["updates"],
                       "gpu_h_per_chain": hours * OVERHEAD, "extrapolated": extrapolated})
chains = pd.DataFrame(chains)
chains["gpu_h_all_seeds"] = chains.gpu_h_per_chain * SEEDS
by_task = chains.groupby(["dataset", "task"]).agg(updates=("updates", "first"), gpu_h=("gpu_h_all_seeds", "sum"),
                                                  extrapolated=("extrapolated", "any"))
by_task.sort_values("gpu_h", ascending=False).round(2)

# %% [markdown]
# ## Makespan on the GPU allowance, and partition fit
#
# Chains are independent. Longest first onto the least-loaded GPU gives the
# makespan. A chain longer than one job's limit is split across jobs that resume.

# %%
jobs = []
for _, c in chains.iterrows():
    for seed in range(SEEDS):
        jobs.append((c.gpu_h_per_chain, f"{c.dataset}/{c.task}/{c['mode']}/s{seed}"))
jobs.sort(reverse=True)
lanes = [(0.0, i) for i in range(GPUS)]
heapq.heapify(lanes)
for hours, _ in jobs:
    load, i = heapq.heappop(lanes)
    heapq.heappush(lanes, (load + hours, i))
makespan_h = max(load for load, _ in lanes)
total_h = float(chains.gpu_h_all_seeds.sum())
chains["jobs_at_limit"] = np.ceil(chains.gpu_h_per_chain / CHAIN_JOB_LIMIT_H).astype(int)
chains["fits"] = [next((p for p, lim in PARTITIONS.items() if h <= lim * 0.9), "split") for h in chains.gpu_h_per_chain]
print(f"total {total_h:.1f} GPU-h for {len(jobs)} chains; makespan on {GPUS} GPUs {makespan_h:.1f} h ({makespan_h / 24:.1f} days)")
chains.groupby("fits").size().rename("chains")

# %% [markdown]
# ## Gate

# %%
assert len(bench) > 0, "no benchmarks found in out/bench; run slurm/rci/submit_benchmarks.sh first"
assert chains.gpu_h_per_chain.notna().all()
assert total_h <= BUDGET_GPU_H, f"estimated {total_h:.0f} GPU-h exceeds the budget {BUDGET_GPU_H}"
budget = {"total_gpu_h": total_h, "makespan_h": makespan_h, "gpus": GPUS, "seeds": SEEDS, "regimes": REGIMES,
          "stopping_rule": rule, "fits": {str(k): v for k, v in fits.items()},
          "chains": chains.to_dict(orient="records")}
(out / "budget.json").write_text(json.dumps(budget, indent=2, default=str))
print("GATE 08 PASSED")
