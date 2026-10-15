# %% [markdown]
# # 07 — Deployment simulation: update frequency × budget (RQ4, rel-f1)
#
# **Purpose.** Test the RQ4 machinery on driver-dnf.
#
# - **Shared start.** One model is trained up to t0. Every configuration adopts it.
# - **Configurations.** Each one (regime, update interval ΔI, steps per update B)
#   then updates on its own schedule.
# - **Timeline score.** Every anchor after t0 is predicted by the latest model
#   updated before it. The score is the mean over anchor times, with a
#   block-bootstrap interval, because neighbouring anchors are correlated.
# - **Cost.** Updates × B steps, plus measured training seconds.
# - **Pareto front** of cost against score.
#
# rel-f1 is not an RQ4 dataset in the paper (too few rows per update); here it only
# exercises the machinery.
#
# **Gate.**
# 1. Every configuration covers the same anchor times after t0, and every anchor
#    has a model.
# 2. Timeline scores and their intervals are finite.
# 3. The Pareto front is computed.

# %% [parameters]
DATASET = "rel-f1"
TASK = "driver-dnf"
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.2
T0 = "2020-01-01"
DELTA_I_DAYS = [30, 60, 120, 365, 0]  # 0 = never update after t0
BUDGETS = [50, 200]
REGIMES = ["incremental", "cumulative", "scratch_all"]
INIT_STEPS = 2000
BLOCK = 3

# %%
import json
import os
import shutil
import sys
from pathlib import Path

root = Path.cwd()
while not (root / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.continuous_learning.chain_runner import ChainConfig, build_context, run_chain
from redelex.continual import analysis as A

torch.set_num_threads(THREADS)
if SCALE == "smoke":
    DELTA_I_DAYS, BUDGETS, REGIMES, INIT_STEPS = [365, 0], [20], ["incremental", "cumulative"], 60
out = Path(OUT)
work = out / "07_rq4"
if SCALE == "smoke":
    shutil.rmtree(work, ignore_errors=True)


def cfg(mode, variant, **kw):
    base = dict(device=DEVICE, torch_threads=THREADS, holdout_frac=HOLDOUT_FRAC, anytime=False, stale_windows=0)
    base.update(kw)
    return ChainConfig(DATASET, TASK, mode, 0, str(work), variant=variant, **base)

# %% [markdown]
# ## The shared model at t0
#
# Scratch on all labelled history up to t0, trained to convergence. Its first
# update time is t0, and every configuration adopts it.

# %%
init_cfg = cfg("scratch_all", "init", first_update=T0, delta_i_days=0, max_steps=INIT_STEPS,
               check_every=min(50, INIT_STEPS // 4), patiences=(5,))
run_chain(init_cfg)
init_model = init_cfg.seed_dir() / "ep001" / "model.pt"
print("shared model at", T0, ":", init_model)

# %% [markdown]
# ## Every configuration

# %%
configs = []
for mode in REGIMES:
    for di in DELTA_I_DAYS:
        for b in BUDGETS if di else BUDGETS[:1]:  # ΔI = ∞ never trains: one budget is enough
            variant = f"di{di}_b{b}"
            c = cfg(mode, variant, first_update=T0, delta_i_days=di, stop_rule="budget", max_steps=b,
                    init_model=str(init_model))
            summary = run_chain(c)
            configs.append({"mode": mode, "ΔI days": di if di else np.inf, "B": b if di else 0, "cfg": c,
                            "updates": summary["episodes_done"]})
pd.DataFrame(configs).drop(columns="cfg")

# %% [markdown]
# ## Timeline score and cost

# %%
results, timelines = [], {}
t0 = pd.Timestamp(T0)
for row in configs:
    c = row["cfg"]
    rows = pd.read_parquet(c.chain_root() / "rows.parquet")
    eps = sorted(c.seed_dir().glob("ep*"))
    preds = {int(p.name[2:]): pd.read_parquet(p / "preds.parquet") for p in eps}
    recs = [json.loads((p / "episode.json").read_text()) for p in eps]
    metric = recs[0]["metric"]
    tl = A.timeline_scores(rows, preds, metric, start=t0)
    mean, lo, hi = A.block_bootstrap_ci(tl, block=BLOCK, n_boot=1000)
    cost = A.horizon_cost(recs)
    key = (row["mode"], row["ΔI days"], row["B"])
    timelines[key] = tl
    results.append({"mode": row["mode"], "ΔI days": row["ΔI days"], "B": row["B"], "updates": row["updates"],
                    "steps": cost["steps"], "train s": round(cost["train_seconds"], 1),
                    "anchor times": len(tl), "score": mean, "ci low": lo, "ci high": hi})
results = pd.DataFrame(results)
results["pareto"] = A.pareto_front(results["steps"], results["score"])
results.sort_values(["mode", "ΔI days", "B"]).round(4)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for mode, g in results.groupby("mode"):
    ax.errorbar(g["steps"] + 1, g["score"], yerr=[g["score"] - g["ci low"], g["ci high"] - g["score"]],
                fmt="o", capsize=3, label=mode)
front = results[results.pareto].sort_values("steps")
ax.plot(front["steps"] + 1, front["score"], "k--", lw=0.8, label="Pareto front")
ax.set_xscale("log")
ax.set_xlabel("total training steps after t0 (+1, log)")
ax.set_ylabel(f"timeline score ({metric}, oriented)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Gate

# %%
lengths = {k: len(v) for k, v in timelines.items()}
assert len(set(lengths.values())) == 1, lengths  # every configuration covers the same anchor times
any_tl = next(iter(timelines.values()))
assert any_tl.index.min() >= t0
assert np.isfinite(results[["score", "ci low", "ci high"]].to_numpy()).all(), results
assert results["pareto"].any()
(out / f"07_rq4_{DATASET}_{SCALE}.json").write_text(results.to_json(orient="records", indent=2))
print("GATE 07 PASSED")
