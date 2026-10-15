# %% [markdown]
# # 06 — From chains to the paper's numbers (rel-f1)
#
# **Purpose.** Run the five main regimes as independent per-seed chains and compute
# every metric the paper reports, end to end, from the files the chains write.
#
# - **R matrix** (model updated at T_e, scored on window j): next-window score R[e, e+1].
# - **Staleness penalty** S(k) = mean_j R[j−1, j] − R[j−1−k, j].
# - **Held-out BWT** on rows never trained on.
# - **Anytime curves.** Area under the curve, and steps to reach the converged
#   scratch score minus one SD.
# - **Ranks across tasks**, with Friedman and Wilcoxon–Holm tests.
#
# At smoke scale the chains are far too short for the numbers to mean anything.
# This notebook checks the pipeline: index conventions, schemas, seed handling.
#
# **Gate.**
# 1. Every chain is complete.
# 2. R has the expected shape and the next-window diagonal is finite.
# 3. Every metric is computed for every task, regime and seed.
# 4. The seeds are independent chains: same protocol, different weights.

# %% [parameters]
DATASET = "rel-f1"
TASKS = ["driver-position", "driver-dnf", "driver-top3"]
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.2
SEEDS = [0, 1]
MAX_STEPS = 2000
CHECK_EVERY = 50
PATIENCES = (5, 10)
MAX_UPDATES = None
STALE_WINDOWS = 4

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

import numpy as np
import pandas as pd
import torch

from experiments.continuous_learning.chain_runner import ChainConfig, build_context, run_chain
from experiments.continuous_learning.modes import MAIN_ROSTER
from redelex.continual import analysis as A

torch.set_num_threads(THREADS)
if SCALE == "smoke":
    TASKS, MAX_STEPS, CHECK_EVERY, PATIENCES, MAX_UPDATES = TASKS[1:2], 30, 10, (2, 3), 3
out = Path(OUT)
work = out / "06_chains"
REGIMES = [m.name for m in MAIN_ROSTER]


def cfg(task, mode, seed):
    return ChainConfig(DATASET, task, mode, seed, str(work), variant="analysis", device=DEVICE,
                       torch_threads=THREADS, holdout_frac=HOLDOUT_FRAC, max_steps=MAX_STEPS,
                       check_every=CHECK_EVERY, patiences=PATIENCES, max_updates=MAX_UPDATES,
                       stale_windows=STALE_WINDOWS)

# %% [markdown]
# ## Run the chains

# %%
status = []
for task in TASKS:
    ctx = build_context(cfg(task, REGIMES[0], SEEDS[0]))
    for mode in REGIMES:
        for seed in SEEDS:
            s = run_chain(cfg(task, mode, seed), ctx=ctx)
            status.append({"task": task, "mode": mode, "seed": seed, "status": s["status"], "updates": s["episodes_done"]})
status = pd.DataFrame(status)
status.pivot_table(index=["task", "mode"], columns="seed", values="updates")

# %% [markdown]
# ## Load predictions and build R

# %%
def load_chain(task, mode, seed):
    c = cfg(task, mode, seed)
    rows = pd.read_parquet(c.chain_root() / "rows.parquet")
    eps = sorted(c.seed_dir().glob("ep*"))
    preds = {(seed, int(p.name[2:])): pd.read_parquet(p / "preds.parquet") for p in eps}
    recs = [json.loads((p / "episode.json").read_text()) for p in eps]
    return rows, preds, recs


long_all, recs_all = [], []
for task in TASKS:
    for mode in REGIMES:
        for seed in SEEDS:
            rows, preds, recs = load_chain(task, mode, seed)
            metric = recs[0]["metric"]
            lg = A.score_long(rows, preds, metric)
            lg["task"], lg["mode"], lg["metric"] = task, mode, metric
            long_all.append(lg)
            for r in recs:
                recs_all.append({"task": task, "mode": mode, "seed": seed, **{k: r[k] for k in ("episode", "metric")},
                                 "rec": r})
long_all = pd.concat(long_all, ignore_index=True)
task0, mode0 = TASKS[0], "cumulative"
sel = long_all[(long_all.task == task0) & (long_all["mode"] == mode0)]
R = A.matrix(sel, sel.metric.iloc[0])
print(f"R for {task0} / {mode0} (oriented, mean over seeds); rows = model e, columns = window j")
R.round(3)

# %% [markdown]
# ## Per task, regime and seed: next-window score, staleness, held-out BWT

# %%
summary = []
for (task, mode, seed), g in long_all.groupby(["task", "mode", "seed"]):
    metric = g.metric.iloc[0]
    Rs = A.matrix(g, metric, seed=seed)
    Rh = A.matrix(g, metric, subset="holdout", seed=seed)
    nxt = A.next_window(Rs)
    s1, _ = A.staleness_penalty(Rs, 1)
    s2, _ = A.staleness_penalty(Rs, 2)
    summary.append({"task": task, "mode": mode, "seed": seed, "next-window mean": nxt.mean(),
                    "S(1)": s1, "S(2)": s2, "holdout BWT": A.holdout_bwt(Rh)})
summary = pd.DataFrame(summary)
summary.groupby(["task", "mode"])[["next-window mean", "S(1)", "S(2)", "holdout BWT"]].mean().round(4)

# %% [markdown]
# ## Anytime curves: area and steps to the scratch target

# %%
anytime = []
for r in recs_all:
    rec, metric = r["rec"], r["metric"]
    pts = sorted(rec["anytime"], key=lambda a: a["step"])
    steps = [p["step"] for p in pts]
    scores = list(A.orient([p[metric] for p in pts], metric))
    anytime.append({"task": r["task"], "mode": r["mode"], "seed": r["seed"], "episode": r["episode"],
                    "steps": steps, "scores": scores, "final": scores[-1], "stop": rec["cost"]["steps"]})
anytime = pd.DataFrame(anytime)
targets = (anytime[anytime["mode"] == "scratch_all"].groupby(["task", "episode"])["final"]
           .apply(A.scratch_target).rename("target"))
anytime = anytime.join(targets, on=["task", "episode"])
horizon = int(anytime["stop"].max())
anytime["AUC"] = [A.anytime_auc(s, v, horizon) for s, v in zip(anytime.steps, anytime.scores)]
anytime["steps to target"] = [A.steps_to_target(s, v, t) for s, v, t in zip(anytime.steps, anytime.scores, anytime.target)]
anytime.groupby(["task", "mode"])[["AUC", "steps to target", "stop"]].median().round(3)

# %% [markdown]
# ## Ranks across tasks
#
# With one or three rel-f1 tasks these statistics have no power; the cell shows the
# computation that runs on every task of the full grid.

# %%
table = summary.groupby(["task", "mode"])["next-window mean"].mean().unstack("mode")
ranks = A.average_ranks(table)
display(ranks.rename("average rank"))
if len(table) >= 2:
    stat, p = A.friedman(table)
    print(f"Friedman chi2 = {stat:.2f}, p = {p:.3f} over {len(table)} tasks")

# %% [markdown]
# ## Gate

# %%
assert (status["status"] == "complete").all(), status
n_upd = status.groupby("task")["updates"].first()
for task in TASKS:
    g = long_all[(long_all.task == task) & (long_all["mode"] == mode0)]
    Rt = A.matrix(g, g.metric.iloc[0])
    nxt = A.next_window(Rt)
    assert len(nxt) == n_upd[task] and np.isfinite(nxt).all(), (task, nxt)
    assert Rt.index.min() == 1 and Rt.columns.max() <= n_upd[task] + 1 + STALE_WINDOWS
assert summary[["next-window mean", "S(1)"]].notna().all().all()
assert anytime["AUC"].notna().all()
# Seeds are independent chains: same protocol, different weights.
a = torch.load(cfg(TASKS[0], "cumulative", SEEDS[0]).seed_dir() / "ep002" / "model.pt")
b = torch.load(cfg(TASKS[0], "cumulative", SEEDS[1]).seed_dir() / "ep002" / "model.pt")
assert any(not torch.equal(a[k], b[k]) for k in a)
(out / f"06_analysis_{DATASET}.json").write_text(json.dumps(
    {"summary": summary.to_dict(orient="records"), "ranks": ranks.to_dict()}, indent=2, default=str))
print("GATE 06 PASSED")
