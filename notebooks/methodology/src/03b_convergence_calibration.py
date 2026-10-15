# %% [markdown]
# # 03b — Choosing the stopping rule (rel-f1)
#
# **Purpose.** Replay every candidate stopping rule on the probe histories of 03a and
# pick the cheapest one that does not cost next-window quality.
#
# For each probe update and each rule (check interval N, patience P, monitored value):
# - **stop step**: where the rule would have stopped; this is the cost;
# - **chosen step**: the rule's best check, i.e. the weights the update keeps;
# - **regret**: the best next-window score the run reached at any step, minus the
#   score at the chosen step (oriented, higher is better).
#
# The rule never sees the next window; regret uses it only to judge the rule. This
# notebook also compares the constant learning rate with a plateau schedule.
#
# **Gate.**
# 1. The chosen rule fires before the probe's end on at least 95% of updates.
# 2. Its median regret is no worse than `REGRET_TOL` above the best rule's.
# 3. The choice (N, P1, P2, monitored value, safety cap) is saved.

# %% [parameters]
DATASET = "rel-f1"
OUT = "notebooks/methodology/out"
EVERY = [10, 20, 50, 100, 200]
PATIENCE = [3, 5, 10, 20]
MONITORS = ["val_loss", "metric"]
REGRET_TOL = 0.005  # oriented score units: AUROC points, or MAE units on regression
MIN_FIRED = 0.95

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

import numpy as np
import pandas as pd

from redelex.continual.analysis import orient, replay_stopping

probe_root = Path(OUT) / "03_probe"
records = []
for path in sorted(probe_root.glob("*/seed*/ep*/episode.json")):
    rec = json.loads(path.read_text())
    chain = path.parts[-4]
    _, task, mode, variant = chain.split("__")
    records.append({"task": task, "mode": mode, "variant": variant, "seed": rec["seed"], "episode": rec["episode"],
                    "metric": rec["metric"], "steps_per_epoch": rec["steps_per_epoch"], "rec": rec})
print(len(records), "probe updates from", probe_root)
pd.DataFrame(records).groupby(["task", "mode", "variant"]).size().rename("updates")

# %% [markdown]
# ## Replay every rule on every probe update

# %%
def deploy_at(anytime, metric, step):
    """Oriented next-window score of the anytime point at or just before ``step``."""
    pts = [(r["step"], r[metric]) for r in anytime if r["step"] <= step]
    return float(orient(pts[-1][1], metric)) if pts else np.nan


rows = []
for r in records:
    rec, metric = r["rec"], r["metric"]
    anytime = sorted(rec["anytime"], key=lambda a: a["step"])
    oracle = max(float(orient(a[metric], metric)) for a in anytime)
    hist = {"val_loss": rec["convergence"]["history"],
            "metric": rec["convergence"]["history_extra"][f"val_{metric}"]}
    every_epoch = max(10, int(round(0.25 * r["steps_per_epoch"] / 10.0)) * 10)
    for monitor in MONITORS:
        mode = "min" if (monitor == "val_loss" or metric == "mae") else "max"
        for every in EVERY + ["epoch/4"]:
            n = every_epoch if every == "epoch/4" else every
            for p in PATIENCE:
                res = replay_stopping(hist[monitor], every=n, patience=p, mode=mode)
                if res["best_step"] is None:
                    continue
                chosen = deploy_at(anytime, metric, res["best_step"])
                rows.append({"task": r["task"], "mode": r["mode"], "variant": r["variant"], "seed": r["seed"],
                             "episode": r["episode"], "monitor": monitor, "every": str(every), "N": n, "P": p,
                             "fired": res["fired"], "stop_step": res["stop_step"], "best_step": res["best_step"],
                             "regret": oracle - chosen})
replay = pd.DataFrame(rows)
const = replay[replay.variant == "probe"]
summary = (const.groupby(["monitor", "every", "P"])
           .agg(fired=("fired", "mean"), median_regret=("regret", "median"), p90_regret=("regret", lambda x: x.quantile(0.9)),
                median_stop=("stop_step", "median"), p99_stop=("stop_step", lambda x: x.quantile(0.99)))
           .reset_index())
summary.sort_values(["monitor", "median_stop"]).round(4)

# %% [markdown]
# ## Pick the rule
#
# Among the rules that fire on at least `MIN_FIRED` of updates and whose median regret
# is within `REGRET_TOL` of the best rule's, take the one with the lowest median stop
# step. P2, the sensitivity patience, is the next larger P with the same N.

# %%
ok = summary[summary.fired >= MIN_FIRED]
best_regret = ok.median_regret.min() if len(ok) else np.nan
eligible = ok[ok.median_regret <= best_regret + REGRET_TOL]
pick = eligible.sort_values(["median_stop", "median_regret"]).iloc[0] if len(eligible) else None
if pick is not None:
    larger = sorted(p for p in PATIENCE if p > pick.P)
    p2 = larger[0] if larger else int(pick.P)
    cap = int(np.ceil(max(summary.p99_stop.max(), 1) * 2 / 100.0) * 100)
    choice = {"monitor": pick.monitor, "check_every": pick.every, "P1": int(pick.P), "P2": int(p2),
              "median_regret": float(pick.median_regret), "median_stop_steps": float(pick.median_stop),
              "fired": float(pick.fired), "safety_cap_steps": cap}
else:
    choice = None
choice

# %% [markdown]
# ## Per task and regime, under the chosen rule

# %%
if pick is not None:
    chosen = const[(const.monitor == pick.monitor) & (const["every"] == pick.every) & (const.P == pick.P)]
    display(chosen.groupby(["task", "mode"])[["stop_step", "regret"]].describe().round(3))

# %% [markdown]
# ## Constant LR vs. plateau schedule
#
# Same rule, same updates. With a plateau schedule the trajectory depends on the
# stopping rule, so this is a comparison of outcomes only.

# %%
if pick is not None and (replay.variant == "probe_plateau").any():
    key = ["task", "mode", "seed", "episode"]
    sel = replay[(replay.monitor == pick.monitor) & (replay["every"] == pick.every) & (replay.P == pick.P)]
    a = sel[sel.variant == "probe"].set_index(key)[["stop_step", "regret"]]
    b = sel[sel.variant == "probe_plateau"].set_index(key)[["stop_step", "regret"]]
    both = a.join(b, lsuffix="_constant", rsuffix="_plateau", how="inner")
    display(both.describe().round(3))

# %% [markdown]
# ## Gate

# %%
assert len(records) > 0, "no probe updates found; run 03a first"
assert choice is not None, "no rule fires often enough"
assert choice["fired"] >= MIN_FIRED
(Path(OUT) / f"03_stopping_rule_{DATASET}.json").write_text(json.dumps({"choice": choice, "summary": summary.to_dict(orient="records")}, indent=2, default=str))
print("GATE 03b PASSED", choice)
