# %% [markdown]
# # 04 — Anytime evaluation and the cost meter (rel-f1)
#
# **Purpose.** The compute results (RQ2, RQ4) rest on two instruments.
#
# - **Anytime curves.** The next-window score at fixed optimiser steps of every
#   update. Evaluating must not change what is trained.
# - **Cost meter.** It splits each update's wall-clock into training compute, data
#   loading, validation and anytime evaluation, and counts samples per source.
#   Its buckets must add up, and the mixing shares it measures must match p.
#
# All three checks run on the real rel-f1 model and graph.
#
# **Gate.**
# 1. Final weights are bit-identical with and without anytime evaluation (CPU).
# 2. Every metered update's buckets account for its fit wall-clock within `RECONCILE_TOL`.
# 3. The measured new-data share of every mixing regime is within ±0.02 of p.

# %% [parameters]
DATASET = "rel-f1"
TASK = "driver-dnf"
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.2
STEPS = 120
RECONCILE_TOL = 0.10  # wall-clock not covered by any bucket (Lightning's own loop overhead)
MIX_MODES = {"replay_p50": 0.5, "replay_p25": 0.25}

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

import pandas as pd
import torch

from experiments.continuous_learning.chain_runner import ChainConfig, build_context, run_chain

torch.set_num_threads(THREADS)
if SCALE == "smoke":
    STEPS = min(STEPS, 60)
out = Path(OUT)
work = out / "04_work"
shutil.rmtree(work, ignore_errors=True)


def cfg(mode, variant, **kw):
    base = dict(device=DEVICE, torch_threads=THREADS, holdout_frac=HOLDOUT_FRAC, max_steps=STEPS,
                check_every=20, patiences=(1000,), max_updates=2)
    base.update(kw)
    return ChainConfig(DATASET, TASK, mode, 0, str(work), variant=variant, **base)


ctx = build_context(cfg("cumulative", "x"))

# %% [markdown]
# ## 1. Anytime evaluation does not change training
#
# The same two updates, once with the next window scored every 10 steps and once
# with no anytime evaluation at all. Only on CPU is the comparison bitwise.

# %%
with_any = cfg("cumulative", "anytime_on", anytime_every=10)
without = cfg("cumulative", "anytime_off", anytime=False)
run_chain(with_any, ctx=ctx)
run_chain(without, ctx=ctx)

identical = {}
for e in (1, 2):
    a = torch.load(with_any.seed_dir() / f"ep{e:03d}" / "model.pt")
    b = torch.load(without.seed_dir() / f"ep{e:03d}" / "model.pt")
    identical[e] = all(torch.equal(a[k], b[k]) for k in a)
rec = json.loads((with_any.seed_dir() / "ep002" / "episode.json").read_text())
print("bit-identical final weights per update:", identical)
print("anytime points of update 2:", [r["step"] for r in rec["anytime"]][:8], "...")

# %% [markdown]
# ## 2. The cost meter's buckets add up

# %%
rows = []
for c in (with_any, without):
    for ep_dir in sorted(c.seed_dir().glob("ep*")):
        r = json.loads((ep_dir / "episode.json").read_text())
        s = r["cost"]["seconds"]
        accounted = s["train_compute"] + s["train_data"] + s["val"] + s.get("anytime", 0.0)
        rows.append({"variant": c.variant, "update": r["episode"], "steps": r["cost"]["steps"],
                     "fit wall s": round(r["cost"]["fit_wall_seconds"], 2),
                     "train compute s": round(s["train_compute"], 2), "data s": round(s["train_data"], 2),
                     "val s": round(s["val"], 2), "anytime s": round(s.get("anytime", 0.0), 2),
                     "unaccounted": round(1 - accounted / r["cost"]["fit_wall_seconds"], 3),
                     "val / train": round(s["val"] / max(s["train_compute"] + s["train_data"], 1e-9), 3)})
cost = pd.DataFrame(rows)
cost

# %% [markdown]
# ## 3. Mixing shares as measured during training

# %%
shares = []
for mode, p in MIX_MODES.items():
    c = cfg(mode, "mix", anytime=False)
    run_chain(c, ctx=ctx)
    for ep_dir in sorted(c.seed_dir().glob("ep*")):
        r = json.loads((ep_dir / "episode.json").read_text())
        by = r["cost"]["samples_by_source"]
        total = sum(by.values())
        has_old = r["train_rows"].get("old", 0) > 0
        shares.append({"mode": mode, "update": r["episode"], "p": p if has_old else 1.0,
                       "new share measured": round(by.get("new", 0) / total, 4), "samples": total,
                       "new rows": r["train_rows"]["new"], "old rows": r["train_rows"].get("old", 0)})
shares = pd.DataFrame(shares)
shares

# %% [markdown]
# ## Gate

# %%
assert all(identical.values()), identical
assert (cost["unaccounted"].abs() <= RECONCILE_TOL).all(), cost
assert ((shares["new share measured"] - shares["p"]).abs() <= 0.02).all(), shares
(out / f"04_anytime_cost_{DATASET}.json").write_text(json.dumps(
    {"identical": identical, "cost": cost.to_dict(orient="records"), "shares": shares.to_dict(orient="records")},
    indent=2, default=str))
print("GATE 04 PASSED")
