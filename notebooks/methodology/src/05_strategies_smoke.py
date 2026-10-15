# %% [markdown]
# # 05 — Every update regime, mechanically (rel-f1)
#
# **Purpose.** Run every regime, main and ablation, for three updates on real rel-f1
# data, and check that each one does what its definition says. Training is far too
# short to compare quality; this notebook only checks mechanics:
#
# - **Training rows.** Each update trains on exactly the rows its definition names.
#   Counts are recomputed independently from the rows table: all labelled history,
#   the last episode, the last k episodes, a p mixture, or a reservoir.
# - **Warm and cold starts.** A cold regime never loads weights; a warm regime always
#   does from update 2 on.
# - **Step 0 is the previous model.** A warm update's step-0 anytime score equals the
#   previous update's model scored on the same window, which ties the anytime path
#   to the scoring path. Shrink-and-perturb is the exception: it changes the
#   weights before training starts.
# - **Chain state.** The EWC anchor accumulates; the reservoir stays within capacity
#   and only replays trainable rows.
# - **Resume.** A chain resumed after update 2 equals an uninterrupted one.
#
# **Gate.** Every check above passes for every regime.

# %% [parameters]
DATASET = "rel-f1"
TASK = "driver-dnf"
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.2
STEPS = 40
STEP0_TOL = 0.02  # neighbour sampling differs between the two scoring batches

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
from experiments.continuous_learning.modes import ABLATION_ROSTER, MAIN_ROSTER
from experiments.continuous_learning.score_chain import metric_value
from redelex.continual.anchors import ROW_COL

torch.set_num_threads(THREADS)
out = Path(OUT)
work = out / "05_work"
shutil.rmtree(work, ignore_errors=True)
REGIMES = [m for m in MAIN_ROSTER + ABLATION_ROSTER]


def cfg(mode, variant="smoke", **kw):
    base = dict(device=DEVICE, torch_threads=THREADS, holdout_frac=HOLDOUT_FRAC, max_steps=STEPS,
                check_every=10, patiences=(2,), max_updates=3, anytime_schedule=(0, 10))
    base.update(kw)
    return ChainConfig(DATASET, TASK, mode.name if hasattr(mode, "name") else mode, 0, str(work), variant=variant, **base)


ctx = build_context(cfg("cumulative"))
rows = ctx.rows
dw = ctx.task.timedelta
eps = ctx.plan.episodes

# %% [markdown]
# ## Run every regime

# %%
records = {}
for spec in REGIMES:
    c = cfg(spec)
    run_chain(c, ctx=ctx)
    records[spec.name] = [json.loads((c.seed_dir() / f"ep{e:03d}" / "episode.json").read_text()) for e in (1, 2, 3)]
print("ran", len(records), "regimes x 3 updates")

# %% [markdown]
# ## Training rows match each definition

# %%
def expected_rows(spec, e):
    ep = eps[e - 1]
    t = rows["time"]
    train = ~rows["holdout"].to_numpy()
    labelled = (t <= ep.update_time - dw).to_numpy() & train
    new = labelled & ((t > ep.new_after).to_numpy() if ep.new_after is not None else True)
    if spec.buffer_size is not None:
        return {"new": int(new.sum())}
    if spec.history == "all" and spec.new_share is None:
        return {"all": int(labelled.sum())}
    if spec.history == "all":
        return {"new": int(new.sum()), "old": int((labelled & ~new).sum())}
    if spec.history == "last":
        return {"new": int(new.sum())}
    oldest = e - spec.window_episodes
    after = eps[oldest - 1].update_time - dw if oldest >= 1 else None
    win = labelled & ((t > after).to_numpy() if after is not None else True)
    return {"window": int(win.sum())}


checks = []
for spec in REGIMES:
    for rec in records[spec.name]:
        e = rec["episode"]
        exp = expected_rows(spec, e)
        got = {k: v for k, v in rec["train_rows"].items() if k in exp}
        checks.append({"regime": spec.name, "update": e, "rows ok": got == exp,
                       "warm ok": rec["warm_started"] == (spec.warm_start and e > 1),
                       "buffer rows": rec["train_rows"].get("buffer")})
checks = pd.DataFrame(checks)
checks.groupby("regime")[["rows ok", "warm ok"]].all()

# %% [markdown]
# ## Step 0 of a warm update is the previous model

# %%
def window_score(regime, model_update, window):
    c = cfg(regime)
    preds = pd.read_parquet(c.seed_dir() / f"ep{model_update:03d}" / "preds.parquet").merge(rows, on=ROW_COL)
    w = preds[preds.window == window]
    return metric_value(ctx.task.task_type, w["target"], w["pred"])


step0 = []
metric = records["cumulative"][0]["metric"]
for spec in REGIMES:
    for rec in records[spec.name][1:]:
        e = rec["episode"]
        at0 = rec["anytime"][0][metric]
        prev = window_score(spec.name, e - 1, e + 1)
        step0.append({"regime": spec.name, "update": e, "step-0 score": round(at0, 4),
                      "previous model on same window": round(prev, 4), "diff": round(abs(at0 - prev), 4),
                      "warm": spec.warm_start, "shrink_perturb": spec.shrink is not None})
step0 = pd.DataFrame(step0)
step0

# %% [markdown]
# ## Chain state of the stateful ablations

# %%
state = []
held = set(rows.loc[rows.holdout, ROW_COL])
for name in ("ewc", "reservoir"):
    for rec in records[name]:
        cl = rec["cl_state"]
        st = torch.load(cfg(name).seed_dir() / f"ep{rec['episode']:03d}" / "cl_state.pt", weights_only=False)
        buf = st["buffer"]["row_ids"] if st.get("buffer") else np.array([], dtype=int)
        state.append({"regime": name, "update": rec["episode"], **cl,
                      "buffer ⊆ trainable": held.isdisjoint(buf)})
state = pd.DataFrame(state)
state

# %% [markdown]
# ## Resume equals an uninterrupted chain

# %%
straight = cfg("replay_p50", variant="resume_a")
resumed = cfg("replay_p50", variant="resume_b", max_updates=2)
run_chain(straight, ctx=ctx)
run_chain(resumed, ctx=ctx)
resumed.max_updates = 3
again = run_chain(resumed, ctx=ctx)
a = pd.read_parquet(straight.seed_dir() / "ep003" / "preds.parquet")
b = pd.read_parquet(resumed.seed_dir() / "ep003" / "preds.parquet")
resume_ok = again["ran_now"] == [3] and a.equals(b)
print("resume ran update", again["ran_now"], "identical predictions:", a.equals(b))

# %% [markdown]
# ## Gate

# %%
assert checks["rows ok"].all(), checks[~checks["rows ok"]]
assert checks["warm ok"].all(), checks[~checks["warm ok"]]
plain_warm = step0[step0.warm & ~step0.shrink_perturb]
assert (plain_warm["diff"] <= STEP0_TOL).all(), plain_warm
assert (step0[step0.shrink_perturb]["diff"] > 0).all()
ewc = state[state.regime == "ewc"]
assert ewc["anchor_updates"].tolist() == [1, 2, 3] and (ewc["anchor_params"] > 0).all()
res = state[state.regime == "reservoir"]
assert (res["buffer_rows"] <= 10_000).all() and res["buffer ⊆ trainable"].all()
assert resume_ok
(out / f"05_strategies_{DATASET}.json").write_text(json.dumps(
    {"checks": checks.to_dict(orient="records"), "step0": step0.to_dict(orient="records"),
     "state": state.to_dict(orient="records"), "resume_ok": bool(resume_ok)}, indent=2, default=str))
print("GATE 05 PASSED")
