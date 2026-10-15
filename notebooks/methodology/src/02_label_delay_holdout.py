# %% [markdown]
# # 02 — Label delay and held-out entities (rel-f1)
#
# **Purpose.** Two protocol rules decide what an update may see.
#
# 1. **Label delay.** At update time T only anchors with t ≤ T − Δw have complete
#    labels. This notebook counts what the rule defers, on the native grid and on
#    the dense one.
# 2. **Held-out entities.** A fixed fraction of entities is never trained on. Their
#    rows across the whole last episode decide when an update stops, and their rows
#    in past windows measure forgetting. On rel-f1 there are few drivers per
#    window, so this notebook measures how noisy that stopping signal is at 10, 20
#    and 25%, and picks the fraction per task.
#
# It also audits every regime end to end: no held-out entity may ever be a
# training seed.
#
# **Gate.**
# 1. No training row has t > T − Δw.
# 2. The leak audit is clean for every regime.
# 3. A holdout fraction is chosen for every task and saved.

# %% [parameters]
DATASET = "rel-f1"
TASKS = ["driver-position", "driver-dnf", "driver-top3"]
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
FRACS = [0.1, 0.2, 0.25]
MIN_SELECTION_ENTITIES = 8  # per update, for the stopping signal to mean something
NOISE_STEPS = 150  # training steps of the probe model used to measure noise
AUDIT_MODES = ["scratch_all", "scratch_last", "cumulative", "replay_p50", "incremental",
               "replay_p25", "scratch_k2", "shrink_perturb"]

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
import torch

from experiments.continuous_learning.chain_runner import ChainConfig, build_context, run_chain
from experiments.continuous_learning.score_chain import metric_value, predict_table
from redelex.continual.anchors import HOLDOUT_COL
from redelex.continual.protocol import entity_holdout_mask, trainable_mask

torch.set_num_threads(THREADS)
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
contexts = {t: build_context(ChainConfig(DATASET, t, "scratch_all", 0, OUT, device=DEVICE)) for t in TASKS}

# %% [markdown]
# ## 1. Label delay
#
# For each update: the rows of its own window [T_{e−1}, T_e), and how many of those are
# still unlabelled at T_e. Deferred rows become trainable at the next update. Every
# anchor is newly labelled exactly once.

# %%
delay = []
for name, ctx in contexts.items():
    times = ctx.wrapper.full_table.df[ctx.task.time_col]
    for ep in ctx.plan.episodes:
        lo, hi = ctx.plan.window_bounds(ep.index)
        in_window = (times < hi) if lo is None else ((times >= lo) & (times < hi))
        labelled = trainable_mask(times, ep.update_time, ctx.task.timedelta)
        new = trainable_mask(times, ep.update_time, ctx.task.timedelta, after=ep.new_after)
        delay.append({"task": name, "update": ep.index, "T": ep.update_time.date(),
                      "window rows": int(in_window.sum()),
                      "deferred by label delay": int((in_window & ~labelled).sum()),
                      "newly labelled rows": int(new.sum())})
delay = pd.DataFrame(delay)
delay.groupby("task")[["window rows", "deferred by label delay", "newly labelled rows"]].sum()

# %%
delay[delay.task == TASKS[0]]

# %% [markdown]
# On a dense grid (stride Δw/4) the rule removes more: every anchor in the last Δw
# before an update waits for the next one.

# %%
dense = []
for name in TASKS:
    dctx = build_context(ChainConfig(DATASET, name, "scratch_all", 0, OUT, device=DEVICE,
                                     anchor_stride_days=contexts[name].task.timedelta.days / 4))
    times = dctx.wrapper.full_table.df[dctx.task.time_col]
    deferred = rows = 0
    for ep in dctx.plan.episodes:
        lo, hi = dctx.plan.window_bounds(ep.index)
        in_window = (times < hi) if lo is None else ((times >= lo) & (times < hi))
        deferred += int((in_window & ~trainable_mask(times, ep.update_time, dctx.task.timedelta)).sum())
        rows += int(in_window.sum())
    dense.append({"task": name, "rows": rows, "deferred": deferred, "deferred %": round(100 * deferred / max(rows, 1), 1)})
pd.DataFrame(dense).set_index("task")

# %% [markdown]
# ## 2. Held-out entities: how many per update?
#
# The stopping signal of update e is the loss on held-out rows of the whole last
# episode, i.e. the rows newly labelled at T_e.

# %%
sizes = []
for name, ctx in contexts.items():
    df = ctx.wrapper.full_table.df
    times, ents = df[ctx.task.time_col], df[ctx.task.entity_col].to_numpy()
    salt = ctx.protocol.holdout_salt
    for frac in FRACS:
        held = entity_holdout_mask(ents, frac, salt)
        for ep in ctx.plan.episodes:
            new = trainable_mask(times, ep.update_time, ctx.task.timedelta, after=ep.new_after)
            sel = new & held
            sizes.append({"task": name, "frac": frac, "update": ep.index,
                          "selection rows": int(sel.sum()), "selection entities": int(len(np.unique(ents[sel])))})
sizes = pd.DataFrame(sizes)
sizes.groupby(["task", "frac"])[["selection rows", "selection entities"]].agg(["min", "median"])

# %% [markdown]
# ## 3. How noisy is the stopping signal?
#
# For each task, train one probe model (scratch, all history up to the middle update,
# a fixed `NOISE_STEPS` steps). Score the selection rows of each update, and bootstrap
# over **entities** (rows of one driver move together) to get the standard error of
# loss and metric at each fraction. A stopping rule cannot resolve improvements
# smaller than this.

# %%
import lightning as L
from redelex.loaders import MixtureLoader, Source
from redelex.nn.train import LightningEntityTaskWrapper
from redelex.nn.train.utils import get_loss
from redelex.transforms import AttachTargetTransform
from experiments.continuous_learning.chain_runner import new_model
from experiments.continuous_learning.utils import get_table_input


def probe_predictions(ctx, steps):
    torch.manual_seed(0)
    mid = ctx.plan.episodes[len(ctx.plan.episodes) // 2]
    table = ctx.wrapper.trainable(mid.update_time)
    inp = get_table_input(table, ctx.task)
    loader = MixtureLoader(ctx.data, ctx.task.entity_table, {"all": Source(inp.nodes[1], inp.time, inp.target, 1.0)},
                           [32, 16], 128, steps * 128, seed=0, transform_factory=AttachTargetTransform)
    model = new_model(ctx, ChainConfig(DATASET, ctx.task_name, "scratch_all", 0, OUT)).to(ctx.device)
    wrapper = LightningEntityTaskWrapper(model, torch.optim.Adam(model.parameters(), lr=1e-3), ctx.task, modes=("val",))
    for batch in loader:
        batch = batch.to(ctx.device)
        pred, target = wrapper(batch)
        loss = wrapper.loss_fn(pred.float(), target)
        wrapper.optimizer.zero_grad()
        loss.backward()
        wrapper.optimizer.step()
    full = ctx.wrapper.full_table
    pred = predict_table(model, ctx.data, ctx.task, full, [32, 16], 512, ctx.device)
    return pd.Series(pred, index=full.df["__row__"].to_numpy())


def cluster_bootstrap(entities, target, pred, task_type, loss_fn, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(entities)
    groups = {e: np.flatnonzero(entities == e) for e in uniq}
    losses, metrics = [], []
    for _ in range(n_boot):
        pick = np.concatenate([groups[e] for e in rng.choice(uniq, size=len(uniq), replace=True)])
        y, p = torch.as_tensor(target[pick], dtype=torch.float32), torch.as_tensor(pred[pick], dtype=torch.float32)
        losses.append(float(loss_fn(p, y)))
        metrics.append(metric_value(task_type, target[pick], pred[pick]))
    return float(np.std(losses)), float(np.nanstd(metrics))


noise = []
for name, ctx in contexts.items():
    preds = probe_predictions(ctx, NOISE_STEPS if SCALE == "full" else min(NOISE_STEPS, 60))
    df = ctx.wrapper.full_table.df
    times, ents = df[ctx.task.time_col], df[ctx.task.entity_col].to_numpy()
    target = df[ctx.task.target_col].to_numpy().astype(float)
    pred_all = preds.reindex(df["__row__"]).to_numpy()
    loss_fn = get_loss(ctx.task.task_type)
    for frac in FRACS:
        held = entity_holdout_mask(ents, frac, ctx.protocol.holdout_salt)
        for ep in ctx.plan.episodes:
            sel = trainable_mask(times, ep.update_time, ctx.task.timedelta, after=ep.new_after) & held
            if sel.sum() < 2 or len(np.unique(ents[sel])) < 2:
                continue
            loss_se, metric_se = cluster_bootstrap(ents[sel], target[sel], pred_all[sel], ctx.task.task_type, loss_fn)
            noise.append({"task": name, "frac": frac, "update": ep.index, "loss SE": loss_se, "metric SE": metric_se})
noise = pd.DataFrame(noise)
noise.groupby(["task", "frac"])[["loss SE", "metric SE"]].median().round(4)

# %% [markdown]
# ## 4. Leak audit: every regime, end to end
#
# Two-update mini chains per regime on the first task, with a hook recording the
# entity of every training seed.

# %%
import shutil

audit_root = out / "02_audit"
shutil.rmtree(audit_root, ignore_errors=True)  # a resumed chain would skip training and audit nothing
audit_task = "driver-dnf" if "driver-dnf" in TASKS else TASKS[0]
actx = contexts[audit_task]
held_entities = set(actx.wrapper.full_table.df.loc[actx.wrapper.full_table.df[HOLDOUT_COL], actx.task.entity_col])
audit = []
for mode in AUDIT_MODES:
    seen = []
    cfg = ChainConfig(DATASET, audit_task, mode, 0, str(audit_root), variant="audit", device=DEVICE,
                      max_steps=20, check_every=10, patiences=(1, 2), anytime=False, max_updates=2,
                      extra={"audit_seen": seen})
    run_chain(cfg, ctx=actx)
    leaked = held_entities & set(seen)
    audit.append({"mode": mode, "seeds seen": len(seen), "distinct entities": len(set(seen)),
                  "held-out entities seen": len(leaked)})
audit = pd.DataFrame(audit).set_index("mode")
audit

# %% [markdown]
# ## 5. Choose the holdout fraction per task
#
# Rule: the smallest fraction whose selection set has at least
# `MIN_SELECTION_ENTITIES` entities at every update. If none qualifies, take the
# largest, and flag the task for a stratified holdout.

# %%
choice = {}
for name in TASKS:
    s = sizes[sizes.task == name]
    ok = [f for f in FRACS if s[s.frac == f]["selection entities"].min() >= MIN_SELECTION_ENTITIES]
    choice[name] = {"holdout_frac": min(ok) if ok else max(FRACS), "meets_minimum": bool(ok),
                    "min_entities_at_choice": int(s[s.frac == (min(ok) if ok else max(FRACS))]["selection entities"].min())}
choice = pd.DataFrame(choice).T
choice

# %% [markdown]
# ## Gate

# %%
for name, ctx in contexts.items():
    for ep in ctx.plan.episodes:
        tr = ctx.wrapper.trainable(ep.update_time).df
        assert (tr[ctx.task.time_col] <= ep.update_time - ctx.task.timedelta).all(), (name, ep.index)
        assert not tr[HOLDOUT_COL].any(), (name, ep.index)
assert (audit["held-out entities seen"] == 0).all(), audit
assert (audit["seeds seen"] > 0).all()
(out / f"02_holdout_choice_{DATASET}.json").write_text(
    json.dumps({"choice": choice.reset_index().rename(columns={"index": "task"}).to_dict(orient="records"),
                "audit": audit.reset_index().to_dict(orient="records"),
                "noise_median": noise.groupby(["task", "frac"])[["loss SE", "metric SE"]].median().reset_index().to_dict(orient="records")},
               indent=2, default=str)
)
print("GATE 02 PASSED")
