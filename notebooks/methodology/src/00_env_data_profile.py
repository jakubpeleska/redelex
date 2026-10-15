# %% [markdown]
# # 00 — Environment and data profile (rel-f1)
#
# **Purpose.** Before anything is trained, confirm that the data loads on the uniform
# anchor grid the rerun uses, and record what each update and window contains:
# rows, entities, held-out entities. Also measure seconds per training step, the
# unit every later cost estimate is built on.
#
# **Gate.**
# 1. Every task loads, and its plan has update times and windows.
# 2. Tasks with at least 5 updates are main-grid tasks; fewer go to the appendix.
# 3. Every window with rows has at least one held-out entity.
# 4. The step-time benchmark is finite and saved.

# %% [parameters]
DATASET = "rel-f1"
TASKS = ["driver-position", "driver-dnf", "driver-top3"]
DEVICE = "auto"
SCALE = "smoke"  # "smoke" (CPU, minutes) or "full"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.1

# %%
import json
import os
import platform
import sys
import time
from pathlib import Path

root = Path.cwd()
while not (root / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(THREADS)
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)

from experiments.continuous_learning.chain_runner import ChainConfig, build_context
from redelex.continual.anchors import native_anchor_timestamps

# %% [markdown]
# ## Environment

# %%
from importlib.metadata import version

env = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "relbench": version("relbench"),
    "torch_geometric": version("torch_geometric"),
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
    "threads": torch.get_num_threads(),
    "host": platform.node(),
}
pd.Series(env)

# %% [markdown]
# ## Load every task on the uniform grid
#
# Anchors are placed every Δw through the RelBench validation timestamp. Updates are
# placed every native ΔI (the RelBench test − validation span), starting from the
# first update that has at least `min_initial_rows` trainable rows.

# %%
contexts = {}
load_seconds = {}
for task_name in TASKS:
    cfg = ChainConfig(DATASET, task_name, "scratch_all", 0, OUT, device=DEVICE, holdout_frac=HOLDOUT_FRAC)
    t0 = time.perf_counter()
    contexts[task_name] = build_context(cfg)
    load_seconds[task_name] = time.perf_counter() - t0

summary = []
for task_name, ctx in contexts.items():
    p, plan = ctx.protocol, ctx.plan
    summary.append(
        {
            "task": task_name,
            "type": ctx.task.task_type.value,
            "Δw (days)": p.delta_w.days,
            "ΔI (days)": p.delta_i.days,
            "anchors": ctx.rows["time"].nunique(),
            "rows": len(ctx.rows),
            "entities": ctx.rows["entity"].nunique(),
            "held-out entities": ctx.rows.loc[ctx.rows.holdout, "entity"].nunique(),
            "updates": plan.num_updates,
            "first update": plan.update_times[0].date(),
            "last update": plan.update_times[-1].date(),
            "load s": round(load_seconds[task_name], 1),
        }
    )
summary = pd.DataFrame(summary).set_index("task")
summary

# %% [markdown]
# ## What each window holds
#
# Window 1 is the initial history; the model updated at T_e is scored on window e+1.
# A window's held-out rows decide when the update that labels them stops.

# %%
def window_profile(ctx):
    r = ctx.rows[ctx.rows.window >= 1]
    g = r.groupby("window")
    prof = pd.DataFrame(
        {
            "start": g["time"].min().dt.date,
            "end": g["time"].max().dt.date,
            "anchors": g["time"].nunique(),
            "rows": g.size(),
            "entities": g["entity"].nunique(),
            "held-out rows": g["holdout"].sum(),
            "held-out entities": r[r.holdout].groupby("window")["entity"].nunique(),
        }
    ).fillna(0)
    if ctx.task.task_type.value == "binary_classification":
        prof["positive rate"] = g["target"].mean().round(3)
    else:
        prof["target mean"] = g["target"].mean().round(2)
    return prof


profiles = {t: window_profile(ctx) for t, ctx in contexts.items()}
for t, prof in profiles.items():
    print(f"\n{t}")
    display(prof)

# %% [markdown]
# ## Native RelBench tables vs. the uniform grid
#
# RelBench's native tables use only `num_eval_timestamps` anchors from the validation
# and from the test timestamp. Its test anchors start at `test_timestamp`, which is
# generally not a whole number of Δw after the validation timestamp, so they sit off
# the uniform grid. Counted here over anchors that produce at least one row.

# %%
gap_rows = []
for task_name, ctx in contexts.items():
    task, db = ctx.task, ctx.db
    native_with_rows, native_test = set(), set()
    for split in ("train", "val", "test"):
        tab = task.get_table(split, mask_input_cols=False).df
        ts = set(pd.to_datetime(tab[task.time_col].unique()))
        native_with_rows |= ts
        if split == "test":
            native_test = ts
    grid = set(pd.to_datetime(ctx.rows["time"].unique()))
    gap_rows.append(
        {
            "task": task_name,
            "num_eval_timestamps": task.num_eval_timestamps,
            "native anchors with rows": len(native_with_rows),
            "grid anchors with rows": len(grid),
            "on both": len(grid & native_with_rows),
            "grid only (gaps filled)": len(grid - native_with_rows),
            "native test anchors off the grid": len(native_test - grid),
        }
    )
pd.DataFrame(gap_rows).set_index("task")

# %% [markdown]
# ## Step-time benchmark
#
# One update's training loop on the device this notebook runs on: driver-dnf at its
# last update (largest training set), cumulative mixing over all labelled history.

# %%
from redelex.loaders import MixtureLoader, Source
from redelex.nn.train import LightningEntityTaskWrapper
from redelex.transforms import AttachTargetTransform
from experiments.continuous_learning.chain_runner import new_model, training_sources
from experiments.continuous_learning.modes import get_mode
from experiments.continuous_learning.utils import get_table_input

bench_task = "driver-dnf" if "driver-dnf" in contexts else TASKS[0]
ctx = contexts[bench_task]
cfg = ChainConfig(DATASET, bench_task, "cumulative", 0, OUT, device=DEVICE)
device = ctx.device
ep = ctx.plan.episodes[-1]
sources = training_sources(ctx, get_mode("cumulative"), ep)
srcs = {}
for name, (table, share) in sources.items():
    inp = get_table_input(table, ctx.task)
    srcs[name] = Source(inp.nodes[1], inp.time, inp.target, share)
steps = 30 if SCALE == "smoke" else 200
loader = MixtureLoader(ctx.data, ctx.task.entity_table, srcs, [32, 16], 128, steps * 128, seed=0,
                       transform_factory=AttachTargetTransform)
model = new_model(ctx, cfg).to(device)
wrapper = LightningEntityTaskWrapper(model, torch.optim.Adam(model.parameters(), lr=1e-3), ctx.task, modes=("val",))
opt = wrapper.optimizer
times, data_times = [], []
it = iter(loader)
for i in range(steps):
    t0 = time.perf_counter()
    batch = next(it).to(device)
    t1 = time.perf_counter()
    pred, target = wrapper(batch)
    loss = wrapper.loss_fn(pred.float(), target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    if i >= 3:  # skip warm-up
        data_times.append(t1 - t0)
        times.append(t2 - t1)

bench = {
    "task": bench_task,
    "device": env["device_name"],
    "threads": torch.get_num_threads(),
    "train_rows": int(sum(len(s.nodes) for s in srcs.values())),
    "steps_timed": len(times),
    "compute_s_per_step": float(np.median(times)),
    "data_s_per_step": float(np.median(data_times)),
    "s_per_step": float(np.median(np.add(times, data_times))),
}
bench["samples_per_s"] = 128 / bench["s_per_step"]
pd.Series(bench)

# %%
record = {
    "env": env,
    "summary": summary.reset_index().astype(str).to_dict(orient="records"),
    "windows": {t: p.reset_index().astype(str).to_dict(orient="records") for t, p in profiles.items()},
    "benchmark": bench,
}
path = out / f"00_profile_{DATASET}_{'gpu' if device.type == 'cuda' else 'cpu'}.json"
path.write_text(json.dumps(record, indent=2))
print("saved", path)

# %% [markdown]
# ## Gate

# %%
main_tasks = [t for t in TASKS if contexts[t].plan.num_updates >= 5]
appendix_tasks = [t for t in TASKS if t not in main_tasks]
print("main-grid tasks:", main_tasks)
print("appendix tasks (< 5 updates):", appendix_tasks)

for t, ctx in contexts.items():
    assert ctx.plan.num_updates >= 1 and ctx.plan.num_windows == ctx.plan.num_updates + 1, t
    prof = profiles[t]
    with_rows = prof[prof["rows"] > 0]
    assert (with_rows["held-out entities"] >= 1).all(), f"{t}: a window has rows but no held-out entity"
assert np.isfinite(bench["s_per_step"]) and bench["s_per_step"] > 0
assert path.exists()
print("GATE 00 PASSED")
