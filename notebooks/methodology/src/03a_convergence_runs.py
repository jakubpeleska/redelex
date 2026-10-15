# %% [markdown]
# # 03a — Convergence probe runs (rel-f1)
#
# **Purpose.** Collect the raw material to choose the stopping rule. Each probe
# update logs:
# - its held-out loss and metric every `CHECK_RES` steps;
# - its next-window score every `ANYTIME_RES` steps;
# - all of this for far longer than any candidate rule would wait.
#
# Notebook 03b then replays every rule (check interval N, patience P, loss or metric)
# on these histories offline.
#
# The probes use the next window **only for calibration**: choosing a rule by
# its regret against the best next-window score. The rule itself never looks at
# the next window.
#
# Shard the work across GPUs with the `TASKS`, `MODES` and `SEEDS` parameters; every
# shard writes to the same `OUT/03_probe` tree and resumes where it stopped.
#
# **Gate.** Every planned chain is complete, and every update logged its check
# history and its anytime curve.

# %% [parameters]
DATASET = "rel-f1"
TASKS = ["driver-position", "driver-dnf"]
MODES = ["scratch_all", "cumulative", "incremental"]
SEEDS = [0]
PLATEAU_MODES = ["cumulative"]  # also run these with a plateau LR schedule
DEVICE = "auto"
SCALE = "smoke"
OUT = "notebooks/methodology/out"
THREADS = 8
HOLDOUT_FRAC = 0.1
CHECK_RES = 10
ANYTIME_RES = 10
PROBE_PATIENCE_STEPS = 1500
CAP = 10_000
MAX_UPDATES = None  # None = every update

# %%
import json
import os
import sys
import time
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
    TASKS, MODES, SEEDS, PLATEAU_MODES = TASKS[:1], MODES[:2], SEEDS[:1], PLATEAU_MODES[:1]
    CAP, PROBE_PATIENCE_STEPS, MAX_UPDATES = 120, 60, 2
probe_root = Path(OUT) / "03_probe"


def probe_cfg(task, mode, seed, plateau=False):
    return ChainConfig(
        DATASET, task, mode, seed, str(probe_root),
        variant="probe_plateau" if plateau else "probe",
        device=DEVICE, torch_threads=THREADS, holdout_frac=HOLDOUT_FRAC,
        max_steps=CAP, check_every=CHECK_RES, patiences=(max(1, PROBE_PATIENCE_STEPS // CHECK_RES),),
        anytime_every=ANYTIME_RES, max_updates=MAX_UPDATES,
        plateau_patience=5 if plateau else None,
    )

# %% [markdown]
# ## Run the probes

# %%
runs = []
for task in TASKS:
    ctx = build_context(probe_cfg(task, MODES[0], SEEDS[0]))
    plan = [(m, s, False) for m in MODES for s in SEEDS] + [(m, s, True) for m in PLATEAU_MODES for s in SEEDS]
    for mode, seed, plateau in plan:
        cfg = probe_cfg(task, mode, seed, plateau)
        t0 = time.time()
        summary = run_chain(cfg, ctx=ctx)
        runs.append({**summary, "task": task, "mode": mode, "plateau": plateau, "seconds": round(time.time() - t0)})
        print(json.dumps(runs[-1]))
runs = pd.DataFrame(runs)
runs[["task", "mode", "plateau", "seed", "status", "episodes_done", "episodes_total", "seconds"]]

# %% [markdown]
# ## Gate

# %%
assert (runs["status"] == "complete").all(), runs
for _, r in runs.iterrows():
    cfg = probe_cfg(r.task, r["mode"], r.seed, r.plateau)
    for ep_dir in sorted(cfg.seed_dir().glob("ep*")):
        rec = json.loads((ep_dir / "episode.json").read_text())
        assert rec["convergence"]["history"], ep_dir
        assert len(rec["anytime"]) > 1, ep_dir
print("GATE 03a PASSED")
