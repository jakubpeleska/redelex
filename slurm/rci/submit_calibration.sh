#!/bin/bash
# Submit the convergence-calibration probes (notebook 03a) as one job per
# (task, regime) shard. Each shard writes to notebooks/methodology/out/03_probe
# and resumes where a previous attempt stopped; resubmit a shard that timed out.
#
#   bash slurm/rci/submit_calibration.sh            # submit every shard
#   DRY=1 bash slurm/rci/submit_calibration.sh      # print the sbatch lines only
#
# Then, once all shards pass, notebook 03b (CPU-only) replays the stopping rules.
set -euo pipefail
cd "${CL_REPO:-$HOME/git/claude-redelex}"
mkdir -p logs/rci/notebooks

TASKS=(driver-position driver-dnf)
MODES=(scratch_all cumulative incremental)
SEEDS="[0,1]"
HOLDOUT=0.2

for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    plateau="[]"
    [ "$mode" = cumulative ] && plateau="['cumulative']"
    cmd=(sbatch --job-name="cl2-cal-${task}-${mode}" slurm/rci/run_notebook.sh 03a "a100-${task}-${mode}"
         SCALE full TASKS "['${task}']" MODES "['${mode}']" SEEDS "${SEEDS}"
         PLATEAU_MODES "${plateau}" HOLDOUT_FRAC "${HOLDOUT}")
    if [ -n "${DRY:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
  done
done
