#!/bin/bash
#SBATCH --job-name=cl2-bench
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rci/bench/%x-%j.out
#
# Per-dataset step-cost benchmark for the rerun budget (notebook 08).
#   sbatch --mem=80G slurm/rci/run_benchmark.sh rel-ratebeer beer-churn user-churn
set -euo pipefail
REPO="${CL_REPO:-$HOME/git/claude-redelex}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
DATASET=$1
shift
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit $(git rev-parse --short HEAD)"
for TASK in "$@"; do
    echo "=== $DATASET / $TASK"
    .venv/bin/python scripts/benchmark_step_cost.py --dataset "$DATASET" --task "$TASK" --device cuda \
        --threads "${SLURM_CPUS_PER_TASK:-8}" || echo "FAILED $DATASET / $TASK"
done
