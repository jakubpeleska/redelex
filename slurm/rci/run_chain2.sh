#!/bin/bash
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rci/chains/%x-%j.out
#
# Run (or resume) chains of the rerun, one after another, until all are complete
# or the job's deadline is near; chain_runner stops before SLURM_JOB_END_TIME and
# the next job in the lane resumes.
#
#   sbatch slurm/rci/run_chain2.sh <chains.txt> [extra chain_runner flags...]
#
# Each line of chains.txt: dataset task mode seed variant
set -uo pipefail
REPO="${CL_REPO:-$HOME/git/claude-redelex}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
LIST=$1
shift
OUTDIR="${CL_OUT:-$REPO/logs/rci/cl2}"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit $(git rev-parse --short HEAD)"
while read -r DATASET TASK MODE SEED VARIANT; do
    [ -z "${DATASET:-}" ] && continue
    echo "=== $DATASET $TASK $MODE seed $SEED ($VARIANT)"
    .venv/bin/python -m experiments.continuous_learning.chain_runner \
        --dataset "$DATASET" --task "$TASK" --mode "$MODE" --seed "$SEED" --variant "$VARIANT" \
        --out_dir "$OUTDIR" --device cuda --torch_threads "${SLURM_CPUS_PER_TASK:-8}" "$@"
    rc=$?
    if [ $rc -eq 3 ]; then echo "deadline reached; the next job in this lane resumes"; exit 0; fi
    if [ $rc -ne 0 ]; then echo "FAILED rc=$rc: $DATASET $TASK $MODE $SEED"; fi
done < "$LIST"
