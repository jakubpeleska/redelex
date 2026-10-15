#!/bin/bash
#SBATCH --job-name=cl2-notebook
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rci/notebooks/%x-%j.out
#
# Execute methodology notebooks on an RCI GPU node.
#
#   mkdir -p logs/rci/notebooks
#   sbatch slurm/rci/run_notebook.sh 03 a100-full SCALE full
#
# Arguments: notebook prefix, run tag, then NAME VALUE pairs passed as papermill
# parameters. DEVICE and THREADS are set from the allocation.
set -euo pipefail

REPO="${CL_REPO:-$HOME/git/claude-redelex}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1

PREFIX=$1
TAG=$2
shift 2
PARAMS=()
while [ $# -ge 2 ]; do
    PARAMS+=(-p "$1" "$2")
    shift 2
done

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit $(git rev-parse --short HEAD)"
.venv/bin/python scripts/run_methodology_notebooks.py "$PREFIX" --tag "$TAG" \
    -p DEVICE cuda -p THREADS "${SLURM_CPUS_PER_TASK:-8}" "${PARAMS[@]}"
