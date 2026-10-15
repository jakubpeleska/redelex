#!/bin/bash
#SBATCH --job-name=cl2_setup
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rci/setup_env_%j.log
#
# Build or refresh the project environment on RCI after a code sync. A batch job,
# because syncing the venv downloads wheels, which does not belong on a login node.
#
#   mkdir -p logs/rci && sbatch slurm/rci/setup_env.sh
#
# Not a GPU job: `uv sync` needs no GPU, and cpufast queues faster.

set -euo pipefail
cd "${CL_REPO:-$HOME/git/claude-redelex}"
mkdir -p logs/rci
export PATH="$HOME/.local/bin:$PATH"

echo "=== node: $(hostname)  commit: $(git rev-parse --short HEAD) ==="
uv --version

# cu128, never the default `cpu` group: a plain `uv sync` silently installs the
# CPU build of torch and every later run is several times slower with no error.
echo "=== uv sync (cu128) ==="
uv sync --no-group cpu --group cu128

echo "=== interpreter ==="
.venv/bin/python -c "
from importlib.metadata import version
import torch
print(f'  torch {torch.__version__}  cuda compiled: {torch.version.cuda}')
for pkg in ('relbench', 'torch_geometric', 'lightning', 'papermill'):
    print(f'  {pkg} {version(pkg)}')
"

echo "=== import the rerun modules ==="
.venv/bin/python -c "
from experiments.continuous_learning.modes import MAIN_ROSTER, ABLATION_ROSTER
from experiments.continuous_learning.chain_runner import run_chain
print('  main roster:', ', '.join(m.name for m in MAIN_ROSTER))
print('  ablations:  ', ', '.join(m.name for m in ABLATION_ROSTER))
"

echo "=== unit tests ==="
.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3

echo "=== DONE ==="
