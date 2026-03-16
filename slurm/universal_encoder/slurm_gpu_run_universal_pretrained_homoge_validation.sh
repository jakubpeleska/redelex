#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=universal_pretrained_homoge_validation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-0

declare -a dataset_tasks=(
    # 'rel-f1 driver-dnf'
    # 'rel-f1 driver-top3'
    # 'rel-f1 driver-position'
    # 'rel-stack user-engagement'
    # 'rel-stack user-badge'
    # 'rel-stack post-votes'
    # 'rel-trial study-outcome'
    # 'rel-trial study-adverse'
    # 'rel-trial site-success'
    # 'rel-amazon user-churn'
    # 'rel-amazon user-ltv'
    # 'rel-amazon item-churn'
    # 'rel-amazon item-ltv'
    # 'rel-avito ad-ctr'
    # 'rel-avito user-visits'
    'rel-avito user-clicks'
)

VENV_PATH="/home/pelesjak/git/ctu-relational-py/.venv"

# Activate the local virtual environment
source "${VENV_PATH}/bin/activate"

EXPERIMENT_NAME="universal_pretrained_homoge_validation"

echo $SLURM_ARRAY_JOB_ID

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=1
MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Read tuple from array
read -r eval_dataset task <<< "${dataset_tasks[$SLURM_ARRAY_TASK_ID]}"
pretrain_dataset="${eval_dataset}"

CHECKPOINTS=()
for tablayers in 4 2; do
    CHECKPOINT_NAME="${pretrain_dataset}_tablayers${tablayers}_homoge.pt"
    if [ -f "experiments/universal_encoder/pretrained_models/${CHECKPOINT_NAME}" ]; then
        CHECKPOINTS+=("experiments/universal_encoder/pretrained_models/${CHECKPOINT_NAME}")
    fi
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found, skipping."
    exit 0
fi

echo "Evaluating ${#CHECKPOINTS[@]} checkpoints on ${eval_dataset} task ${task}"

# Create log directory
run_name="${EXPERIMENT_ID}_${eval_dataset}_${task}"
log_dir="logs/${EXPERIMENT_ID}/${eval_dataset}_${task}"
mkdir -p "$log_dir"

python -u experiments/universal_encoder/universal_encoder_supervised.py \
  --dataset=${eval_dataset} \
  --task=${task} \
  --pretrained_checkpoint "${CHECKPOINTS[@]}" \
  --pretrained_row_encoder \
  --ray_address="local" --ray_storage=${log_dir} \
  --run_name=${run_name} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_GPU} --num_gpus=1 \
  --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log"
