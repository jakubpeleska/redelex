#!/bin/bash
#SBATCH --job-name=universal_encoder_baseline
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

declare -a dataset_pairs=(
    'rel-amazon user-churn'
    'rel-amazon user-ltv'
    'rel-amazon item-churn'
    'rel-amazon item-ltv'
)

VENV_PATH="./venv"

# Activate the local virtual environment
source "${VENV_PATH}/bin/activate"

EXPERIMENT_NAME="universal_encoder_baseline"

echo $SLURM_ARRAY_JOB_ID

# START_TIME=$(sacct -j ${SLURM_JOB_ID} --format=Start -n | head -n 1)

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=5

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

# ******************************************

# Run experiment on different datasets

pair=${dataset_pairs[$SLURM_ARRAY_TASK_ID]}
read -a strarr <<< "$pair"  # uses default whitespace IFS
dataset=${strarr[0]}
task=${strarr[1]}

log_dir=${experiment_dir}/${dataset}_${task}
mkdir -p $log_dir


python -u experiments/pretrain/universal_encoder_baseline.py --dataset=${dataset} --task=${task} \
  --ray_address="local" --ray_storage=${log_dir} \
  --run_name=${EXPERIMENT_ID}_${dataset}_${task} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_GPU} --num_gpus=1 \
  --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log"



