#!/bin/bash
#SBATCH --job-name=cl_ft_upsample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-11

declare -a dataset_pairs=(
    'rel-f1 driver-position'
    'rel-f1 driver-dnf'
    'rel-f1 driver-top3'
    'rel-stack user-engagement'
    'rel-stack post-votes'
    'rel-stack user-badge'
    'rel-trial study-outcome'
    'rel-trial study-adverse'
    'rel-trial site-success'
    'rel-avito ad-ctr'
    'rel-avito user-visits'
    'rel-avito user-clicks'
)

VENV_PATH="/home/pelesjak/git/ctu-relational-py/.venv"

# Activate the local virtual environment
source "${VENV_PATH}/bin/activate"

LEARNING_MODE="ft_upsample"
EXPERIMENT_NAME="cl_${LEARNING_MODE}"

echo $SLURM_ARRAY_JOB_ID

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=5

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir


# Run experiment on different datasets

pair=${dataset_pairs[$SLURM_ARRAY_TASK_ID]}
read -a strarr <<< "$pair"  # uses default whitespace IFS
dataset=${strarr[0]}
task=${strarr[1]}

log_dir=${experiment_dir}/${dataset}_${task}
mkdir -p $log_dir

python -u experiments/continuous_learning/continuous_learning.py \
  --dataset=${dataset} --task=${task} --learning_mode=${LEARNING_MODE} \
  --ray_address="local" --ray_storage=${log_dir} --model_save_dir=${log_dir}/models \
  --run_name=${EXPERIMENT_ID}_${dataset}_${task} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_GPU} --num_gpus=1 \
  --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log"
