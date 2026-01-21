#!/bin/bash
#SBATCH --job-name=universal_encoder_baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-16

declare -a dataset_pairs=(
    'rel-f1 driver-position'
    'rel-f1 driver-dnf'
    'rel-f1 driver-top3'
    'rel-stack user-engagement'
    'rel-stack post-votes'
    'rel-stack user-badge'
    'rel-stack user-post-comment'
    'rel-stack post-post-related'
    'rel-trial study-outcome'
    'rel-trial study-adverse'
    'rel-trial site-success'
    'rel-trial condition-sponsor-run'
    'rel-trial site-sponsor-run'
    'rel-avito ad-ctr'
    'rel-avito user-visits'
    'rel-avito user-clicks'
    'rel-avito user-ad-visit'
)

VENV_PATH="/home/pelesjak/git/ctu-relational-py/.venv"

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


python -u experiments/universal_encoder/universal_encoder_baseline.py --dataset=${dataset} --task=${task}\
  --ray_address="local" --ray_storage=${log_dir} \
  --run_name=${EXPERIMENT_ID}_${dataset}_${task} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_TASK} \
  --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log"



