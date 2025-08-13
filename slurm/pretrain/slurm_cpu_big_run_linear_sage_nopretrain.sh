#!/bin/bash
#SBATCH --job-name=linear_sage_nopretrain
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

datasets=('rel-f1' 'rel-avito' 'rel-trial' 'rel-stack')

conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

EXPERIMENT_NAME="linear_sage_nopretrain"

echo $SLURM_ARRAY_JOB_ID

# START_TIME=$(sacct -j ${SLURM_JOB_ID} --format=Start -n | head -n 1)

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=1

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

# ******************************************

# Run experiment on different datasets

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

log_dir=${experiment_dir}/${dataset}
mkdir -p $log_dir


python -u experiments/pretrain/dbgnn_nopretrain.py --dataset=${dataset} --rgnn_model="sage" \
  --tabular_model="linear" --ray_address="local" --ray_storage=${log_dir} \
  --run_name=${EXPERIMENT_ID}_${dataset} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_TASK} \
  --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log"



