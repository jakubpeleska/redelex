#!/bin/bash
#SBATCH --partition=cpulong
#SBATCH --job-name=exp1_3_pretrain_ablation
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=80G
#SBATCH --time=72:00:00
#SBATCH --array=0-4

declare -a leave_out_datasets=(
    'rel-amazon'
    'rel-avito'
    'rel-f1'
    'rel-stack'
    'rel-trial'
)

VENV_PATH="/home/pelesjak/git/ctu-relational-py/.venv"

# Activate the local virtual environment
source "${VENV_PATH}/bin/activate"

EXPERIMENT_NAME="exp1_3_pretrain_ablation"

echo $SLURM_ARRAY_JOB_ID

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=1

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

# Run experiment on different datasets
dataset=${leave_out_datasets[$SLURM_ARRAY_TASK_ID]}

log_dir=${experiment_dir}/leave_out_${dataset}
mkdir -p $log_dir

python -u experiments/universal_encoder/universal_encoder_pretrained.py \
  --leave_out_dataset=${dataset} \
  --gnn_type="heterogeneous" \
  --no_stype_emb --no_name_emb --no_stats_emb \
  --ray_address="local" --ray_storage=${log_dir} \
  --run_name=${EXPERIMENT_ID}_leave_out_${dataset} --mlflow_uri=${MLFLOW_TRACKING_URI} \
  --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} --num_cpus=${SLURM_CPUS_PER_TASK} --num_gpus=0 \
  --num_samples=${NUM_SAMPLES} \
  --model_save_path="${experiment_dir}/${dataset}_pretrained.pt" &> "${log_dir}/run.log"
