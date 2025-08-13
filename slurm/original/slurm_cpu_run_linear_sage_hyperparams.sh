#!/bin/bash
#SBATCH --job-name=linear_sage_hyperparams
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0

declare -a dataset_pairs=(
    
    # 'rel-f1 driver-position'
    # 'rel-f1 driver-dnf'
    # 'rel-f1 driver-top3'

    # 'ctu-adventureworks adventureworks-original'
    # 'ctu-adventureworks adventureworks-temporal'
    # 'ctu-atherosclerosis atherosclerosis-original'
    # 'ctu-basketballmen basketballmen-original'
    # 'ctu-basketballwomen basketballwomen-original'
    # 'ctu-biodegradability biodegradability-original'
    # 'ctu-bupa bupa-original'
    # 'ctu-carcinogenesis carcinogenesis-original'
    # 'ctu-cde cde-original'
    'ctu-chess chess-original'
    # 'ctu-classicmodels classicmodels-original'
    # 'ctu-classicmodels classicmodels-temporal'
    # 'ctu-cora cora-original'
    # 'ctu-countries countries-original'
    # 'ctu-craftbeer craftbeer-original'
    # 'ctu-credit credit-original'
    # 'ctu-dallas dallas-original'
    # 'ctu-dallas dallas-temporal'
    # 'ctu-dcg dcg-original'
    # 'ctu-diabetes diabetes-original'
    # 'ctu-dunur dunur-original'
    # 'ctu-elti elti-original'
    # 'ctu-ergastf1 ergastf1-original'
    # 'ctu-financial financial-original'
    # 'ctu-financial financial-temporal'
    # 'ctu-fnhk fnhk-original'
    # 'ctu-fnhk fnhk-temporal'
    # 'ctu-ftp ftp-original'
    # 'ctu-ftp ftp-temporal'
    # 'ctu-geneea geneea-original'
    # 'ctu-geneea geneea-temporal'
    # 'ctu-genes genes-original'
    # 'ctu-hepatitis hepatitis-original'
    # 'ctu-hockey hockey-original'
    # 'ctu-lahman lahman-original'
    # 'ctu-lahman lahman-temporal'
    # 'ctu-mesh mesh-original'
    # 'ctu-mondial mondial-original'
    # 'ctu-mooney mooney-original'
    # 'ctu-movielens movielens-original'
    # 'ctu-musklarge musklarge-original'
    # 'ctu-musksmall musksmall-original'
    # 'ctu-mutagenesis mutagenesis-original'
    # 'ctu-ncaa ncaa-original'
    # 'ctu-northwind northwind-original'
    # 'ctu-northwind northwind-temporal'
    # 'ctu-pima pima-original'
    # 'ctu-premiereleague premiereleague-original'
    # 'ctu-premiereleague premiereleague-temporal'
    # 'ctu-restbase restbase-original'
    # 'ctu-sakila sakila-original'
    # 'ctu-sakila sakila-temporal'
    # 'ctu-samegen samegen-original'
    # 'ctu-satellite satellite-original'
    # 'ctu-sfscores sfscores-original'
    # 'ctu-sfscores sfscores-temporal'
    # 'ctu-shakespeare shakespeare-original'
    # 'ctu-stats stats-original'
    # 'ctu-stats stats-temporal'
    # 'ctu-studentloan studentloan-original'
    # 'ctu-thrombosis thrombosis-original'
    # 'ctu-toxicology toxicology-original'
    # 'ctu-tpcc tpcc-original'
    # 'ctu-triazine triazine-original'
    # 'ctu-uwcse uwcse-original'
    # 'ctu-voc voc-original'
    # 'ctu-voc voc-temporal'
    # 'ctu-webkp webkp-original'
    # 'ctu-world world-original'
)

conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

EXPERIMENT_NAME="linear_sage_hyperparams"

echo $SLURM_ARRAY_JOB_ID

# START_TIME=$(sacct -j ${SLURM_JOB_ID} --format=Start -n | head -n 1)

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=3

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

ray_address="local"
# ******************************************

# Run experiment on different datasets

pair=${dataset_pairs[$SLURM_ARRAY_TASK_ID]}
read -a strarr <<< "$pair"  # uses default whitespace IFS
dataset=${strarr[0]}
task=${strarr[1]}

log_dir=${experiment_dir}/${dataset}_${task}
mkdir -p $log_dir

python -u experiments/original/dbgnn_hyperparams.py --ray_address=${ray_address} \
  --ray_storage=${log_dir} --run_name=${EXPERIMENT_ID}_${dataset}_${task} --dataset=${dataset} \
  --mlflow_uri=${MLFLOW_TRACKING_URI} --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} \
  --aim_repo=logs/.aim --task=${task} --model="sage" --num_samples=${NUM_SAMPLES} \
  --num_cpus=${SLURM_CPUS_PER_TASK} --row_encoder="linear" &> "${log_dir}/run.log"



