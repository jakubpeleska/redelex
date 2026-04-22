## From Scratch (Baseline)
```bash
sbatch -o logs/slurm_gpu_run_cl_from_scratch_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/continuous_learning/slurm_gpu_run_cl_from_scratch.sh
```


## Finetune on Full Dataset
```bash
sbatch -o logs/slurm_gpu_run_cl_ft_full_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/continuous_learning/slurm_gpu_run_cl_ft_full.sh
```


## Finetune with Upsampling New Data
```bash
sbatch -o logs/slurm_gpu_run_cl_ft_upsample_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/continuous_learning/slurm_gpu_run_cl_ft_upsample.sh
```


## Finetune on New Data Only
```bash
sbatch -o logs/slurm_gpu_run_cl_ft_newonly_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/continuous_learning/slurm_gpu_run_cl_ft_newonly.sh
```