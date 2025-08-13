# Pretraining

## Linear SAGE

```bash
sbatch -o logs/slurm_cpu_big_run_linear_sage_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_linear_sage_pretrain.sh

sbatch -o logs/slurm_cpu_xl_run_linear_sage_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_linear_sage_pretrain.sh
```


## ResNet SAGE

```bash
sbatch -o logs/slurm_cpu_big_run_resnet_sage_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_resnet_sage_pretrain.sh

sbatch -o logs/slurm_cpu_xl_run_resnet_sage_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_resnet_sage_pretrain.sh
```

## Linear DBFormer

```bash
sbatch -o logs/slurm_cpu_big_run_linear_dbformer_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_linear_dbformer_pretrain.sh

sbatch -o logs/slurm_cpu_xl_run_linear_dbformer_pretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_linear_dbformer_pretrain.sh
```


# No Pretraining

## Linear SAGE

```bash
sbatch -o logs/slurm_cpu_big_run_linear_sage_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_linear_sage_nopretrain.sh

sbatch -o logs/slurm_cpu_xl_run_linear_sage_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_linear_sage_nopretrain.sh
```

## ResNet SAGE

```bash
sbatch -o logs/slurm_cpu_big_run_resnet_sage_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_resnet_sage_nopretrain.sh

sbatch -o logs/slurm_cpu_xl_run_resnet_sage_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_resnet_sage_nopretrain.sh
```

## Linear DBFormer

```bash
sbatch -o logs/slurm_cpu_big_run_linear_dbformer_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_big_run_linear_dbformer_nopretrain.sh

sbatch -o logs/slurm_cpu_xl_run_linear_dbformer_nopretrain_$(date '+%d-%m-%Y_%H:%M:%S').log experiments/scripts/pretrain/slurm_cpu_xl_run_linear_dbformer_nopretrain.sh
```