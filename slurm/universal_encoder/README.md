## HeteroEncoder HeteroSAGE
```bash
sbatch -o logs/slurm_gpu_big_run_baseline_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_big_run_baseline.sh
```

```bash
sbatch -o logs/slurm_cpu_xl_run_baseline_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_cpu_xl_run_baseline.sh
```


## UniversalEncoder HeteroSAGE
```bash
sbatch -o logs/slurm_gpu_big_run_universal_hetero_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_big_run_universal_hetero.sh
```

```bash
sbatch -o logs/slurm_cpu_xl_run_universal_hetero_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_cpu_xl_run_universal_hetero.sh
```


## UniversalEncoder HomogeneousSAGE
```bash
sbatch -o logs/slurm_gpu_big_run_universal_homogeneous_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_big_run_universal_homogeneous.sh
```

```bash
sbatch -o logs/slurm_cpu_xl_run_universal_homogeneous_$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_cpu_xl_run_universal_homogeneous.sh
```

## UniversalEncoder HomogeneousGAT
```bash
sbatch -o logs/slurm_gpu_big_run_universal_homogeneous_gat$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_big_run_universal_homogeneous_gat.sh
```

```bash
sbatch -o logs/slurm_cpu_xl_run_universal_homogeneous_gat$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_cpu_xl_run_universal_homogeneous_gat.sh
```


## UniversalEncoder HomogeneousGAT
```bash
sbatch -o logs/slurm_gpu_big_run_universal_homogeneous_gat$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_big_run_universal_homogeneous_gat.sh
```

```bash
sbatch -o logs/slurm_cpu_xl_run_universal_homogeneous_gat$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_cpu_xl_run_universal_homogeneous_gat.sh
```

## Pretraining UniversalEncoder HomogeneousSAGE
```bash
sbatch -o logs/slurm_gpu_xl_run_universal_pretrained_homoge$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_xl_run_universal_pretrained_homoge.sh
```

## Pretraining UniversalEncoder HomogeneousSAGE Ablations
```bash
sbatch -o logs/slurm_gpu_xl_run_universal_pretrained_homoge_abla$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_xl_run_universal_pretrained_homoge_abla.sh
```

## Pretraining UniversalEncoder HeterogenousSAGE
```bash
sbatch -o logs/slurm_gpu_xl_run_universal_pretrained_hetero$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_xl_run_universal_pretrained_hetero.sh
```

## Pretrained UniversalEncoder with HomogeneousSAGE validation 
```bash
sbatch -o logs/slurm_gpu_run_universal_pretrained_homoge_validation$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_run_universal_pretrained_homoge_validation.sh
```


## Pretrained UniversalEncoder with HeterogenousSAGE validation 
```bash
sbatch -o logs/slurm_gpu_run_universal_pretrained_hetero_validation$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_run_universal_pretrained_hetero_validation.sh
```

## Pretrained UniversalEncoder with HomogeneousSAGE Ablations validation 
```bash
sbatch -o logs/slurm_gpu_run_universal_pretrained_homoge_abla_validation$(date '+%d-%m-%Y_%H:%M:%S').log slurm/universal_encoder/slurm_gpu_run_universal_pretrained_homoge_abla_validation.sh
```