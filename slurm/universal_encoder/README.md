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