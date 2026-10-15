#!/bin/bash
# One benchmark job per dataset, sized by the memory each needed in September.
#   bash slurm/rci/submit_benchmarks.sh          (DRY=1 prints the sbatch lines)
set -euo pipefail
cd "${CL_REPO:-$HOME/git/claude-redelex}"
mkdir -p logs/rci/bench
declare -A MEM=( [rel-f1]=16G [rel-trial]=32G [rel-stack]=64G [rel-hm]=64G [rel-amazon]=224G
                 [rel-ratebeer]=80G [rel-event]=32G [rel-avito]=64G )
declare -A TASKS=(
  [rel-f1]="driver-position driver-dnf driver-top3"
  [rel-trial]="study-outcome study-adverse site-success"
  [rel-stack]="user-engagement post-votes user-badge"
  [rel-hm]="user-churn item-sales"
  [rel-amazon]="user-churn user-ltv item-churn item-ltv"
  [rel-ratebeer]="beer-churn user-churn user-count brewer-dormant"
  [rel-event]="user-repeat user-ignore user-attendance"
  [rel-avito]="ad-ctr user-visits user-clicks"
)
for ds in rel-f1 rel-trial rel-stack rel-hm rel-ratebeer rel-event rel-avito rel-amazon; do
  cmd=(sbatch --mem="${MEM[$ds]}" --job-name="cl2-bench-$ds" slurm/rci/run_benchmark.sh "$ds" ${TASKS[$ds]})
  if [ -n "${DRY:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
