# Settled decisions

Append-only. One entry per decision plus the evidence that settled it. Do not re-litigate without
new evidence. September decisions live on `archive/september-cl:notes/DECISIONS.md`; only the ones
still binding are restated here.

## Rerun decisions (2026-10-14/15, Jakub)

- **Everything is rerun; nothing earlier is reused.** Training now runs until convergence and the
  code changes. Neither the NeurIPS runs (MLflow exp 90-93; ~47% of sampled checkpoints carry NaN
  feature encoders) nor the September grid (exp 115-150) is comparable.
- **Lineage:** 5 independent chains per config; seed s continues its own model. No best-of-k parent.
- **Main roster:** `scratch_all`, `scratch_last`, `cumulative` (uniform over all data; Jakub's
  "p = 0"), `replay_p50`, `incremental` (p = 1). CL methods, other p, last-k scratch, the RQ4 grid
  and RelGNN are ablations on a subset chosen after the main results.
- **Datasets:** rel-f1, rel-trial, rel-stack, rel-hm, rel-amazon, rel-ratebeer; rel-event if it
  keeps >= 5 episodes; rel-avito in the appendix.
- **Stopping:** on held-out entities (~10% by ID hash, never trained on) across the whole last
  episode; forward-only check every N optimizer steps (N counts steps, not batch size); patience P;
  restore best. **Never** on the next episode: it is the reported window, and its labels do not
  exist at update time.
- **Notebooks before experiments:** rel-f1 notebooks test every piece of methodology first.

## Findings that shaped the protocol (verified on the September code, 2026-10-14)

- **Selection on the reported window.** The September driver selected checkpoints on
  `[splits[i], splits[i+1])` and reported that same window as the next-window score; on windows
  <= 25k rows the rows were identical. The new protocol never selects or stops on it.
- **Label delay held only implicitly** (RelBench anchors on a Δw grid ending at `val_ts - Δw`).
  The new protocol filters `t <= T - Δw` explicitly; dense or shifted anchors would leak otherwise.
- **`rnd_uni` mixing was proportional to loader size**, so the published "Upsampled 50/50" regime
  ran at 5-10% new data after episode 1. Exact mixing needs per-sample weights.

## Carried-over environment facts (still binding)

- **Stay on relbench 2.1.1.** 3.x is a breaking rewrite (HF-backed loaders); RelArena pins 2.1.2.
- **rel-stack: load with `download=False`.** Upstream republished `db.zip` without refreshing the
  pinned SHA256; the archive was extracted manually into `~/.cache/relbench/rel-stack/db/`.
- **rel-amazon: load with `download=True`.** Building from raw fetches a UCSD URL that now 404s.
- **rel-event:** DB span holds sentinel dates (1912 -> 2222) and `user-ignore` fails its hash.
- **RCI partitions:** `amdgpufast` (4 h), `amdgpu` (24 h), `amdgpulong` (72 h) are NVIDIA
  A100-SXM4-40GB (the "amd" is the host CPU); `gpu*` partitions are V100-SXM2-32GB.
- **MLflow:** `http://potato.felk.cvut.cz:2222`. The Python `MlflowClient` stalled on a large
  `search_runs` (2026-10-14) while the REST API answered in < 1 s; page the REST endpoint if it hangs.

## Checkpoint A (2026-10-15, Jakub)

- **rel-f1 holdout fraction: 20%.** The smallest fraction with at least 8 held-out drivers at every
  update (10% leaves 4). Evidence: `notebooks/methodology/out/02_holdout_choice_rel-f1.json`. The
  stopping signal stays noisy at 20% (entity-bootstrap SE: AUROC ~0.07, MAE ~0.8); notebook 03
  measures what that costs.
- **Calibration scope (notebook 03a):** driver-position + driver-dnf, regimes scratch_all /
  cumulative / incremental, seeds 0-1, plus a plateau-LR arm on cumulative; ~13 GPU-h on RCI.
