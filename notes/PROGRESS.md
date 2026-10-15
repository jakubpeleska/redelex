# PROGRESS — CL benchmark rerun (Oct 2026 revision)

**Entry point for any new session.** Read this, then `notes/DECISIONS.md`.

- Plan: `~/.claude/plans/home-pelesjak-claude-uploads-1ef0c075-1-idempotent-beaver.md`
  (shared copy: Claude Doc "Incremental RDL Rerun Plan", https://claude.ai/artifact/5QMxFGYwT13mm9cHfPgxzT).
- Jakub's draft revision plan: `notes/revision-plan-2026-10-14.md`.
- Paper status: **rejected** at NeurIPS 2026 E&D (2026-09-25; NHfu 4, CMu9 3, d22i 3, JDci 2).
  Revision venue **undecided**, so work stays anytime-shaped.

## Base

This branch restarts from `develop` (PR #24) plus Jakub's `WIP` commit 474259e (weighted
`ComposedLoader`, per-validation plateau stepping and `modes` in `entity_wrapper.py`).
The September CL work (59 commits: cl_modes, redelex/continual, RCI launcher, analysis notes) is kept
**locally** on `archive/september-cl` as a source to port from. Nothing from it is assumed present.

## Current phase: methodology rewrite + rel-f1 notebooks (started 2026-10-15)

Every incremental-learning experiment is rerun with training to convergence; no earlier result is
reused. Nothing launches on RCI until `notebooks/methodology/` passes checkpoints A-D and a
rel-f1 + rel-trial pilot passes E.

| step | status |
|---|---|
| State check of repo, RCI, MLflow vs. the Oct-14 draft | DONE 2026-10-14 |
| Plan approved, shared as a Claude Doc | DONE 2026-10-15 |
| RCI clone reset to the GitHub branch (474259e), `analysis_out/` removed | DONE 2026-10-15 |
| NaN-encoder fix ported onto the restarted base (`models.py`) | DONE 2026-10-15 |
| Library primitives + unit tests: protocol, anchor tables + held-out entities, exact-share mixture loader, convergence / cost-meter / anytime callbacks, analysis | DONE 2026-10-15 |
| New driver `chain_runner.py` + inline scoring; all 13 regimes incl. EWC and reservoir | DONE 2026-10-15 |
| Notebook tooling (papermill, `scripts/run_methodology_notebooks.py`, `slurm/rci/run_notebook.sh`) | DONE 2026-10-15 |
| Notebooks 00-02 -> checkpoint A | 00, 01 PASS on CPU; 02 running |
| Notebooks 03-04 -> checkpoint B | TODO |
| Notebooks 05-07 -> checkpoint C | TODO |
| Notebook 08 budget -> checkpoint D | TODO |
| Pilot rel-f1 + rel-trial main grid -> checkpoint E | TODO |

## Where things are

- Library: `redelex/continual/{protocol,anchors,analysis,regularization,replay,shrink_perturb}.py`,
  `redelex/loaders/mixture.py`, `redelex/nn/train/{callbacks,anytime}.py`.
- Driver: `experiments/continuous_learning/{chain_runner,modes,score_chain,continuous_task}.py`.
  The legacy `continuous_learning.py` is untouched and not used by the rerun.
- Notebooks: sources `notebooks/methodology/src/*.py`, built with `scripts/build_notebooks.py`,
  executed copies + `status.json` in `notebooks/methodology/executed/`, small JSON outputs in
  `notebooks/methodology/out/`.
- Tests on CPU cap torch at 4 threads (`tests/conftest.py`); the default oversubscribes this box.

## Environment facts

- Compute: **RCI only**, up to **6 concurrent GPUs**. Potato GPUs are blocked by the container
  device cgroup (`nvidia-smi`: "Failed to initialize NVML: Unknown Error"), as of 2026-10-14.
- RCI storage is not limited (no per-user quota, 33 T free). Old outputs (~377 GB under
  `logs/`, `data/`, `~/git/ctu-relational-py/logs`) are kept unless Jakub asks to delete them.
- This machine has no GitHub credentials: commits are pushed by Jakub; RCI then resets to GitHub.
- Always run python via `.venv/bin/python` (or `uv run --no-group cpu --group cu128`); a plain
  `uv run` resyncs the CPU torch build.
