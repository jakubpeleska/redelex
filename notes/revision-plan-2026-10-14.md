# Incremental RDL — Revision Plan

Oct 14, 2026 · @Jakub Peleška

## Context

Goal: a minimal-effort revision of the rejected paper that shifts the pitch from "incremental learning" to keeping RDL models current cheaply on growing databases.

The paper was rejected at the NeurIPS 2026 Evaluations & Datasets track (decision 25 Sept 2026). Reviewer scores: NHfu 4, CMu9 3, d22i 3, JDci 2. The AC's stated reasons:

- Narrow breadth: 12 tasks on 4 databases, some with only 2 training episodes (rel-avito, driver-top3).
- Limited baselines: no CL methods, a single backbone (GraphSAGE + ResNet).
- Scope excludes updates and deletions (append-only, OLAP-like growth).
- The decay metric has little practical effect: it changed the regime ranking only for driver-position at α = 0.9.

Commitments made in the rebuttal: add RelGNN as a backbone, ablate ΔI, sweep the upsampling ratio, correct the Fig. 4 wording, cite Gama et al. 2013 and Hidalgo et al. 2019, and use the OLTP/OLAP framing.

## Framing

The paper should say that the problem on growing databases is staying current cheaply, not avoiding forgetting.

**Terminology.** In the CL literature, "incremental learning" is a synonym for continual learning (task-, domain-, class-incremental; van de Ven et al. 2022). Do not argue "IL is not CL". Define the setting by its properties instead, and state which CL assumptions fail.

**Properties of the setting:**

1. One fixed task: the query, label space and head stay the same. The data distribution shifts over time; in CL terms this is closest to domain-incremental.
2. No restriction on past data: the database is append-only and keeps everything. Most CL methods (EWC, LwF, bounded replay) assume past data is unavailable.
3. The binding costs are compute per update and staleness, not memory.
4. Deployment evaluates only the future. Old windows matter only if old patterns recur.

**Two kinds of staleness.** RDL models read the current context G≤t at inference, even without retraining, so data staleness does not occur. Retraining only fixes weight staleness: the mapping was learned on older label windows.

**Label delay.** A label for anchor t is known at t + Δw. At any update time T, the newest usable training label comes from anchor T − Δw. Weights are therefore always at least Δw stale, however often you update.

**Closest prior work** (check the details before citing):

| Work | Relevance |
| --- | --- |
| Wild-Time (Yao et al., NeurIPS 2022 D&B) | Temporal-shift benchmark on non-relational data, with CL baselines; as I recall, nothing consistently beat ERM |
| Prabhu et al., CVPR 2023, compute-budgeted CL | Under a compute budget, simple sampling baselines beat CL methods |
| Ash & Adams, NeurIPS 2020, warm-starting | Warm starts generalized worse than fresh init; our warm start ≥ scratch result runs counter to this |
| Dohare et al., Nature 2024, loss of plasticity | Explains why plasticity, not forgetting, can be the limit |
| Gama et al. 2014, drift survey | Sliding-window baseline; drift-adaptation framing |
| Gama et al. 2013; Hidalgo et al. 2019 | Fading factors (promised citations for the decay metric) |

## Research problem

An RDL model runs on a database that grows by appending rows, and the relation between the relational context and the target changes over time. All history stays accessible, but every update costs compute. The paper asks three things:

- how quickly models lose accuracy (staleness);
- which update strategy, at which update frequency and per-update budget, gives the best prediction quality per unit of compute;
- how much these conclusions depend on protocol choices.

**Core measurement.** All metrics derive from a performance matrix R, where R\[i, j\] is the score of the model updated through episode i, evaluated on window j. This is the standard CL accuracy matrix (Lopez-Paz & Ranzato 2017). The current decay metric is a weighted average over one row of R.

## Research questions and methodology

Four RQs. RQ2 and RQ4 carry the compute argument; RQ1 motivates it and RQ3 supplies the CL baselines.

### RQ1: How fast do RDL models go stale, separated from window difficulty?

- **Metric: staleness penalty.** S(k) = mean over windows j of R\[j−1, j\] − R\[j−1−k, j\], with the sign flipped for MAE. Models of different ages are compared on the same window, so differences in window difficulty cancel out. This fixes reviewer JDci's W4 on Fig. 4.
- **What it shows.** A positive S(k) shows performance-relevant temporal shift. It does not prove that P(y|x) changed rather than P(x); say so in the paper instead of claiming "concept drift".
- **Already have:** From Scratch models M\_i for every episode, and the target statistics in Fig. 3 (kept as descriptive).
- **New:** evaluate every M\_i on all later windows. If the checkpoints were saved, this is inference only.

### RQ2: When a new increment arrives, how much compute does each strategy need to reach a given next-window performance, and what does it reach at a fixed budget?

- **Strategies as one family:** replay with mixing ratio p, the probability of drawing a batch from the newest increment.
  - p = 1 is Incremental.
  - p = 0.5 is the current Upsampled regime.
  - p equal to the newest increment's share of the data is Cumulative under uniform sampling (check the sampler).
- **Scratch baselines as one sweep over the training window k** (the number of most recent episodes used):
  - k = all: From Scratch on the full history (the current baseline).
  - k = 1: From Scratch on the last episode only.
  - k ∈ {2, 4, 8}: From Scratch on the last k episodes, capped at the number of available episodes.
  - Option A (chosen), sweep over time: run every k at every episode on one dataset with many episodes (rel-stack, 16–18 episodes). This shows whether the best k changes over time.
  - Option B (not used), single point in time: run every k once per task, trained up to the original RelBench val timestamp and evaluated on the original val/test windows. This covers every dataset, costs one run per k per task, and is directly comparable to RelBench numbers.
  - Decision: option A on rel-stack. The k sweep is not run on the other datasets.
  - Caveat: under the step budget every k costs the same per update, so k changes compute only under epoch-based training. On small-increment tasks (rel-f1, rel-avito), k = 1 may be too little data to train from scratch; report the number of training rows per k.
- **Sweep:** p ∈ {0.25, 0.5, 0.75}, as committed in the rebuttal.
- **Train to convergence to measure compute.** The current protocol caps every regime at 2000 steps of 128 samples, and per-step cost is roughly constant (fixed fanout 32, 16). Under that cap, regimes cost about the same, so the cap hides the cost differences. For the cost results, train every regime until convergence instead:
  - Convergence = no improvement in held-out validation loss for P consecutive checks, with a check every N steps. The stopping point defines the cost.
  - The held-out set must not be the evaluation window. Use a random \~10% of the newest labelled rows, split by entity and never trained on; this also avoids the selection bias in the pipeline checks.
  - Raise or remove the step cap and the 2-hour wall-clock limit. Report any run that still hits a cap.
  - Record steps to convergence, samples processed and GPU time, all excluding validation overhead (report that separately).
  - Report results for two patience values P, because the stopping rule directly sets the measured cost.
  - Incremental saves training targets, not graph construction: every regime needs the full history as message-passing context.
- **Metrics:**
  1. Anytime curve: the next-window score at steps {0, 10, 25, 50, 100, 250, 500, 1000, 2000} after an update starts. Step 0 is the unchanged previous model (the "don't update" baseline); for Scratch it is a random model.
  2. Steps-to-target: steps until the score reaches the fully trained Scratch score minus one seed SD. Report the median across episodes, and "never" when not reached.
  3. Normalized area under the anytime curve: one number per task and regime, used for ranking.
  4. Total horizon cost: steps to convergence, samples processed and GPU time, summed over all episodes, together with the final next-window score at convergence. The anytime curve is the trajectory up to that stopping point. Fixed-budget results (e.g. 100 steps) stay as a secondary view.
- **Aggregation across tasks:** average rank, plus a Friedman or Wilcoxon test over tasks.
- **Backbone:** GraphSAGE only for all main results; it is enough to evaluate the setting. A backbone sweep (e.g. RelGNN) runs on one smaller dataset only, as a robustness check of the regime ranking. Candidate: rel-trial, which has 9 episodes and moderate size. This partly covers the RelGNN commitment from the rebuttal.
- **Learning rate:** no LR sweep. All regimes use one fixed LR (currently 1e-3), stated in the paper, because the goal is to compare regimes, not to build the best model. No calibration run: the LR comes from established From Scratch configurations on the full datasets (RelBench / ReDeLEx baselines), cited in the paper. Caveat: a high LR can erase what warm-started models carry over, so a single LR is not fully neutral between fine-tuning and scratch. Mention this as a limitation.
- **Already have:** 4 regimes × 5 seeds × 12 tasks; validation every 100 steps on the next increment; Fig. 5a (full) and Fig. 5b (100 steps); the epoch-cost table from the CMu9 rebuttal.
- **New:** step-0 evaluation; short runs to fill steps 10–50; the p sweep; the scratch k sweep (option A, rel-stack); the backbone sweep on one small dataset.

### RQ3: Do CL methods designed for restricted data access help or hurt here?

- **Methods:**
  - EWC (regularization);
  - replay with a bounded reservoir buffer (memory-limited replay);
  - shrink-and-perturb (a plasticity method, about one line of code).
- **Hypothesis:** EWC and bounded replay do not beat unbounded replay or Incremental on next-window score, because they pull the model toward the old mapping. Report whatever the results show.
- **Measuring forgetting directly.** Hold out about 10% of task rows in each past window and never train on them. Report BWT = mean over j < i of R\[i, j\] − R\[j, j\] on those rows. This replaces the rebuttal's indirect argument (Incremental stays competitive, so forgetting is minor) with a measurement. It is a diagnostic, since past windows are not evaluated in deployment.

### RQ4: How do update frequency and per-update budget trade off against prediction quality?

This is the main cost question.

- **Two protocol axes: episode size ΔI and anchor timestamps.** (1) Episode size ΔI sets how often the model is updated. (2) Anchor timestamps are given by the task. They can be shifted or given a different stride, but their natural stride is the task's future window Δw, which is the default here. With stride Δw, ΔI takes multiples of Δw, because an update more often than Δw has no new labelled anchors. Use a denser anchor stride s < Δw (e.g. Δw/4) on datasets with a large Δw (rel-stack, 91 days), so that episodes shorter than the window (ΔI < Δw) can be tested. This needs extra anchor timestamps generated for the task. Label windows then overlap and consecutive rows are correlated. To keep configurations comparable, all ΔI values on rel-stack use the same dense anchors, for training and for the evaluation timeline; otherwise ΔI would be confounded with the amount of training data.
- **Deployment simulation, so granularities are comparable:**
  - One shared timeline: every task anchor (stride Δw) from t₀ to t\_end is a prediction point, identical across configurations.
  - The model used at each point is the latest one updated before it.
  - Every configuration starts from one model trained up to t₀.
  - Configuration = (regime, ΔI, steps per update B).
  - Score = mean over the timeline. Cost = number of updates × B, plus GPU time.
  - Output: score vs total cost with the Pareto frontier, plus comparisons at matched total compute (many small updates vs few large ones).
- **Grid:**
  - ΔI ∈ {Δw, 2Δw, 4Δw, …, current ΔI, ∞}; on rel-stack also {Δw/4, Δw/2} (dense anchors). ∞ means no updates after t₀ (the pure staleness baseline).
  - B ∈ {50, 200, 1000}.
  - Regimes: Incremental, replay at p = 0.5 and one other p, Cumulative, Scratch.
- **Predictions:**
  - With small increments (possibly smaller than one batch), Incremental becomes noisy or overfits and replay matters more, so the best p rises with increment size. Report increment size in samples, not just days.
  - Score levels off as ΔI approaches Δw, because of the label-delay bound.
- **Anchor shift (second axis).** Shift the task's anchor timestamps by δ ∈ {0, ¼, ½, ¾}·Δw, keeping stride Δw. This changes which anchors and label windows exist and how they fall into episodes. It tests whether regime rankings depend on the arbitrary anchor placement; I expect it to add variance but not change rankings.
- **Subset:** two datasets only, to keep the grid affordable: rel-stack (Δw = 91 days, 16–18 episodes) and rel-hm (Δw = 7 days, to be checked), 3 seeds. rel-amazon is the fallback if rel-hm turns out unsuitable. The paper states that the other datasets are excluded from RQ4 for cost, and that rel-f1 and rel-avito also have too few rows per increment or too short a span.
- **Optional:** drift-triggered updates (update when held-out loss on the newest labels rises). Otherwise future work.

## Datasets

Add rel-amazon and rel-hm to the current four databases. Report the number of episodes per task, and put tasks with fewer than about 5 episodes in the appendix rather than the main aggregate.

| Database | Tasks | Status | Note |
| --- | --- | --- | --- |
| rel-stack | post-votes, user-badge, user-engagement | have | Many episodes; use for RQ4 |
| rel-trial | site-success, study-adverse, study-outcome | have | 9 episodes |
| rel-f1 | driver-position, driver-dnf, driver-top3 | have | Small increments; driver-top3 has 2 episodes; exclude from RQ4 |
| rel-avito | ad-ctr, user-clicks, user-visits | have | Short time span, 2 episodes; appendix; exclude from RQ4 |
| rel-amazon | user-churn, item-churn, user-ltv, item-ltv | add | Long history; RQ4 fallback if rel-hm is unsuitable |
| rel-hm | user-churn, item-sales | add | I believe the windows are 7 days (check); use for RQ4 |
| rel-event | user-repeat, user-ignore, user-attendance | check | Check the time span first |
| RelBench v2 additions | — | check | Which have temporal entity tasks? |

Recommendation tasks stay out, with one sentence of justification: they need different heads and ranking metrics, and the candidate set grows over time.

Earlier results on the potato server are treated as nonexistent for now.

## Pipeline checks before rerunning

The first two items can bias every result; the last three decide how much rerunning is needed. Check all of them in the code before launching new runs.

- [ ] **Checkpoint selection on the reported window.** Sec. 5.3 says the best checkpoint is selected on the next increment (t\_i to t\_i+1), and Fig. 5 reports the score on that same increment. If it is the same data, the numbers are optimistically biased, possibly by a different amount per regime. Fix: select on episode i+1 and report on i+2, or select on a held-out tail of the data available at t\_i. The anytime curves (RQ2) need no selection, so they avoid this problem.
- [ ] **Label delay.** At horizon t\_i, only anchors with t ≤ t\_i − Δw have fully observed labels. The paper says anchors in (t\_i−1, t\_i\] are revealed at t\_i; if they are trained on, their labels contain events after t\_i. I believe RelBench's fixed split ends training anchors at val\_timestamp − timedelta for this reason. Check the incremental loader.
- [ ] **Task-table generation.** Can the loaders produce task tables at an arbitrary anchor stride and phase (e.g. via the SQL task queries in ReDeLEx), or only RelBench's fixed tables? RQ4 depends on this.
- [ ] **Logs.** Do the Lightning logs keep the metric at every validation step, or only the best checkpoint's? Per-step logs would give RQ2 anytime curves at 100-step resolution for free.
- [ ] **Checkpoints.** Were the per-episode checkpoints saved? If so, RQ1 and step-0 evaluation are inference only.

## Changes to the paper

The paper changes its pitch and metrics; the existing protocol and runs stay.

**New contribution list:**

1. A multi-episode protocol for growing relational databases, with metrics derived from the matrix R (staleness penalty, anytime curves, BWT).
2. Staleness measurements across about 18 tasks.
3. A compute-aware comparison of update strategies, including CL baselines (GraphSAGE, with a backbone check on one dataset).
4. A study of update frequency and per-update budget, with Pareto frontiers.

**Cut or demote:**

- The decay metric: drop it as a contribution and keep it in the appendix as one aggregation of a row of R.
- Current Fig. 4: replace it with the staleness penalty plot.

**Fixes carried over from the rebuttal:**

- [ ] OLTP vs OLAP framing; append-only stated as scope.
- [ ] Fig. 4 wording ("degradation" was wrong).
- [ ] Use "schema/rows" instead of "heading/body".
- [ ] Remove the duplicate "Mean" label in Fig. 3b.
- [ ] State the episode counts per task in the main text.
- [ ] Cite Gama et al. 2013, Hidalgo et al. 2019 and Galke et al. 2020.

## Effort order and open questions

Do the work cheapest first. The RQ4 grid costs the most compute and runs only on a subset of tasks.

1. Pipeline checks (above).
2. Build R and step-0 scores from existing checkpoints (RQ1; inference only).
3. Extract anytime curves from the existing validation logs (RQ2).
4. Add rel-amazon and rel-hm with the existing pipeline.
5. Run the p sweep, sliding window, shrink-and-perturb and EWC (small code changes).
6. Add bounded replay; run the backbone sweep on one small dataset.
7. Generate dense-anchor task tables and run the RQ4 grid on 3–4 tasks.

**Open questions:**

- Resolved: the anchor ablation has two axes, episode size ΔI and anchor shift at the natural stride Δw. A denser stride (Δw/4) is used on rel-stack to test ΔI < Δw.
- Which RelBench v2 datasets have suitable temporal entity tasks?
- Target venue and deadline?
