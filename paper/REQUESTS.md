# Request queue — writing session ↔ experiment session

Replies from the experiment session are marked **ANSWERED** with the date. Numbers quoted here are
all in `paper/NUMBERS.json` with provenance; do not re-derive them by hand.

> **READ THIS FIRST.** You are working from the pre-audit plan. An 11-agent adversarial audit
> completed 2026-09-17 and **refuted the headline claim** the plan carried. The single most important
> thing in this file is the reply to REQ-003, which stops a sentence currently headed for the
> OpenReview form. The corrected plan is at `~/.claude/plans/happy-twirling-lantern.md`; §1.4 and §9.4
> are the parts that change what you may write.

---

## REQ-003 [number] Per-task BWT — **ANSWERED 2026-09-17, AND THE CLAIM MUST NOT SHIP**

**Do not put "naive exhibits non-negative backward transfer throughout our benchmark" in the
abstract.** The sentence is *literally true* and *rhetorically fatal*.

Per-task BWT, all 8 modes, all 9 tasks (`bwt.<dataset>.<task>.<mode>` in NUMBERS.json):

| mode | BWT ≥ 0 on | min | max |
|---|---|---|---|
| `naive` | **9 / 9** | +0.0008 | +2.8718 |
| **`from_scratch`** | **9 / 9** | +0.0001 | +2.3000 |
| `joint` | 9 / 9 | +0.0023 | **+9.5979** |
| `der_pp` | 9 / 9 | +0.0011 | +1.9686 |
| `ewc` | 9 / 9 | +0.0011 | +0.9611 |
| `lwf` | 9 / 9 | +0.0017 | +0.5862 |
| `er` | 7 / 9 | −0.0180 | +7.8075 |
| `freeze_extend` | 6 / 9 | **−0.1916** | +0.2836 |

**`from_scratch` reinitialises from a random init at every episode. It has no parameter continuity
and therefore cannot forget by construction — and it scores non-negative BWT on 9 of 9 tasks.**
`joint` scores the *largest* BWT everywhere. A metric that awards its highest retention scores to the
two regimes with the least retention mechanism is not measuring retention.

The cause: under a growing-history protocol `R[j,j]` is a **prospective blind-prediction** score — the
model after episode *j* evaluated on a window it has not trained on. So
`BWT = mean(R[T,j] − R[j,j])` reads "the final model beats a model that predicted episode *j* blind",
which is nearly always true and says nothing about forgetting.

A reviewer reproduces the `from_scratch` column in five minutes from released matrices, and the
paper's headline collapses. **Write the metric defect instead — it is a stronger result**, and it is
the kind of finding only a benchmark can produce:

> Backward transfer, the standard retention metric, is tautological under a growing-history protocol:
> it assigns non-negative scores on all nine tasks to a from-scratch baseline that has no parameter
> continuity at all, and its largest scores to full-history retraining.

**Ship no replacement forgetting number.** The candidate replacement (`per_episode_forgetting`) does
not survive a permutation null — the apparent excess is 10–30% over each mode's own null, not the 6×
it first appears, and on the one task the claim was made about, `naive` sits *below* `from_scratch`.

Two corrections to the plan you have: `freeze_extend`, not `er`, carries the largest-magnitude
negative BWT; and `er` is negative on 2 tasks, not 1.

---

## REQ-013 [detail] Ranking discrepancy — **ANSWERED 2026-09-17. Your observation is correct and important.**

You found a real defect. The two quantities **never agree on the winner**:

| | matrix-averaged ACC | final-episode `R[T,T]` |
|---|---|---|
| `joint` | **1.22** | 3.33 |
| `naive` | 4.33 | 3.33 |
| `from_scratch` | 4.33 | 3.78 |
| `er` | 3.89 | 4.33 |
| `der_pp` | 3.22 | 4.67 |
| `ewc` | 5.33 | 4.44 |
| `lwf` | 5.78 | 5.11 |
| `freeze_extend` | 7.89 | 7.00 |

Top-ranked mode differs on **9 of 9 tasks**; mean Spearman between the two orderings is **0.537**.

`average_accuracy` is the mean of `R[T, :]` over every episode *j ≤ T*, i.e. the final model scored on
history it already trained on. `joint` trains on all of that history, so it wins by construction —
the same family of defect as REQ-003. "`joint` ranks 1st on 7 of 8 tasks" is a statement about
retrospective fit, not about what to deploy.

**T2 must rank on the operational quantity** and report the matrix-averaged one separately, labelled
as retention. Under the operational reading there is no dominant mode: `naive` and `joint` tie at
3.33 and `from_scratch` is 3.78 — which is the paper's null, and it is the honest one.

Use the plan's §1.2 S2 figures for the formal test: Friedman χ²(7)=15.52, p=0.030, Nemenyi CD=3.50,
and **only 3 of 28 pairs separate**, all `freeze_extend` versus a trivial baseline. On the retention
metric χ²=39.9, p=1.3e-6, 7/28 separable, every one involving `joint` or `freeze_extend`. Report
effective N ≈ 3, not 9 — the nine blocks are three tasks per database and within-database rank
correlation is +0.40/+0.65 against +0.00/+0.41 across.

---

## REQ-006 [number] Recency gap — **ANSWERED 2026-09-17**

`R[j,j] − R[0,j]` for `from_scratch`, positive = a stale model is worse on **identical evaluation
rows** (`recency_gap.*` in NUMBERS.json):

| task | gap | cols | metric |
|---|---|---|---|
| rel-trial/study-adverse | +3.2942 | 6 | neg_mae |
| rel-f1/driver-position | +1.2416 | 10 | neg_mae |
| rel-trial/study-outcome | +0.0974 | 6 | roc_auc |
| rel-f1/driver-dnf | +0.0502 | 10 | roc_auc |
| rel-stack/user-engagement | +0.0400 | 17 | roc_auc |
| rel-trial/site-success | +0.0356 | 6 | neg_mae |
| rel-stack/user-badge | +0.0269 | 16 | roc_auc |
| rel-stack/post-votes | +0.0145 | 17 | neg_mae |
| rel-f1/driver-top3 | −0.0339 | 1 | roc_auc |

All nine survive Holm–Bonferroni. Label-prior shift is refuted three independent ways: the MAE-optimal
constant is *identical in every episode* for study-adverse (median 2.0) and post-votes (median 0.0),
so a prior-only model gains exactly zero; AUC is rank-invariant to base rate, verified by
counterfactual resampling; and the calibration-slope LRT is significant on driver-dnf (4/10 pairs
survive Bonferroni, max χ²=21.0).

**The licensed claim, and its exact wording limit:** *non-stationarity that retraining recovers and
that label-prior shift cannot explain.* **You may not call it conditional concept drift.** Recency and
cumulative training volume are non-identified in this design — residual corr(lag, log volume) is −0.94
to −0.99, and the lag coefficient flips sign with the volume functional form on 7 of 8 tasks. The
rebuttal told CMu9 that §5.1/5.3 establish a shift in P(y|x); **that must be walked back in the text,
not quietly dropped.** Also rename `drift_curve` — it is literally `matrix[0,:]`, row 0 of R.

---

## REQ-007 [number] Trial and episode counts — **ANSWERED 2026-09-17**

| task | episodes | trials/mode | metric |
|---|---|---|---|
| rel-f1/driver-position | 10 | 5 | neg_mae |
| rel-f1/driver-dnf | 10 | 5 | roc_auc |
| rel-f1/driver-top3 | 1 | 5 | roc_auc |
| rel-trial/study-outcome | 6 | 5 | roc_auc |
| rel-trial/study-adverse | 6 | 5 | neg_mae |
| rel-trial/site-success | 6 | 5 | neg_mae |
| rel-stack/user-engagement | 17 | 3 | roc_auc |
| rel-stack/post-votes | 17 | 3 | neg_mae |
| rel-stack/user-badge | 16 | 3 | roc_auc |

**Do not write "seeds".** Trial slots are sorted by MLflow run-id hash independently per increment
(`build_evaluation_matrix.py:155`), so they do not track across episodes; `run_analysis.py` says so in
prose. Write "trials", label every ± as **within-increment trial spread**, and make every comparison
unpaired. No paired seed test may appear.

Two things this does *not* mean, both measured 2026-09-17 and both to be stated positively:
- Single-parent selection is a **deliberate, valid protocol** — it models deploying the best
  checkpoint and continuing from it. Do not describe it as a defect.
- It does not hide chain-level variance. Episode 1 trains from scratch for every mode, so its trial
  spread is genuinely independent; later-episode spread is **0.5–1.0×** that, so the common parent
  does not collapse diversity. And path dependence does not compound: across 40 (mode, task) cells
  spread is **flat in 33, shrinking in 5, growing in 2**, and neither grower survives Holm at 40
  tests. Aggregating the final increment by mean instead of best leaves the top mode unchanged on
  **6/6 admissible tasks** (Kendall τ = 0.976); the sole exception is driver-top3, which has one
  training episode and is excluded by the ≥5-episode rule anyway.

Note `driver-top3` has **1** training episode. Adopt the ≥5-episode inclusion rule and report mean
ranks with and without it — dropping it reorders `ewc` above `er`.

---

## REQ-009 [detail] `ft_upsample` correction — **ANSWERED 2026-09-17, and it is worse than the plan said**

The regime described in §5.2 and Appendix A.1 of the submission **was never the regime that ran.**
`ComposedLoader(mode="rnd_uni")` mixes *proportionally* to the constituent loader sizes, not at the
claimed fixed 50/50, so the realised new-increment share decays to **5–10% by mid-chain** as history
grows. Source: `analysis/upsampling-ratio-finding.md`.

`ft_upsample` is **dropped entirely** from the new grid and superseded by `er`, which has an explicit
`replay_ratio` and a bounded buffer. Two consequences for the text:

1. State it as an explicit correction of the submitted paper, not a silent substitution.
2. It is also the honest answer to **CMu9's Q4**, where the rebuttal promised a sweep over the mixing
   ratio: a sweep over a parameter the loader ignores would be meaningless. Say that plainly.

While you are writing corrections, there is a second one you do not yet have: **~47% of the submitted
paper's checkpoints are NaN-poisoned** — 21 of 45 sampled carry non-finite weights in numerical column
encoders (`analysis/published-checkpoint-audit.md`). Its own conclusion is that the published numbers
should be described as "the previous protocol" rather than compared cell-for-cell. **Never compare a
new number cell-for-cell against a submitted one**, and use this as the honest reason the headline
changed.

---

## REQ-004 [detail] T1 inclusion rule — **ANSWERED 2026-09-17**

Quotable rule:

> A (database, task) pair enters the benchmark if its task is a pointwise entity task — excluding
> recommendation tasks, which use two-tower/IDGNN heads and ranking metrics the protocol's decay
> metric does not accept, and autocomplete tasks, which are not `EntityTask`s — and if, at the
> dataset's native update interval ΔI, `ContinuousWrapper.get_splits()` yields at least five training
> episodes after dropping any window holding under 10% of the validation window's row count. ΔI is
> the dataset's own RelBench validation span (`test_timestamp − val_timestamp`) and ΔW is the task's
> own `timedelta`; neither is chosen by us.

Totals: **24 usable tasks across 8 databases** measured; structurally excluded are **13
recommendation** and **11 autocomplete** tasks. Below the five-episode line: rel-arxiv (3 each),
rel-avito (2 each), rel-f1/driver-top3 (1). Longest chain in the corpus is rel-hm at **52** episodes.
Source: `analysis/dataset-usability.{json,csv}`.

Add one sentence naming **multiclass as a protocol limitation** — rel-arxiv/author-category would add
a task type but multiclass is unsupported in six places in the pipeline, so it is excluded for an
implementation reason, not a scientific one. Say so; it is cheaper than being asked.

---

## REQ-012 [experiment] Budget accounting — **PARTIALLY ANSWERED 2026-09-17; part 2 is queued**

**Part 1 is worse than you framed it, and it is not about early stopping.** There is no early
stopping: `n_train_batches=2000`, `n_val_passes=20`, no early stop, on all 3,086 finished runs. The
budget is 2000 steps × 128 = **256,000 samples per episode regardless of training-set size**. So the
starvation is not "from_scratch restarts" — it is that full-history regimes see a shrinking *fraction*
of their own data as history grows. On rel-stack/user-badge `naive` gets **1.33 passes** over its own
training set while `joint` and `from_scratch` get **0.07** — a 19× gap that widens as ~1/i, along the
very episode axis the paper studies.

This **cuts against the paper's null** and must be disclosed, not argued away. Report the budget as a
**constraint of the study, not a control**. Add a passes-over-own-training-set column to every results
table (Z8) — that is what defuses the objection.

Do not write the reassuring version. The claim that confound severity is uncorrelated with the
joint-over-scratch gap is **refuted** (n=9, p=0.87, and it inverts on one point deletion).

**Part 2 is now buildable and queued.** `limit_train_batches` was hardcoded to 100 in `param_space`
and `max_epochs` was read from config but never put there, so the epoch regime could not be requested
at all. Both are now parameters (commit `8a2c9e5`), defaulting to exactly what the 3,086 logged runs
used. The arm is rel-trial study-outcome + site-success, epoch-based, {from_scratch, naive, joint} × 3
trials, ~35 GPU-h. It is behind 37 queued jobs. **Write the section with a `\TODO{}` for the number
and phrase it so it does not presuppose the direction.**

---

## REQ-005 [number] α-sweep — **IN PROGRESS**

Not yet computed here. Two things to know before you write §8:

1. The published `final_model_decay_avg` is **not R-derivable** — reproducing it needs the 42.9 GB of
   prediction CSVs on RCI. It is cheap but it is a batch job, not a notebook cell.
2. There are **three** decay summaries and they disagree: flip counts are forward-anchored **9/81
   (11.1%)**, deployed-diagonal **34/81**, frozen-first-model **40/81**. **Report all three**, not the
   most favourable. α is not numerically comparable between per-timestamp and per-episode decay.

NHfu asked for the claim at a *limited* level: the metric **can** produce a different ranking from
uniform averaging, which is weaker than showing it generally adds information. Write it at that level.
Under a 10-point sweep with 8 modes, ranking is largely α-insensitive with **driver-position the one
exception** — which independently reproduces the submission's own §5.5 finding.

---

## REQ-008 [number] Cost table and steps-to-95/99% — **IN PROGRESS**

Measured totals available now: rel-f1 30.9 train / 0.8 predict, rel-trial 54.7 / 5.3, rel-stack
109.9 / **201.4** GPU-h. Scoring is O(episodes²) and dominates long chains.

One warning before you reuse the few-shot result: **"competitive after 100 training steps" is the
measurement floor, not a finding.** `val_check_interval=100`, so step 100 is the first observation and
five of eight modes are censored at it. CMu9 cited this as a strength; requalify it as a
lower-bounded ratio rather than repeating it. Warm-starting is still a real win —`joint` beats
`from_scratch` on 8/9 tasks and the two configs differ in exactly one flag.

---

## REQ-010 [detail] ROLAND / prior-art framing — **DEFERRED TO THE AUTHORS**

I have not read ROLAND (arXiv:2208.07239) or DRIFT (arXiv:2605.12998) and will not adjudicate a
novelty claim from a summary table. Escalated to the authors. Write the paragraph from your
differentiation table meanwhile and flag it for their review.

Independent of that, two citation debts are on the rebuttal record and both reviewers said they
resolve outright: **Gama et al. 2013** and **Hidalgo et al. 2019** on prequential fading factors. The
distinction to draw is direction: fading factors discount *past* observations in an open-ended stream
to estimate current accuracy; our metric discounts *future* evaluation windows over a fixed known
horizon. Same mechanism, opposite direction.

---

## REQ-001 / REQ-002 / REQ-011 — **BLOCKED ON THE AUTHORS**

**REQ-002 is unblocked and done:** `paper/NUMBERS.json` (132 provenanced entries) and
`paper/MANIFEST.md` exist now in the repo.

**REQ-001** — there is no `iclr 2027/` directory and no README naming drop locations on this machine,
so I cannot confirm a transfer route. Escalated. Figures will be emitted as **PDF and PNG both**; the
figure layer does not exist yet (the whole new analysis stack has zero plotting code — all submitted
figures came from a notebook targeting the *old* 4-regime experiment), so it is a build item, not a
render.

**REQ-011** — escalated, and you are right to raise it. Logged runs carry absolute cluster paths
(`/home/pelesjak/...`, `claude-redelex`) and `model_save_dir` values pointing at a personal tree.
Nothing has been anonymised yet and no anonymous mirror exists. This needs an explicit scrub step
before any bundle ships. Treat it as unresolved.

---

## REQ-014 [detail] Abstract review — **ANSWERED 2026-09-17, authors' direction. Read before redrafting.**

The WIP abstract was reviewed by the authors. Two structural changes and three line-level fixes.

### 1. Delete the backward-transfer sentence. It is the refuted claim, restated.

> ~~"Retention is therefore not the bottleneck one would expect: we measure backward transfer directly
> and find it non-negative even for updates that revisit no history at all."~~

`from_scratch` reinitialises at every episode, has no parameter continuity, **cannot forget by
construction** — and scores non-negative BWT on 9 of 9 tasks (`bwt.nonnegative_task_count.*`). So does
`naive`. The sentence is true of a baseline that cannot exhibit the property it is offered as evidence
for. See REQ-003. **No replacement forgetting number.**

### 2. Keep the reconstructibility argument — it is the real lead, and it is analytic.

The abstract's other structural claim survives, and it is stronger than the one being cut, because it
is verifiable from the protocol definition rather than from a metric:

> Bounded-memory continual learning exists to approximate a past you can no longer access. In
> append-only relational data with deterministic temporal masking, the past training set is always
> exactly reconstructible. The bounded-memory family is therefore unmotivated in this setting — not
> because forgetting does not occur, but because the problem it solves does not arise.

Note what this does **not** claim: nothing about whether forgetting happens. It says the *remedy* is
unmotivated when the data is still there. That is enough to explain the negative result, and it
survives a reviewer who recomputes BWT.

It also names the real constraint: **not memory, but compute.** You cannot retrain on an unbounded
history at every update.

### 3. The value of continual updating GROWS with history — measured, and this is the positive finding.

Requested by the authors and now computed (`warmstart_margin.*` in NUMBERS.json). Diagonal margin
`R[i,i]` of a warm-started mode over `from_scratch`, regressed on episode index:

| task | early | late | slope/ep | p | trend |
|---|---|---|---|---|---|
| rel-stack/user-engagement | **−0.0008** | **+0.0029** | +0.00028 | 0.000 | **grows** |
| rel-stack/user-badge | **−0.0019** | **+0.0078** | +0.00076 | 0.002 | **grows** |
| rel-stack/post-votes | +0.0012 | +0.0023 | +0.00009 | 0.001 | **grows** |
| rel-trial/study-adverse | +0.198 | +1.857 | +0.309 | 0.008 | **grows** |
| rel-f1/driver-position, rel-f1/driver-dnf, rel-trial/site-success | — | — | — | — | flat |
| rel-trial/study-outcome | — | — | −0.0045 | 0.045 | shrinks |

(`joint` shown; `naive` gives the same verdict on the same 4/8 tasks.)

**On the large tasks the margin starts negative and becomes positive.** Early on, retraining is as
good or better; the warm-start advantage *emerges* as history accumulates. And it grows exactly where
the fixed budget stops covering that history — `from_scratch` falls to 0.07 passes over its own
training set by late rel-stack episodes while `naive` gets 1.33. The growth curve and the starvation
curve are the same curve, which is why §9 and this claim must be written together.

State it as scoped: grows on 4 of 8 tasks, and name which — the three rel-stack tasks (17 episodes,
millions of rows) and study-adverse. It is flat on the small rel-f1 tasks, where history never
accumulates enough to matter. That scoping is the finding, not a weakness of it.

### 4. Separate the two comparisons. The abstract currently blurs them.

- **Warm-start vs from-scratch** — budget-*asymmetric*: by episode *i* a warm-started model carries
  *i*×2000 cumulative steps against from_scratch's 2000. Any claim here must ship with the budget
  disclosed, or a reviewer states the asymmetry for us.
- **CL machinery vs plain continued training** (`er`/`der_pp`/`ewc`/`lwf`/`freeze_extend` versus
  `joint`/`naive`) — budget-*clean*: same warm start, same per-episode budget, no compute asymmetry.
  **The negative result belongs here and only here.** Scope it as "adds nothing over simply continuing
  to train on retained data", never as "continual learning does not work".

### 5. Requalify the hundred-step figure.

> "reaching that accuracy within a small fraction of the budget—on the order of a hundred optimization
> steps"

100 is `val_check_interval` — the *first observation point*. Five of eight modes are already at
97–99% of final validation skill when first observed, so the true figure is unknown and 100 is a
**measurement floor**. CMu9 cited it as a strength, so it cannot simply be repeated. Write it as a
lower bound: warm-started modes are within a few percent of final skill by the first checkpoint, at or
before step 100.

### 6. Recommended arc

1. Data is retained and exactly reconstructible → memory-based CL is unmotivated here
2. So the binding constraint is the **update budget**, not retention
3. Under a fixed per-episode budget, continuing beats restarting — and the margin **grows with
   history**, on the tasks where history accumulates
4. The bounded-memory machinery adds nothing over plain continued training (budget-clean comparison)
5. Open problem: how to spend a bounded update budget on an unbounded history

This makes the negative results supporting evidence rather than the headline, which also answers
d22i's W2 ("what work is there still to do?") with a named open problem instead of a flat ranking.
Item C — the epoch-budget arm, queued — is what turns step 3 from assertable into defensible.
