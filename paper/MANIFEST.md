# Artifact manifest — ICLR 2027 revision

Status vocabulary (handoff contract §9.2):

- **`READY`** — cite it, quote its numbers.
- **`PARTIAL:<detail>`** — usable; caption and text must state `<detail>` verbatim.
- **`PENDING:<when>`** — write prose with `\TODO{}`. Do not invent a value, and do not phrase the
  sentence so it presupposes the result's direction.
- **`BLOCKED:<reason>`** — move the claim to Limitations.

**No figure has been rendered yet.** The new analysis stack (`run_analysis.py`,
`build_evaluation_matrix.py`, `redelex/continual/drift.py`) contains zero plotting code and zero LaTeX
emission — every figure in the submitted paper came from a notebook targeting the *old* 4-regime
experiment, which does not map onto the eight current modes. The figure layer is a build item
(~2 days), not a render. Numbers below are nonetheless final where marked `DATA-READY`.

## Figures

| id | title | status | backs |
|---|---|---|---|
| F1 | Protocol schematic | `PENDING:figure-layer` | §3 |
| **F2** | **BWT tautology** — BWT by mode, `from_scratch` highlighted non-negative on 9/9 | `DATA-READY, PENDING:figure-layer` | §6 ★ headline |
| F3 | ER rank vs buffer coverage (log-x, 0.26%→100%) | `DATA-READY, PENDING:figure-layer` | §7, CMu9 Q4 |
| F4 | Budget starvation — passes over own data vs episode index | `DATA-READY, PENDING:figure-layer` | §9, REQ-012 |
| F5 | Recency gap `R[j,j] − R[0,j]` per task, Holm-corrected | `DATA-READY, PENDING:figure-layer` | §5, JDci W4 |
| F6 | Evaluation-matrix heatmaps, 2×4 by mode | `DATA-READY, PENDING:figure-layer` | §6 |
| F7 | α-sensitivity, all three decay summaries | `PENDING:Z5` | §8, NHfu W2 |
| F8 | ΔI ablation (measured) + staleness curves (labelled separately) | `PENDING:item-A-running` | §8, CMu9 gap #2 |
| F9 | Dead relations over episodes | `PENDING:Z9` | CMu9 gap #1 |
| F10 | Cost–accuracy Pareto | `PARTIAL:totals only, per-episode pending` | §10 |
| F11 | Backbone generality (GraphSAGE vs RelGNN) | `BLOCKED:not attempted this cycle` | Limitations |

## Tables

| id | title | status | backs |
|---|---|---|---|
| T1 | Protocol admissibility census | `DATA-READY` — rule in REQUESTS REQ-004 | §4, JDci W1/Q1 |
| **T2** | Main benchmark, 8 modes × 9 tasks | `PARTIAL:must rank on the operational quantity, not matrix-averaged ACC — see REQ-013` | §6 |
| T3 | Buffer coverage + reservoir composition | `PARTIAL:observational only; controlled sweep queued` | §7 |
| T4 | Protocol sensitivity (α, ΔI) | `PENDING:Z5 + item-A` | §8 |
| T5 | Parameter budget incl. dead-parameter fraction | `PENDING:Z9` | §6 |
| T6 | Passes over own training set, per mode per episode | `DATA-READY` | §9, defuses REQ-012 |

## Claims you may NOT make

Full detail in `REQUESTS.md`; the short list, because each one is individually fatal:

1. **Never claim we measured catastrophic forgetting.** BWT is tautological here — `from_scratch`,
   which cannot forget, scores non-negative BWT on 9/9 tasks. Ship the metric defect; ship **no**
   replacement forgetting number.
2. **Never claim conditional concept drift.** Recency and cumulative training volume are
   non-identified. The licensed claim is *non-stationarity that retraining recovers and that
   label-prior shift cannot explain*.
3. **Never write "seeds"** or any paired seed statistic. Trial slots are hash-sorted per increment.
   Say "trials"; label ± as within-increment trial spread.
4. **Never rank on matrix-averaged ACC without labelling it retention.** It and the operational
   quantity disagree on the winner for 9 of 9 tasks.
5. **Never call the fixed-grid staleness analysis a ΔI ablation.** The real one is item A, running.
6. **Never say the decay metric generally changes conclusions** — NHfu asked for the limited level.
7. **Never describe ER as replay at 100% buffer coverage** — there it *is* joint training.
8. **Never repeat "100 training steps" unqualified** — that is `val_check_interval`, the measurement
   floor, with 5 of 8 modes censored.
9. **Never compare cell-for-cell against submitted numbers** — that checkpoint tree is ~47%
   NaN-poisoned. Call it "the previous protocol".
10. **Never retract the cost/efficiency claim** — CMu9 accepted it and closed that gap.
11. **Never imply breadth parity with the submission** — 3 databases now vs 4 then, pending item B.

## Compute in flight (2026-09-17)

| item | jobs | GPU-h | lands |
|---|---|---|---|
| A — ΔI ablation, 6 arms | 18 | ~30 | F8, T4 |
| B — rel-ratebeer, 2 tasks × 8 modes | 16 | ~59 | breadth regression, JDci W1 |
| Buffer sweep — site-success, 3 capacities | 3 | ~6 | F3, T3 |
| C — epoch budget arm | queued behind | ~35 | F4, REQ-012 part 2 |
