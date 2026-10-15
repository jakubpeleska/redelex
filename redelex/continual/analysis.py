r"""Metrics of the rerun, computed from inline predictions.

Conventions (see :mod:`redelex.continual.protocol`):

* ``R[e, j]``: model updated at ``T_e`` scored on window ``j`` (1-based). The
  next-window score is ``R[e, e + 1]``. Windows ``<= e`` were (mostly) trained
  on, so on them only the held-out entities' rows are scored.
* Every matrix handed to the summary functions is **oriented**: higher is
  better. :func:`orient` flips MAE to ``-MAE``; AUROC is left as is.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"mae", "mse", "rmse"}


def orient(score, metric: str):
    """Higher-is-better version of a score."""
    return -np.asarray(score, dtype=float) if metric in LOWER_IS_BETTER else np.asarray(score, dtype=float)


def _metric(metric: str):
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score

        def f(y, p):
            return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

        return f
    if metric == "mae":
        return lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p)))) if len(y) else float("nan")
    raise ValueError(f"unknown metric {metric!r}")


def score_long(
    rows: pd.DataFrame,
    preds: Mapping[tuple, pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    """Scores of every (seed, model, window, subset) present in ``preds``.

    ``rows``: the chain's ``rows.parquet``. ``preds``: ``{(seed, e): frame}``
    with columns ``__row__`` and ``pred``. Subsets: ``all``, ``holdout`` and
    ``trained`` (non-held-out) rows of each window.
    """
    fn = _metric(metric)
    out = []
    for (seed, e), frame in preds.items():
        df = frame.merge(rows, on="__row__", how="inner")
        df = df[df["window"] >= 1]
        for j, g in df.groupby("window"):
            for subset, mask in (
                ("all", np.ones(len(g), bool)),
                ("holdout", g["holdout"].to_numpy()),
                ("trained", ~g["holdout"].to_numpy()),
            ):
                h = g[mask]
                if len(h) == 0:
                    continue
                out.append(
                    {
                        "seed": seed,
                        "model": int(e),
                        "window": int(j),
                        "subset": subset,
                        "n": len(h),
                        "n_entities": int(h["entity"].nunique()),
                        "score": fn(h["target"].to_numpy(), h["pred"].to_numpy()),
                    }
                )
    return pd.DataFrame(out, columns=["seed", "model", "window", "subset", "n", "n_entities", "score"])


def matrix(long: pd.DataFrame, metric: str, subset: str = "all", seed=None) -> pd.DataFrame:
    """Oriented ``R`` (index: model e, columns: window j). Mean over seeds unless one is given."""
    df = long[long["subset"] == subset]
    if seed is not None:
        df = df[df["seed"] == seed]
    R = df.groupby(["model", "window"])["score"].mean().unstack("window").sort_index()
    return pd.DataFrame(orient(R.to_numpy(), metric), index=R.index, columns=R.columns)


def next_window(R: pd.DataFrame) -> pd.Series:
    """``R[e, e + 1]`` for every model e."""
    vals = {e: R.at[e, e + 1] for e in R.index if (e + 1) in R.columns}
    return pd.Series(vals, dtype=float).sort_index()


def staleness_penalty(R: pd.DataFrame, k: int) -> tuple[float, pd.Series]:
    """``S(k) = mean_j R[j-1, j] - R[j-1-k, j]`` on an oriented ``R``.

    Both models are scored on the same window ``j``, so window difficulty
    cancels; ``S(k) > 0`` means a model ``k`` updates older does worse. Returns
    the mean and the per-window differences.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    diffs = {}
    for j in R.columns:
        fresh, stale = j - 1, j - 1 - k
        if fresh in R.index and stale in R.index:
            a, b = R.at[fresh, j], R.at[stale, j]
            if np.isfinite(a) and np.isfinite(b):
                diffs[j] = a - b
    s = pd.Series(diffs, dtype=float)
    return (float(s.mean()) if len(s) else float("nan")), s


def holdout_bwt(R_h: pd.DataFrame, final_only: bool = False) -> float:
    """Backward transfer on held-out rows of past windows (oriented ``R``).

    Reference for window ``j`` is ``R_h[j, j]``, the model right after its
    update at ``T_j``. ``final_only``: compare the last model only (the
    classical definition); otherwise average over every later model.
    """
    diffs = []
    models = list(R_h.index)
    for j in R_h.columns:
        if j not in R_h.index or not np.isfinite(R_h.at[j, j]):
            continue
        later = [models[-1]] if final_only else [i for i in models if i > j]
        for i in later:
            if i > j and np.isfinite(R_h.at[i, j]):
                diffs.append(R_h.at[i, j] - R_h.at[j, j])
    return float(np.mean(diffs)) if diffs else float("nan")


def anytime_auc(steps: Sequence[int], scores: Sequence[float], horizon: int) -> float:
    """Mean oriented score over ``log1p(step)`` from 0 to ``horizon``.

    The curve is held at its last value after the update stops, so a run that
    converged early is not penalised for having no later points.
    """
    s = np.asarray(steps, dtype=float)
    v = np.asarray(scores, dtype=float)
    order = np.argsort(s)
    s, v = s[order], v[order]
    keep = s <= horizon
    s, v = s[keep], v[keep]
    if len(s) == 0:
        return float("nan")
    if s[-1] < horizon:
        s, v = np.append(s, horizon), np.append(v, v[-1])
    x = np.log1p(s)
    if x[-1] == x[0]:
        return float(v[0])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz
    return float(trapezoid(v, x) / (x[-1] - x[0]))


def scratch_target(scores_by_seed: Iterable[float]) -> float:
    """Converged scratch score minus one SD across seeds (oriented)."""
    v = np.asarray(list(scores_by_seed), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    return float(v.mean() - (v.std(ddof=1) if len(v) > 1 else 0.0))


def steps_to_target(steps: Sequence[int], scores: Sequence[float], target: float) -> float:
    """First step whose oriented score reaches ``target``; ``inf`` if never."""
    for step, score in sorted(zip(steps, scores)):
        if np.isfinite(score) and score >= target:
            return float(step)
    return float("inf")


def horizon_cost(episodes: Iterable[Mapping]) -> dict:
    """Total steps, samples and training seconds over a chain's ``episode.json`` records."""
    total = {"steps": 0, "samples": 0, "train_seconds": 0.0, "val_seconds": 0.0, "updates": 0}
    for ep in episodes:
        c = ep["cost"]
        total["steps"] += int(c["steps"])
        total["samples"] += int(c["samples"])
        total["train_seconds"] += float(c["train_seconds"])
        total["val_seconds"] += float(c["seconds"].get("val", 0.0))
        total["updates"] += 1
    return total


def timeline_scores(
    rows: pd.DataFrame,
    preds_by_update: Mapping[int, pd.DataFrame],
    metric: str,
    start: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Score at every anchor time, each scored by the latest model updated before it.

    The model deployed on window ``j`` is the one updated at ``T_{j-1}``, i.e.
    update ``j - 1``; anchors of window 1 have no deployed model and are
    dropped. Returns an oriented score per anchor timestamp (index: time).
    """
    fn = _metric(metric)
    parts = []
    for e, frame in preds_by_update.items():
        df = frame.merge(rows, on="__row__")
        parts.append(df[df["window"] == e + 1])
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts, ignore_index=True)
    if start is not None:
        df = df[df["time"] >= start]
    per_time = df.groupby("time").apply(lambda g: fn(g["target"].to_numpy(), g["pred"].to_numpy()), include_groups=False)
    return pd.Series(orient(per_time.to_numpy(), metric), index=per_time.index).sort_index()


def block_bootstrap_ci(
    values: pd.Series, block: int, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and a moving-block bootstrap interval for a time-ordered series.

    Neighbouring anchors share label windows, so their scores are correlated;
    blocks of ``block`` consecutive values keep that correlation in each resample.
    """
    v = np.asarray(values.dropna(), dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts_max = n - block + 1
    reps = []
    k = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max, size=k)
        sample = np.concatenate([v[s : s + block] for s in starts])[:n]
        reps.append(sample.mean())
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def pareto_front(cost: Sequence[float], score: Sequence[float]) -> np.ndarray:
    """Boolean mask of configurations no other one beats on both lower cost and higher score."""
    c = np.asarray(cost, dtype=float)
    s = np.asarray(score, dtype=float)
    front = np.ones(len(c), dtype=bool)
    for i in range(len(c)):
        dominated = (c <= c[i]) & (s >= s[i]) & ((c < c[i]) | (s > s[i]))
        front[i] = not dominated.any()
    return front


def average_ranks(table: pd.DataFrame) -> pd.Series:
    """Mean rank of each column (configuration) over rows (tasks); rank 1 = best.

    ``table`` holds oriented scores; ties share the average rank.
    """
    ranks = table.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman(table: pd.DataFrame) -> tuple[float, float]:
    """Friedman chi-square and p-value over tasks (rows) for configurations (columns)."""
    from scipy.stats import friedmanchisquare

    clean = table.dropna()
    stat, p = friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    return float(stat), float(p)


def wilcoxon_holm(table: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Paired Wilcoxon of every configuration against ``baseline``, Holm-adjusted."""
    from scipy.stats import wilcoxon

    out = []
    for c in table.columns:
        if c == baseline:
            continue
        pair = table[[c, baseline]].dropna()
        d = pair[c] - pair[baseline]
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        out.append({"config": c, "median_diff": float(d.median()), "n": len(d), "p": p})
    res = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    m = len(res)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(res["p"])])
    res["p_holm"] = adj
    return res


__all__ = [
    "anytime_auc",
    "average_ranks",
    "block_bootstrap_ci",
    "friedman",
    "holdout_bwt",
    "horizon_cost",
    "matrix",
    "next_window",
    "orient",
    "pareto_front",
    "scratch_target",
    "score_long",
    "staleness_penalty",
    "steps_to_target",
    "timeline_scores",
    "wilcoxon_holm",
]
