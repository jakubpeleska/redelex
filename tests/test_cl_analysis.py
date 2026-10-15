import numpy as np
import pandas as pd
import pytest

from redelex.continual import analysis as A


def _R(values):
    """Oriented R from a dict {(e, j): value}."""
    s = pd.Series(values)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["model", "window"])
    return s.unstack("window")


def test_orient_flips_mae_only():
    assert A.orient([0.2], "mae")[0] == -0.2
    assert A.orient([0.7], "roc_auc")[0] == 0.7


def test_score_long_and_matrix_follow_the_convention():
    rows = pd.DataFrame(
        {
            "__row__": np.arange(8),
            "entity": [0, 1, 2, 3, 0, 1, 2, 3],
            "time": pd.date_range("2020-01-01", periods=8, freq="D"),
            "target": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            "window": [1, 1, 1, 1, 2, 2, 2, 2],
            "holdout": [True, False, False, False, True, False, False, False],
        }
    )
    preds = {(0, 1): pd.DataFrame({"__row__": np.arange(8), "pred": np.ones(8)})}

    long = A.score_long(rows, preds, "mae")
    R = A.matrix(long, "mae")
    Rh = A.matrix(long, "mae", subset="holdout")

    assert R.at[1, 1] == pytest.approx(-1.5)  # MAE 1.5 on window 1, oriented
    assert R.at[1, 2] == pytest.approx(0.0)
    assert Rh.at[1, 1] == pytest.approx(0.0)  # held-out row 0: target 1, pred 1
    assert A.next_window(R).to_dict() == {1: 0.0}


def test_staleness_penalty_compares_models_on_the_same_window():
    # Window difficulty varies (j=3 hard), but the fresher model is always 0.1 better.
    R = _R({(1, 2): 0.80, (2, 3): 0.60, (1, 3): 0.50, (3, 4): 0.90, (2, 4): 0.80, (1, 4): 0.70})

    s1, per = A.staleness_penalty(R, 1)
    s2, _ = A.staleness_penalty(R, 2)

    assert s1 == pytest.approx(0.1) and sorted(per.index) == [3, 4]
    assert s2 == pytest.approx(0.2)
    with pytest.raises(ValueError):
        A.staleness_penalty(R, 0)


def test_holdout_bwt():
    Rh = _R({(1, 1): 0.8, (2, 1): 0.7, (3, 1): 0.6, (2, 2): 0.9, (3, 2): 0.9, (3, 3): 0.5})

    # Later minus own-update score: (0.7-0.8), (0.6-0.8), (0.9-0.9).
    assert A.holdout_bwt(Rh) == pytest.approx(np.mean([-0.1, -0.2, 0.0]))
    assert A.holdout_bwt(Rh, final_only=True) == pytest.approx(np.mean([-0.2, 0.0]))


def test_anytime_auc_holds_the_last_value():
    flat = A.anytime_auc([0, 10, 100], [0.5, 0.5, 0.5], horizon=1000)
    rising = A.anytime_auc([0, 10], [0.0, 1.0], horizon=10)

    assert flat == pytest.approx(0.5)
    assert rising == pytest.approx(0.5)  # linear in log1p(step) from 0 to 1
    assert A.anytime_auc([0, 10], [0.0, 1.0], horizon=1000) > rising


def test_steps_to_target_and_scratch_target():
    assert A.scratch_target([0.8, 0.9, 1.0]) == pytest.approx(0.9 - 0.1)
    assert A.steps_to_target([0, 10, 25, 50], [0.1, 0.5, 0.85, 0.9], 0.8) == 25
    assert A.steps_to_target([0, 10], [0.1, 0.2], 0.8) == float("inf")


def test_horizon_cost_sums_episodes():
    eps = [
        {"cost": {"steps": 10, "samples": 1280, "train_seconds": 1.5, "seconds": {"val": 0.5}}},
        {"cost": {"steps": 20, "samples": 2560, "train_seconds": 2.5, "seconds": {"val": 1.0}}},
    ]
    assert A.horizon_cost(eps) == {"steps": 30, "samples": 3840, "train_seconds": 4.0, "val_seconds": 1.5, "updates": 2}


def test_timeline_uses_the_latest_model_before_each_anchor():
    t = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01"])
    rows = pd.DataFrame(
        {"__row__": [0, 1, 2, 3], "entity": [0, 0, 0, 1], "time": t, "target": [1.0, 1.0, 1.0, 3.0],
         "window": [1, 2, 3, 3], "holdout": False}
    )
    # Model 1 is deployed on window 2, model 2 on window 3; model 1's window-3 preds are ignored.
    preds = {
        1: pd.DataFrame({"__row__": [1, 2, 3], "pred": [1.0, 9.0, 9.0]}),
        2: pd.DataFrame({"__row__": [2, 3], "pred": [1.0, 1.0]}),
    }

    s = A.timeline_scores(rows, preds, "mae")

    assert list(s.index) == list(pd.to_datetime(["2020-02-01", "2020-03-01"]))
    assert s.tolist() == pytest.approx([0.0, -1.0])


def test_block_bootstrap_brackets_the_mean():
    v = pd.Series(np.r_[np.zeros(50), np.ones(50)])
    mean, lo, hi = A.block_bootstrap_ci(v, block=10, n_boot=500)

    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi


def test_pareto_front():
    cost = [1, 2, 3, 2]
    score = [0.5, 0.7, 0.6, 0.7]
    assert A.pareto_front(cost, score).tolist() == [True, True, False, True]


def test_ranks_and_tests():
    table = pd.DataFrame(
        {"a": [0.9, 0.8, 0.7, 0.9, 0.85], "b": [0.8, 0.7, 0.6, 0.8, 0.75], "c": [0.7, 0.6, 0.5, 0.7, 0.65]}
    )

    ranks = A.average_ranks(table)
    stat, p = A.friedman(table)
    w = A.wilcoxon_holm(table, baseline="a")

    assert ranks.to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert p < 0.05
    assert set(w.config) == {"b", "c"} and (w.p_holm >= w.p).all() and (w.median_diff < 0).all()
