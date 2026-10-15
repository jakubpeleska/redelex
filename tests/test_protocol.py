import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from redelex.continual.protocol import (
    EpisodePlan,
    Protocol,
    entity_holdout_mask,
    grid_update_times,
    trainable_mask,
)

DW = pd.Timedelta(days=30)


def _plan():
    times = [pd.Timestamp("2020-01-01") + k * pd.Timedelta(days=90) for k in range(4)]
    return EpisodePlan(tuple(times), horizon_end=pd.Timestamp("2021-01-01"), delta_w=DW)


def test_label_delay_keeps_only_completed_labels():
    T = pd.Timestamp("2020-06-01")
    anchors = pd.Series(pd.to_datetime(["2020-04-01", "2020-05-02", "2020-05-03", "2020-06-01"]))

    mask = trainable_mask(anchors, T, DW)

    # 2020-05-02 is exactly T - 30 days: its label window (t, t + Δw] closes at T.
    assert mask.tolist() == [True, True, False, False]


def test_newly_labelled_rows_are_new_exactly_once():
    plan = _plan()
    anchors = pd.Series(pd.date_range("2019-06-01", "2020-12-31", freq="7D"))
    seen = np.zeros(len(anchors), dtype=int)

    for ep in plan.episodes:
        seen += trainable_mask(anchors, ep.update_time, DW, after=ep.new_after)

    labelled_by_last = trainable_mask(anchors, plan.update_times[-1], DW)
    assert (seen[labelled_by_last] == 1).all()
    assert (seen[~labelled_by_last] == 0).all()


def test_deploy_window_is_never_trainable():
    plan = _plan()
    anchors = pd.Series(pd.date_range("2019-06-01", "2020-12-31", freq="3D"))
    windows = plan.assign_windows(anchors)

    for ep in plan.episodes:
        train = trainable_mask(anchors, ep.update_time, DW)
        deploy = windows == ep.deploy_window
        assert deploy.any()
        assert not (train & deploy).any()


def test_window_grid_and_index_convention():
    plan = _plan()
    eps = plan.episodes

    assert plan.num_updates == 4 and plan.num_windows == 5
    assert plan.window_bounds(1) == (None, plan.update_times[0])
    assert plan.window_bounds(5) == (plan.update_times[-1], plan.horizon_end)
    # R[e, j]: model updated at T_e scored on window j; the next window is e + 1.
    assert [e.deploy_window for e in eps] == [2, 3, 4, 5]
    assert eps[0].deploy_span == plan.window_bounds(2)
    assert eps[-1].deploy_span == plan.window_bounds(5)

    anchors = pd.Series(pd.to_datetime(["2019-01-01", "2020-01-01", "2020-12-31", "2021-01-01"]))
    assert plan.assign_windows(anchors).tolist() == [1, 2, 5, 0]


def test_plan_rejects_bad_grids():
    t = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="increasing"):
        EpisodePlan((t, t), horizon_end=t + DW, delta_w=DW)
    with pytest.raises(ValueError, match="horizon_end"):
        EpisodePlan((t,), horizon_end=t, delta_w=DW)


def test_grid_update_times_through_reference():
    ref = pd.Timestamp("2020-01-01")
    di = pd.Timedelta(days=100)

    times = grid_update_times(ref, di, first=ref - 3 * di, last=ref + di)

    assert times == [ref - 2 * di, ref - di, ref, ref + di]
    assert grid_update_times(ref, None, first=ref - di, last=ref) == [ref]


def test_holdout_mask_fraction_and_stability():
    ids = np.arange(20_000)

    mask = entity_holdout_mask(ids, 0.1, salt="rel-f1/drivers/v1")

    assert abs(mask.mean() - 0.1) < 0.01
    # Same answer in a fresh interpreter: crc32, not Python's salted hash().
    code = (
        "import numpy as np;"
        "from redelex.continual.protocol import entity_holdout_mask;"
        "print(int(entity_holdout_mask(np.arange(20000), 0.1, 'rel-f1/drivers/v1').sum()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == int(mask.sum())


def test_holdout_mask_depends_on_salt_not_on_order():
    ids = np.arange(1000)
    a = entity_holdout_mask(ids, 0.2, salt="db/users/v1")

    assert (entity_holdout_mask(ids[::-1], 0.2, salt="db/users/v1") == a[::-1]).all()
    assert (entity_holdout_mask(ids, 0.2, salt="db/items/v1") != a).any()
    with pytest.raises(ValueError):
        entity_holdout_mask(ids, 1.0, salt="x")


def test_protocol_hash_tracks_every_field():
    base = Protocol("rel-f1", "driver-dnf", DW, pd.Timedelta(days=365), DW)

    assert base.hash() == Protocol("rel-f1", "driver-dnf", DW, pd.Timedelta(days=365), DW).hash()
    assert base.hash() != Protocol("rel-f1", "driver-dnf", DW, None, DW).hash()
    assert base.hash() != Protocol("rel-f1", "driver-dnf", DW, pd.Timedelta(days=365), DW, holdout_frac=0.2).hash()
