import numpy as np
import pandas as pd
import pytest

from experiments.continuous_learning.continuous_task import ContinuousWrapper, assert_no_holdout
from redelex.continual.anchors import (
    HOLDOUT_COL,
    ROW_COL,
    grid_anchor_timestamps,
    labelled_anchor_range,
    make_anchor_table,
    native_anchor_timestamps,
)

from .helpers import StreamChurnTask, StreamCountTask, StreamDataset


def _sorted(df, task):
    return df.sort_values([task.time_col, task.entity_col], kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("task_cls", [StreamChurnTask, StreamCountTask])
@pytest.mark.parametrize("split", ["train", "val"])
def test_native_timestamps_reproduce_native_tables(task_cls, split):
    task = task_cls(StreamDataset())
    db = task.dataset.get_db(upto_test_timestamp=True)

    ours = make_anchor_table(task, db, native_anchor_timestamps(task, db, split)).df
    native = _sorted(task.get_table(split, mask_input_cols=False).df, task)

    pd.testing.assert_frame_equal(ours, native, check_dtype=False)


def test_test_split_differs_only_by_entities_relbench_counts_as_dangling():
    """RelBench drops test rows of entities created after test_timestamp,
    because it counts entities against the truncated database. A uniform
    timeline keeps them; nothing else differs."""
    task = StreamChurnTask(StreamDataset())
    full = task.dataset.get_db(upto_test_timestamp=False)
    truncated_users = len(task.dataset.get_db().table_dict["users"])

    ours = make_anchor_table(task, full, native_anchor_timestamps(task, full, "test")).df
    native = _sorted(task.get_table("test", mask_input_cols=False).df, task)

    kept_by_relbench = ours[ours.user < truncated_users].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept_by_relbench, native, check_dtype=False)


def test_rows_are_sorted_and_cached(tmp_path):
    task = StreamChurnTask(StreamDataset())
    db = task.dataset.get_db(upto_test_timestamp=False)
    ts = native_anchor_timestamps(task, db, "test")
    path = tmp_path / "anchors.parquet"

    first = make_anchor_table(task, db, ts, cache_path=path).df
    assert path.exists()
    again = make_anchor_table(task, db, ts[:1], cache_path=path).df  # served from cache

    assert first.equals(_sorted(first, task))
    pd.testing.assert_frame_equal(first, again)


def test_dense_grid_contains_native_anchors_and_phases_are_disjoint():
    task = StreamChurnTask(StreamDataset())
    db = task.dataset.get_db(upto_test_timestamp=False)
    start, end = labelled_anchor_range(task, db)
    ref, dw = task.dataset.val_timestamp, task.timedelta

    native = grid_anchor_timestamps(ref, dw, start, end)
    dense = grid_anchor_timestamps(ref, dw / 4, start, end)
    shifted = grid_anchor_timestamps(ref, dw, start, end, phase=dw / 2)

    assert set(native) <= set(dense)
    assert set(native).isdisjoint(shifted)
    assert set(native_anchor_timestamps(task, db, "train")) <= set(native)
    assert native.max() <= end and native.min() >= start


def _wrapper(frac=0.2):
    task = StreamChurnTask(StreamDataset())
    db = task.dataset.get_db(upto_test_timestamp=False)
    start, end = labelled_anchor_range(task, db)
    ts = grid_anchor_timestamps(task.dataset.val_timestamp, task.timedelta, start, end)
    return task, ContinuousWrapper.from_anchors(task, db, ts, frac, "stream/users/v1")


def test_holdout_is_by_entity_and_partitions_rows():
    task, w = _wrapper()
    df = w.full_table.df

    per_entity = df.groupby(task.entity_col)[HOLDOUT_COL].nunique()
    assert (per_entity == 1).all()  # an entity is held out in every row or none
    assert 0.1 < df.groupby(task.entity_col)[HOLDOUT_COL].first().mean() < 0.3
    assert (df[ROW_COL].to_numpy() == np.arange(len(df))).all()

    lo, hi = df[task.time_col].min(), df[task.time_col].max() + pd.Timedelta(days=1)
    train = w.get_table(lo, hi, part="train").df
    hold = w.get_table(lo, hi, part="holdout").df
    assert len(train) + len(hold) == len(df)
    assert set(train[ROW_COL]).isdisjoint(hold[ROW_COL])


def test_trainable_applies_label_delay_and_excludes_holdout():
    task, w = _wrapper()
    T = pd.Timestamp("2021-01-01")

    rows = w.trainable(T).df
    new = w.trainable(T, after=T - pd.Timedelta(days=120)).df

    assert rows[task.time_col].max() <= T - task.timedelta
    assert not rows[HOLDOUT_COL].any()
    assert new[task.time_col].min() > T - pd.Timedelta(days=120)
    assert set(new[ROW_COL]) <= set(rows[ROW_COL])
    assert w.trainable(T, part="holdout").df[HOLDOUT_COL].all()


def test_assert_no_holdout_guards_training_tables():
    _, w = _wrapper()
    T = pd.Timestamp("2021-01-01")

    assert_no_holdout(w.trainable(T))
    with pytest.raises(AssertionError, match="held-out rows"):
        assert_no_holdout(w.trainable(T, part="all"))
