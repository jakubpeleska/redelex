r"""Task tables at arbitrary anchor timestamps.

RelBench builds a task's tables from three fixed anchor sets: train anchors at
``val_timestamp - k * Δw`` back to the start of the database, then only
``num_eval_timestamps`` anchors from ``val_timestamp`` and from
``test_timestamp``. That leaves gaps on the timeline (e.g. rel-f1 driver-dnf
has no anchors between its 40th validation anchor and the test timestamp).

A continual-learning timeline needs anchors on one uniform grid, so the tables
here are built directly from the task's own query, ``task.make_table``, at any
timestamps. :func:`make_anchor_table` reproduces the native tables exactly when
given the native timestamps (see the tests), so the only thing that changes is
which anchors exist.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table

ROW_COL = "__row__"
HOLDOUT_COL = "__holdout__"


def native_anchor_timestamps(task: EntityTask, db: Database, split: str) -> pd.DatetimeIndex:
    """The anchors RelBench uses for ``split``, mirroring ``BaseTask._get_table``.

    ``db`` must be the database RelBench would use for that split: truncated at
    the test timestamp for "train" and "val", the full one for "test".
    """
    ds, dw = task.dataset, task.timedelta
    if split == "train":
        return pd.date_range(start=ds.val_timestamp - dw, end=db.min_timestamp, freq=-dw)
    if split == "val":
        end = min(ds.val_timestamp + dw * (task.num_eval_timestamps - 1), ds.test_timestamp - dw)
        return pd.date_range(start=ds.val_timestamp, end=end, freq=dw)
    if split == "test":
        end = min(ds.test_timestamp + dw * (task.num_eval_timestamps - 1), db.max_timestamp - dw)
        return pd.date_range(start=ds.test_timestamp, end=end, freq=dw)
    raise ValueError(f"unknown split {split!r}")


def grid_anchor_timestamps(
    reference: pd.Timestamp,
    stride: pd.Timedelta,
    start: pd.Timestamp,
    end: pd.Timestamp,
    phase: pd.Timedelta = pd.Timedelta(0),
) -> pd.DatetimeIndex:
    """Every ``reference + phase + k * stride`` (integer ``k``) in ``[start, end]``."""
    if stride <= pd.Timedelta(0):
        raise ValueError(f"stride must be positive, got {stride}")
    origin = reference + phase
    k_lo = math.ceil((start - origin) / stride)
    k_hi = math.floor((end - origin) / stride)
    return pd.DatetimeIndex([origin + k * stride for k in range(k_lo, k_hi + 1)])


def labelled_anchor_range(task: EntityTask, db: Database) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest anchor whose label window lies inside ``db``."""
    return db.min_timestamp, db.max_timestamp - task.timedelta


def make_anchor_table(
    task: EntityTask,
    db: Database,
    timestamps,
    cache_path: Optional[str | Path] = None,
) -> Table:
    """The task table at ``timestamps``, sorted by (time, entity).

    Dangling entities are dropped against ``db`` itself (RelBench counts them
    against its own default database). With ``cache_path`` the table is stored
    as parquet and read back on the next call.
    """
    if cache_path is not None and Path(cache_path).exists():
        return Table.load(str(cache_path))

    ts = pd.Series(pd.to_datetime(pd.DatetimeIndex(timestamps)).unique()).sort_values()
    table = task.make_table(db, ts.reset_index(drop=True))
    df = table.df
    num_entities = len(db.table_dict[task.entity_table])
    df = df[df[task.entity_col] < num_entities]
    df = df.sort_values([task.time_col, task.entity_col], kind="stable").reset_index(drop=True)
    table = Table(
        df=df,
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.save(str(cache_path))
    return table


def with_row_ids(table: Table, holdout: np.ndarray) -> Table:
    """Attach a stable row id and the held-out flag as columns."""
    df = table.df.copy()
    df[ROW_COL] = np.arange(len(df), dtype=np.int64)
    df[HOLDOUT_COL] = np.asarray(holdout, dtype=bool)
    return Table(
        df=df,
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


__all__ = [
    "HOLDOUT_COL",
    "ROW_COL",
    "grid_anchor_timestamps",
    "labelled_anchor_range",
    "make_anchor_table",
    "native_anchor_timestamps",
    "with_row_ids",
]
