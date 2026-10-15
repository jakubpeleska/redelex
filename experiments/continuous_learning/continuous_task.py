from typing import Literal, Optional

import numpy as np
import pandas as pd

from relbench.base import Database, EntityTask, Table

from redelex.continual.anchors import HOLDOUT_COL, ROW_COL, make_anchor_table, with_row_ids
from redelex.continual.protocol import entity_holdout_mask, trainable_mask

Part = Literal["all", "train", "holdout"]


class ContinuousWrapper:
    def __init__(self, task: EntityTask):
        self.task = task

        self.train_table = self.task.get_table("train", mask_input_cols=False)
        self.val_table = self.task.get_table("val", mask_input_cols=False)
        self.test_table = self.task.get_table("test", mask_input_cols=False)

        df = (
            pd.concat([self.train_table.df, self.val_table.df, self.test_table.df], ignore_index=True)
            .sort_values(self.train_table.time_col)
            .reset_index(drop=True)
        )

        self.full_table = Table(
            df=df,
            fkey_col_to_pkey_table=self.train_table.fkey_col_to_pkey_table,
            pkey_col=self.train_table.pkey_col,
            time_col=self.train_table.time_col,
        )

    @classmethod
    def from_anchors(
        cls,
        task: EntityTask,
        db: Database,
        timestamps,
        holdout_frac: float,
        holdout_salt: str,
        cache_path: Optional[str] = None,
    ) -> "ContinuousWrapper":
        """One task table over ``timestamps``, with a held-out entity flag per row.

        Unlike the default constructor this ignores RelBench's split tables:
        ``full_table`` is the task query evaluated at every given anchor, and
        rows of held-out entities are marked so that no training path reads them.
        """
        table = make_anchor_table(task, db, timestamps, cache_path=cache_path)
        holdout = entity_holdout_mask(table.df[task.entity_col].to_numpy(), holdout_frac, holdout_salt)
        self = cls.__new__(cls)
        self.task = task
        self.train_table = self.val_table = self.test_table = None
        self.full_table = with_row_ids(table, holdout)
        self.holdout_frac = holdout_frac
        self.holdout_salt = holdout_salt
        return self

    @property
    def has_holdout(self) -> bool:
        return HOLDOUT_COL in self.full_table.df

    def _part_mask(self, part: Part) -> np.ndarray:
        df = self.full_table.df
        if part == "all":
            return np.ones(len(df), dtype=bool)
        if not self.has_holdout:
            raise ValueError("this wrapper has no held-out entities; build it with from_anchors")
        holdout = df[HOLDOUT_COL].to_numpy()
        if part == "holdout":
            return holdout
        if part == "train":
            return ~holdout
        raise ValueError(f"unknown part {part!r}")

    def _subset(self, mask: np.ndarray) -> Table:
        return Table(
            df=self.full_table.df[mask].reset_index(drop=True),
            fkey_col_to_pkey_table=self.full_table.fkey_col_to_pkey_table,
            pkey_col=self.full_table.pkey_col,
            time_col=self.full_table.time_col,
        )

    def get_table(self, start: pd.Timestamp, end: pd.Timestamp = None, part: Part = "all"):
        if end is None:
            end = self.full_table.max_timestamp + pd.Timedelta(days=1)
        times = self.full_table.df[self.full_table.time_col]
        mask = (times >= start) & (times < end)
        return self._subset(mask.to_numpy() & self._part_mask(part))

    def trainable(
        self,
        update_time: pd.Timestamp,
        after: Optional[pd.Timestamp] = None,
        part: Part = "train",
    ) -> Table:
        """Rows whose label is complete at ``update_time`` (``t <= T - Δw``).

        ``after`` keeps only rows newer than it, e.g. the previous update's
        labelled cutoff for "newly labelled". ``part`` defaults to the trainable
        entities: held-out rows are never returned unless asked for by name.
        """
        times = self.full_table.df[self.full_table.time_col]
        mask = trainable_mask(times, update_time, self.task.timedelta, after=after)
        return self._subset(mask & self._part_mask(part))

    def get_rows(self, row_ids, part: Part = "train") -> Table:
        """Rows by their ``__row__`` id, still restricted to ``part``."""
        ids = self.full_table.df[ROW_COL].isin(np.asarray(row_ids)).to_numpy()
        return self._subset(ids & self._part_mask(part))

    def get_splits(self, val_delta: pd.Timedelta = None):
        if val_delta is None:
            val_delta = (
                self.task.dataset.test_timestamp - self.task.dataset.val_timestamp
            )

        timestamps = self.full_table.df[self.full_table.time_col].unique()

        splits = [self.task.dataset.test_timestamp, self.task.dataset.val_timestamp]

        previous_timestamp = self.task.dataset.val_timestamp
        for timestamp in reversed(timestamps):
            if timestamp + val_delta <= previous_timestamp:

                splits.append(timestamp)
                previous_timestamp = timestamp

        splits.append(timestamps[0])
        splits.reverse()

        min_split_len = int(
            0.1
            * len(
                self.get_table(
                    start=self.task.dataset.val_timestamp,
                    end=self.task.dataset.test_timestamp,
                )
            )
        )
        filtered_splits = [splits[0]]
        for split in splits[1:]:
            split_len = len(self.get_table(start=filtered_splits[-1], end=split))
            if split_len >= min_split_len:
                filtered_splits.append(split)

        return filtered_splits


def assert_no_holdout(table: Table) -> None:
    """Fail loudly if a table about to be trained on holds any held-out row."""
    if HOLDOUT_COL in table.df and table.df[HOLDOUT_COL].any():
        n = int(table.df[HOLDOUT_COL].sum())
        raise AssertionError(f"{n} held-out rows reached a training table")


__all__ = ["ContinuousWrapper", "assert_no_holdout", "HOLDOUT_COL", "ROW_COL"]
