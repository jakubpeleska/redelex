from typing import Optional

import pandas as pd

from relbench.base import Database, Dataset, Table, TaskType

from .base import BaseTask


class TemporalTaskMixin(BaseTask):
    r"""Mixin class for a temporal task on a dataset.

    Attributes:
        timedelta: The prediction task at `timestamp` is over the time window
            (timestamp, timestamp + timedelta].
        num_eval_timestamps: The number of evaluation time windows. e.g., test
            time windows are (test_timestamp, test_timestamp + timedelta] ...
            (test_timestamp + (num_eval_timestamps - 1) * timedelta, test_timestamp
            + num_eval_timestamps * timedelta].
        Other attributes are inherited from BaseTask.
    """

    # To be set by subclass.
    timedelta: pd.Timedelta
    num_eval_timestamps: int = 1
    task_type: TaskType

    def __init__(self, dataset: Dataset, cache_dir: Optional[str] = None):
        r"""Create a task object.

        Args:
            dataset: The dataset object on which the task is defined.
            cache_dir: A directory for caching the task table objects. If specified,
                we will either process and cache the file (if not available) or use
                the cached file. If None, we will not use cached file and re-process
                everything from scratch without saving the cache.
        """
        super().__init__(dataset=dataset, cache_dir=cache_dir)

        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {self.timedelta}, time "
                f"diff: {time_diff})."
            )

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for (historical) ground truth.
            timestamps: Collection of timestamps to compute labels for. A label can be
            computed for a timestamp using historical data
            upto this timestamp in the database.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def make_split_range(self, db: Database, split: str) -> "pd.Series[pd.Timestamp]":
        r"""Helper function to get a range for a split.
        Args:
            db: The database object to use.
            split: The split to be made.
        Returns:
            A timestamps defining range of the split.
        """

        if split == "train":
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
            freq = -self.timedelta

        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.val_timestamp
            end = min(
                self.dataset.val_timestamp
                + self.timedelta * (self.num_eval_timestamps - 1),
                self.dataset.test_timestamp - self.timedelta,
            )
            freq = self.timedelta

        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp
            end = min(
                self.dataset.test_timestamp
                + self.timedelta * (self.num_eval_timestamps - 1),
                db.max_timestamp - self.timedelta,
            )
            freq = self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. ({len(timestamps)} given)"
            )

        return timestamps


__all__ = ["TemporalTaskMixin"]
