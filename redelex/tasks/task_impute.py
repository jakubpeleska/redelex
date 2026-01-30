from typing import Optional

import pandas as pd

from relbench.base import Database, Dataset, Table, TaskType

from redelex.tasks.mixins import StaticTaskMixin, TemporalTaskMixin, ImputeEntityTaskMixin


class ImputeEntityStaticTask(StaticTaskMixin, ImputeEntityTaskMixin):
    r"""A target imputation static task on a dataset.

    Attributes are inherited from ImputeEntityTaskMixin and StaticTaskMixin.
    """

    removed_entity_cols: list[str] = []
    sampling_table: Optional[str] = None
    entity_col: str
    entity_table: str
    target_col: str
    task_type: TaskType

    def __init__(self, dataset: Dataset, cache_dir: Optional[str] = None):
        super().__init__(dataset, cache_dir=cache_dir)

        if self.sampling_table is None:
            self.sampling_table = self.entity_table

    def make_table(self, db: Database, indices: pd.Index) -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for ground truth.
            indices: Data used to defining entity range for the table.
        """

        entity_table = db.table_dict[self.entity_table]

        df = entity_table.df.loc[indices, [self.entity_col, self.target_col]].reset_index(
            drop=True
        )
        df[self.target_col] = (
            df[self.target_col].map(self.target_transform).astype(self.target_dtype)
        )

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=None,
        )


class ImputeEntityTemporalTask(TemporalTaskMixin, ImputeEntityTaskMixin):
    r"""A target imputation temporal task on a dataset.

    Attributes are inherited from ImputeEntityTaskMixin and TemporalTaskMixin.
    """

    removed_entity_cols: list[str] = []
    timedelta = pd.Timedelta(seconds=1)
    num_eval_timestamps: int = 1
    entity_col: str
    entity_table: str
    target_col: str
    task_type: TaskType

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for (historical) ground truth.
            timestamps: Collection of timestamps to compute labels for. A label can be
            computed for a timestamp using historical data
            upto this timestamp in the database.
        """

        entity_table = db.table_dict[self.entity_table]

        time_col = entity_table.time_col

        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()

        df = entity_table.df

        df = df[
            (df[time_col] >= min_timestamp) & (df[time_col] <= max_timestamp)
        ].reset_index(drop=True)

        df = df[[self.entity_col, time_col, self.target_col]]

        df[self.target_col] = (
            df[self.target_col].map(self.target_transform).astype(self.target_dtype)
        )

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=time_col,
        )

    def make_split_range(self, db: Database, split: str) -> "pd.Series[pd.Timestamp]":
        if split == "train":
            return pd.Series(
                [db.min_timestamp, self.dataset.val_timestamp - self.timedelta]
            )
        elif split == "val":
            return pd.Series(
                [self.dataset.val_timestamp, self.dataset.test_timestamp - self.timedelta]
            )
        else:
            return pd.Series([self.dataset.test_timestamp, db.max_timestamp])
