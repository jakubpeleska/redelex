import numpy as np
import pandas as pd
from relbench.base import Database, Table, TaskType

from .base import BaseTask


class StaticTaskMixin(BaseTask):
    r"""Mixin class for a static task on a dataset.

    Attributes:
        sampling_table: The table from which to sample entities for the task.

    Other attributes are inherited from BaseTask.
    """

    sampling_table: str
    task_type: TaskType

    def make_table(self, db: Database, indices: pd.Index) -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for ground truth.
            indices: Data used to defining entity range for the table.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def make_split_range(self, db: Database, split: str) -> pd.Series:
        r"""Make a indices range for a split.

        Args:
            db: The database object to use.
            split: The split to be made.
        Returns:
            The indices of the split.
        """
        random_state = np.random.RandomState(seed=42)
        sampling_df = db.table_dict[self.sampling_table].df
        train_df = sampling_df.sample(frac=0.8, random_state=random_state)

        if split == "train":
            sampling_df = train_df
        else:
            sampling_df = sampling_df.drop(train_df.index)
            val_df = sampling_df.sample(frac=0.5, random_state=random_state)
            sampling_df = val_df if split == "val" else sampling_df.drop(val_df.index)

        return sampling_df.index


__all__ = ["StaticTaskMixin"]
