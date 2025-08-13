from typing import Callable, Dict, List, Optional, Union
from numpy.typing import NDArray

from copy import deepcopy

import pandas as pd

from torch_frame import stype
from torch_frame.data import stats

from relbench.base import BaseTask, Database, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    macro_f1,
    mae,
    micro_f1,
    mse,
    r2,
    roc_auc,
)

from redelex.datasets import CTUDataset

__all__ = ["CTUBaseEntityTask"]


class CTUBaseEntityTask(BaseTask):
    timedelta = pd.Timedelta(-1)
    dataset: CTUDataset

    entity_col = "__PK__"
    entity_table: Optional[str] = None
    target_col: str

    _stats = None

    @property
    def metrics(self) -> List[Callable[[NDArray, NDArray], float]]:
        if self.task_type == TaskType.REGRESSION:
            return [mae, mse, r2]
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            return [accuracy, average_precision, f1, roc_auc]

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return [accuracy, macro_f1, micro_f1]

    @property
    def stats(self) -> Dict[stats.StatType, Union[list, tuple]]:
        if self._stats is None:
            db = self.dataset.get_db(upto_test_timestamp=False)
            table = self._get_full_table(db)
            target = table.df[self.target_col]
            self._set_stats(target)

        return self._stats

    def __init__(self, dataset: str, cache_dir: Optional[str] = None):
        ALLOWED_TASK_TYPES = [
            TaskType.REGRESSION,
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]
        if self.task_type not in ALLOWED_TASK_TYPES:
            raise ValueError(
                f"Task type {self.task_type.name} not allowed by CTUEntityTask."
            )

        super().__init__(dataset=dataset, cache_dir=cache_dir)

    def make_split(self, split: str, table: Table) -> Table:
        r"""Make a table using the task definition.

        Args:
            split: The split to be made.
            table: The table to be split.

        Returns:
            Table: The split table.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def get_sanitized_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""Get the database object for the task without task target data.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.
        Returns:
            Database: The database object.
        """

        _db = self.dataset.get_db(upto_test_timestamp=upto_test_timestamp)

        db = deepcopy(_db)

        db.table_dict[self.entity_table].df.drop(columns=[self.target_col], inplace=True)

        return db

    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split."""

        db = self.dataset.get_db(upto_test_timestamp=False)

        full_table = self._get_full_table(db)

        if self.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]:
            full_table.df[self.target_col], unique = full_table.df[
                self.target_col
            ].factorize(sort=True)

        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
            full_table.df[self.target_col] = full_table.df[self.target_col].astype(float)
        else:
            full_table.df[self.target_col] = full_table.df[self.target_col].astype(int)

        split_table = self.make_split(split, full_table)

        return split_table

    def _get_full_table(self, db: Database) -> Table:
        time_col = db.table_dict[self.entity_table].time_col

        table = Table(
            df=db.table_dict[self.entity_table].df.copy(),
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=time_col,
        )

        table.df = table.df[table.df[self.target_col].notna()]

        return table

    def _set_stats(self, target: pd.Series):
        t = {
            TaskType.REGRESSION: stype.numerical,
            TaskType.BINARY_CLASSIFICATION: stype.categorical,
            TaskType.MULTICLASS_CLASSIFICATION: stype.categorical,
            TaskType.MULTILABEL_CLASSIFICATION: stype.multicategorical,
        }[self.task_type]
        self._stats = {}
        for s in stats.StatType.stats_for_stype(t):
            self._stats[s] = s.compute(target)
