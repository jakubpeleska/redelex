import time
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from numpy.typing import NDArray
from relbench.base import Database, Dataset, Table, TaskType


class BaseTask:
    r"""Base class for a task on a dataset.

    Attributes:
        task_type: The type of the task.
        metrics: The metrics to evaluate this task on.
    """

    # To be set by subclass.
    task_type: TaskType
    metrics: list[Callable[[NDArray, NDArray], float]]

    def __init__(self, dataset: Dataset, cache_dir: Optional[str] = None):
        r"""Create a task object.

        Args:
            dataset: The dataset object on which the task is defined.
            cache_dir: A directory for caching the task table objects. If specified,
                we will either process and cache the file (if not available) or use
                the cached file. If None, we will not use cached file and re-process
                everything from scratch without saving the cache.
        """
        self.dataset = dataset
        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={repr(self.dataset)})"

    def make_table(
        self,
        db: Database,
        split_range: Union[pd.Series, pd.DataFrame],
    ) -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for ground truth.
            split_range: Data used to defining entity range for the table.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def make_split_range(self, db: Database, split: str) -> Union[pd.Series, pd.DataFrame]:
        r"""Make a data range for a split.

        Args:
            split: The split to be made.
        Returns:
            A data defining range of the split. Timestamps, indices, etc.

        Implemented by TemporalTaskMixin and StaticTaskMixin.
        """
        raise NotImplementedError

    def _get_table(self, split: str, db: Optional[Database] = None) -> Table:
        r"""Helper function to get a table for a split."""

        if db is None:
            db = self.dataset.get_db(upto_test_timestamp=False)

        split_range = self.make_split_range(db, split)

        table = self.make_table(db, split_range)
        table = self.filter_dangling_entities(table)

        return table

    def get_table(self, split, mask_input_cols=None, db: Optional[Database] = None):
        r"""Get a table for a split.

        Args:
            split: The split to get the table for.
            mask_input_cols: If True, keep only the input columns in the table. If
                None, mask the input columns only for the test split. This helps
                prevent data leakage.
            db: The database to use. If None, use the full database.

        Returns:
            The task table for the split.

        The table is cached in memory.
        """

        if mask_input_cols is None:
            mask_input_cols = split == "test"

        table_path = f"{self.cache_dir}/{split}.parquet"
        if self.cache_dir and Path(table_path).exists():
            table = Table.load(table_path)
        else:
            print(f"Making task table for {split} split from scratch...")
            print(
                "(You can also use `get_task(..., download=True)` "
                "for tasks prepared by the RelBench team.)"
            )
            tic = time.time()
            table = self._get_table(split, db=db)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                table.save(table_path)

        if mask_input_cols:
            table = self._mask_input_cols(table)

        return table

    def _mask_input_cols(self, table: Table) -> Table:
        input_cols = [
            table.time_col,
            *table.fkey_col_to_pkey_table.keys(),
        ]
        return Table(
            df=table.df[input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table."""
        raise NotImplementedError

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[list[Callable[[NDArray, NDArray], float]]] = None,
    ):
        r"""Evaluate predictions on the task.

        Args:
            pred: Predictions as a numpy array.
            target_table: The target table. If None, use the test table.
            metrics: The metrics to evaluate the prediction table. If None, use
                the default metrics for the task.
        """
        raise NotImplementedError


__all__ = ["BaseTask"]
