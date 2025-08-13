from typing import Callable, Dict, List, Optional
from numpy.typing import NDArray

import pandas as pd

from relbench.base import Table

from .ctu_entity_task_base import CTUBaseEntityTask

__all__ = ["CTUEntityTaskTemporal"]


class CTUEntityTaskTemporal(CTUBaseEntityTask):
    # To be set by subclass.
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp

    def make_split(self, split: str, table: Table) -> Table:
        if table.time_col is None:
            raise ValueError("The table must have a time column.")

        table.df = table.df[table.df[table.time_col].notna()]

        if split == "train":
            table.df = table.df[table.df[table.time_col] < self.val_timestamp]
        elif split == "val":
            table.df = table.df[
                (table.df[table.time_col] >= self.val_timestamp)
                & (table.df[table.time_col] < self.test_timestamp)
            ]
        elif split == "test":
            table.df = table.df[table.df[table.time_col] >= self.test_timestamp]
        else:
            raise ValueError(f"Invalid split: {split}.")

        table.df = table.df[[self.entity_col, table.time_col, self.target_col]]

        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        return {fn.__name__: fn(target, pred) for fn in metrics}
