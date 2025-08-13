from typing import Callable, Dict, List, Optional
from numpy.typing import NDArray

import numpy as np

from relbench.base import Table

from .ctu_entity_task_base import CTUBaseEntityTask

__all__ = ["CTUEntityTask"]


class CTUEntityTask(CTUBaseEntityTask):
    def make_split(self, split: str, table: Table) -> Table:
        random_state = np.random.RandomState(seed=42)
        train_df = table.df.sample(frac=0.8, random_state=random_state)
        if split == "train":
            table.df = train_df
        else:
            table.df = table.df.drop(train_df.index)
            val_df = table.df.sample(frac=0.5, random_state=random_state)
            if split == "val":
                table.df = val_df
            else:
                table.df = table.df.drop(val_df.index)

        table.df = table.df[[self.entity_col, self.target_col]]
        table.time_col = None

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
