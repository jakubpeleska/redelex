from typing import Callable, Dict, Optional

from numpy.typing import NDArray
from relbench.base import Table, TaskType

from .base import BaseTask


class EntityTaskMixin(BaseTask):
    r"""A node prediction task on a dataset.

    Attributes:
        entity_col: The entity column.
        entity_table: The entity table.
        target_col: The target column.

    Other attributes are inherited from BaseTask.
    """

    entity_col: str
    entity_table: str
    target_col: str
    task_type: TaskType

    # TODO: add proper metrics
    @property
    def metrics(self) -> list[Callable[[NDArray, NDArray], float]]:
        if (
            self.task_type == TaskType.REGRESSION
            or self.task_type == TaskType.BINARY_CLASSIFICATION
            or self.task_type == TaskType.MULTICLASS_CLASSIFICATION
            or self.task_type == TaskType.MULTILABEL_CLASSIFICATION
        ):
            return []

    def filter_dangling_entities(self, table: Table) -> Table:
        db = self.dataset.get_db()
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[list[Callable[[NDArray, NDArray], float]]] = None,
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


__all__ = ["EntityTaskMixin"]
