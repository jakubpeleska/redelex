from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from relbench.base import Dataset, Table, TaskType

from .base import BaseTask


class RecommendationTaskMixin(BaseTask):
    r"""A link prediction task on a dataset.

    Attributes:
        src_entity_col: The source entity column.
        src_entity_table: The source entity table.
        dst_entity_col: The destination entity column.
        dst_entity_table: The destination entity table.
        eval_k: k for eval@k metrics.

    Other attributes are inherited from BaseTask.
    """

    src_entity_col: str
    src_entity_table: str
    dst_entity_col: str
    dst_entity_table: str
    eval_k: int
    task_type: TaskType

    # TODO: add proper metrics
    @property
    def metrics(self) -> list[Callable[[NDArray, NDArray], float]]:
        return []

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset, cache_dir)

    def filter_dangling_entities(self, table: Table) -> Table:
        # filter dangling destination entities from a list
        table.df[self.dst_entity_col] = table.df[self.dst_entity_col].apply(
            lambda x: [i for i in x if i < self.num_dst_nodes]
        )

        # filter dangling source entities and empty list (after above filtering)
        filter_mask = (table.df[self.src_entity_col] >= self.num_src_nodes) | (
            ~table.df[self.dst_entity_col].map(bool)
        )

        if filter_mask.any():
            table.df = table.df[~filter_mask]
            table.df = table.df.reset_index(drop=True)

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

        expected_pred_shape = (len(target_table), self.eval_k)
        if pred.shape != expected_pred_shape:
            raise ValueError(
                f"The shape of pred must be {expected_pred_shape}, but {pred.shape} given."
            )

        pred_isin_list = []
        dst_count_list = []
        for true_dst_nodes, pred_dst_nodes in zip(
            target_table.df[self.dst_entity_col],
            pred,
        ):
            pred_isin_list.append(
                np.isin(np.array(pred_dst_nodes), np.array(true_dst_nodes))
            )
            dst_count_list.append(len(true_dst_nodes))
        pred_isin = np.stack(pred_isin_list)
        dst_count = np.array(dst_count_list)

        return {fn.__name__: fn(pred_isin, dst_count) for fn in metrics}

    @property
    def num_src_nodes(self) -> int:
        return len(self.dataset.get_db().table_dict[self.src_entity_table])

    @property
    def num_dst_nodes(self) -> int:
        return len(self.dataset.get_db().table_dict[self.dst_entity_table])


__all__ = ["RecommendationTaskMixin"]
