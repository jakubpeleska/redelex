from typing import NamedTuple, Optional, Tuple

import numpy as np

import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeType

from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from relbench.base import TaskType, EntityTask, Table

from redelex.transforms import AttachTargetTransform
from redelex.utils import to_unix_time


class NodeTrainTableInput(NamedTuple):
    r"""Training table input for node prediction.

    - nodes is a Tensor of node indices.
    - time is a Tensor of node timestamps.
    - target is a Tensor of node labels.
    - transform attaches the target to the batch.
    """

    nodes: Tuple[NodeType, torch.Tensor]
    time: Optional[torch.Tensor]
    target: Optional[torch.Tensor]
    transform: Optional[BaseTransform]


def get_node_train_table_input(
    table: Table,
    task: EntityTask,
) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[torch.Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[torch.Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(table.df[task.target_col].values.astype(target_type))
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


def get_metrics(
    task_type: TaskType, **metrics_kwargs
) -> tuple[dict[str, Metric], str, bool]:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return (
            {
                "accuracy": BinaryAccuracy(**metrics_kwargs),
                "precision": BinaryPrecision(**metrics_kwargs),
                "f1": BinaryF1Score(**metrics_kwargs),
                "roc_auc": BinaryAUROC(**metrics_kwargs),
            },
            "roc_auc",
            True,
        )

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return (
            {
                "macro_accuracy": MulticlassAccuracy(average="macro", **metrics_kwargs),
                "micro_accuracy": MulticlassAccuracy(average="micro", **metrics_kwargs),
                "macro_f1": MulticlassF1Score(average="macro", **metrics_kwargs),
                "micro_f1": MulticlassF1Score(average="micro", **metrics_kwargs),
                "macro_roc_auc": MulticlassAUROC(average="macro", **metrics_kwargs),
                "micro_roc_auc": MulticlassAUROC(average="micro", **metrics_kwargs),
            },
            "macro_roc_auc",
            True,
        )

    elif task_type == TaskType.REGRESSION:
        return (
            {
                "mae": MeanAbsoluteError(**metrics_kwargs),
                "mse": MeanSquaredError(**metrics_kwargs),
                "r2": R2Score(**metrics_kwargs),
            },
            "mae",
            False,
        )
    else:
        raise ValueError(f"Task type {task_type} is unsupported")


def get_loss(task_type: TaskType, **loss_kwargs) -> torch.nn.Module:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return torch.nn.BCEWithLogitsLoss(**loss_kwargs)

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.nn.CrossEntropyLoss(**loss_kwargs)

    elif task_type == TaskType.REGRESSION:
        return torch.nn.MSELoss(**loss_kwargs)

    else:
        raise ValueError(f"Task type {task_type} is unsupported")
