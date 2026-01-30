import torch
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

from relbench.base import TaskType


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
        return torch.nn.L1Loss(**loss_kwargs)

    else:
        raise ValueError(f"Task type {task_type} is unsupported")
