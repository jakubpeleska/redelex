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

from torch_geometric.data import HeteroData

from relbench.base import TaskType


def get_num_neighbors_dict(
    data: HeteroData,
    input_type: str,
    base_neighbors: int,
    num_layers: int,
    depth_factor: int,
) -> dict[str, list[int]]:
    num_neighbors_dict: dict[tuple[str, str, str], list[int]] = {
        e: [] for e in data.edge_types
    }
    layer_types = [input_type]
    edge_types = [e for e in data.edge_types if e[2] in layer_types]
    used_edge_types = []
    for n in range(num_layers):
        for e in data.edge_types:
            num_neighbors_dict[e].append(0)
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            num_edges = len([e for e in edge_types if e[2] == dst_type])

            num_neighbors = max(
                1,
                base_neighbors // (depth_factor**n) // num_edges,
            )
            num_neighbors_dict[edge_type][-1] += num_neighbors

        used_edge_types.extend(edge_types)
        layer_types = list(set([src_type for src_type, _, dst_type in edge_types]))
        edge_types = [
            e
            for e in data.edge_types
            if e[2] in layer_types
            and all(
                [
                    not (e[0] == ue[0] and e[1] == ue[1] and e[2] == ue[2])
                    for ue in used_edge_types
                ]
            )
        ]
    return num_neighbors_dict


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
