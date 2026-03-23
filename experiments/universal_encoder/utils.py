from typing import Dict, Optional, Any, Union

import json
from pathlib import Path

import numpy as np

import torch

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

from torch_frame import stype
from torch_frame.data import StatType


from relbench.base import Database, EntityTask, TaskType, Table
from relbench.modeling.graph import NodeTrainTableInput
from relbench.datasets import get_dataset

from redelex.data import (
    guess_schema,
    make_pkey_fkey_graph,
    make_tensor_stats_dict,
    TextEmbedder,
    GloveTextEmbedder,
    PotionTextEmbedder,
)
from redelex.db import DBSchema
from redelex.tasks import mixins, is_temporal_task
from redelex.transforms import (
    AttachTargetTransform,
    AttachDictTransform,
    AttachValuesTransform,
)
from redelex.utils import to_unix_time


def get_dataset_data(
    dataset_name: str,
    cache_path: str,
    text_embedder,
    device: torch.device,
    target: Optional[tuple[str, str]] = None,
):
    dataset = get_dataset(dataset_name)
    db = dataset.get_db.__wrapped__(dataset, False)

    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
        target=target,
    )

    tensor_stats_dict = {}
    name_embeddings_dict = {}
    for tname, col_stats in col_stats_dict.items():
        tensor_stats_dict[tname] = make_tensor_stats_dict(
            col_stats_dict=col_stats,
            col_names_dict=data[tname].tf.col_names_dict,
            text_embedder=text_embedder,
            device=device,
        )
        name_embeddings_dict[tname] = {
            s: text_embedder(s).to(device) for s in [tname, *col_stats_dict[tname].keys()]
        }

    return data, col_stats_dict, tensor_stats_dict, name_embeddings_dict


def get_task_data(
    task: EntityTask,
    task_name: str,
    data: HeteroData,
    name_embeddings_dict: dict[str, torch.Tensor],
    tensor_stats_dict: dict[str, torch.Tensor],
    col_stats_dict: dict[str, dict[str, Any]],
    config: dict[str, Any],
    splits: list[str] = ["train", "val"],
    normalize_target: bool = True,
) -> dict[str, Any]:
    is_temporal = is_temporal_task(task)
    print(f"Task {task_name} is temporal: {is_temporal}")

    dict_transform = AttachDictTransform(
        [("name_embeddings", name_embeddings_dict), ("tensor_stats", tensor_stats_dict)]
    )

    values_transform = AttachValuesTransform(
        [
            ("task_name", task_name),
            ("task_type", task.task_type),
            ("entity_table", task.entity_table),
        ]
    )

    loader_dict: dict[str, NeighborLoader] = {}
    for split in splits:
        table = task.get_table.__wrapped__(task, split, mask_input_cols=False)
        if normalize_target and task.task_type == TaskType.REGRESSION:
            # normalize target for regression
            if isinstance(task, mixins.ImputeEntityTaskMixin):
                minimum = col_stats_dict[task.entity_table][task.target_col][
                    StatType.QUANTILES
                ][0]
                maximum = col_stats_dict[task.entity_table][task.target_col][
                    StatType.QUANTILES
                ][4]
            elif isinstance(task, EntityTask):
                minimum = task.stats()["total"]["min_target"]
                maximum = task.stats()["total"]["max_target"]
            table.df[task.target_col] = (table.df[task.target_col] - minimum) / (
                maximum - minimum
            )
        table_input = get_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(config["num_neighbors"] / 2**i) for i in range(config["gnn_layers"])
            ],
            time_attr="time" if is_temporal else None,
            input_nodes=table_input.nodes,
            input_time=table_input.time if is_temporal else None,
            transform=T.Compose([values_transform, table_input.transform, dict_transform]),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
    return task, loader_dict


def get_text_embedder(
    embedder_name: str, device: Optional[torch.device] = None
) -> TextEmbedder:
    if embedder_name == "glove":
        return GloveTextEmbedder(device=device)
    elif embedder_name == "potion":
        return PotionTextEmbedder(device=device)
    else:
        raise ValueError(f"Text embedder {embedder_name} is not supported")


def get_hyperparams_logging(
    config: dict[str, Any],
) -> dict[str, Union[str, int, float, bool]]:
    hyperparams_logging = {}
    for key, value in config.items():
        if type(value) in [str, int, float, bool]:
            hyperparams_logging[key] = value
    return hyperparams_logging


def get_attribute_schema(
    schema_cache_path: str,
    db: Database,
    db_schema: Optional[DBSchema] = None,
    task: Optional[mixins.BaseTask] = None,
) -> Dict[str, Dict[str, stype]]:
    try:
        with open(schema_cache_path, "r") as f:
            attribute_schema = json.load(f)
        for tname, table_attribute_schema in attribute_schema.items():
            for col, stype_str in table_attribute_schema.items():
                if isinstance(stype_str, str):
                    table_attribute_schema[col] = stype(stype_str)
    except FileNotFoundError:
        if db_schema is not None:
            attribute_schema = guess_schema(db, db_schema, task=task)
        else:
            attribute_schema = guess_schema(db, task=task)
        Path(schema_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(schema_cache_path, "w") as f:
            json.dump(attribute_schema, f, indent=2, default=str)

    return attribute_schema


def get_table_input(table: Table, task: EntityTask):
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
