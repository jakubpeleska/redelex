from typing import Dict, Optional, Any, Union

import json
from pathlib import Path

import numpy as np

import torch

from torch_frame import stype

from relbench.base import Database, EntityTask, TaskType, Table
from relbench.modeling.graph import NodeTrainTableInput

from redelex.data import guess_schema, TextEmbedder, GloveTextEmbedder, PotionTextEmbedder
from redelex.db import DBSchema
from redelex.tasks import mixins
from redelex.transforms import AttachTargetTransform
from redelex.utils import to_unix_time


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
