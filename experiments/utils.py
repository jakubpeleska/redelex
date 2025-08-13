from typing import Dict, List, Optional

import json

from pathlib import Path

import sqlalchemy as sa

import torch
from torch.nn import BCEWithLogitsLoss, L1Loss, CrossEntropyLoss

from sentence_transformers import SentenceTransformer
from torch_frame import stype
from torch_frame.data import StatType
from torch_frame.config.text_embedder import TextEmbedderConfig

from torch_geometric.data import HeteroData

from relbench.base import Database, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    macro_f1,
    mae,
    micro_f1,
    mse,
    r2,
    roc_auc,
)

from redelex.datasets import DBDataset
from redelex.tasks import CTUBaseEntityTask
from redelex.utils import (
    guess_schema,
    convert_timedelta,
    standardize_db_dt,
    merge_tf,
)


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def get_text_embedder():
    return TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
    )


def get_cache_path(dataset_name: str, task_name: str, cache_dir: str):
    task = get_task(dataset_name, task_name)
    if isinstance(task, CTUBaseEntityTask):
        return Path(f"{cache_dir}/{dataset_name}/{task_name}")

    elif isinstance(task, EntityTask):
        return Path(f"{cache_dir}/{dataset_name}")

    else:
        raise ValueError(f"Task type {type(task)} is unsupported")


def get_metrics(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.REGRESSION:
        return [mae, mse, r2]

    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        return [accuracy, average_precision, f1, roc_auc]

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return [accuracy, macro_f1, micro_f1]
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_tune_metric(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc", True

    elif task.task_type == TaskType.REGRESSION:
        return "mae", False

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return "macro_f1", True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_loss(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return BCEWithLogitsLoss(), 1

    elif task.task_type == TaskType.REGRESSION:
        return L1Loss(), 1

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return CrossEntropyLoss(), len(task.stats[StatType.COUNT][0])

    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_attribute_schema(
    schema_cache_path: str,
    db: Database,
    sql_schema: Optional[Dict[str, Dict[str, sa.types.TypeEngine]]] = None,
) -> Dict[str, Dict[str, stype]]:
    try:
        with open(schema_cache_path, "r") as f:
            attribute_schema = json.load(f)
        for tname, table_attribute_schema in attribute_schema.items():
            for col, stype_str in table_attribute_schema.items():
                if isinstance(stype_str, str):
                    table_attribute_schema[col] = stype(stype_str)
    except FileNotFoundError:
        if sql_schema is not None:
            attribute_schema = guess_schema(db, sql_schema)
        else:
            attribute_schema = guess_schema(db)
        Path(schema_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(schema_cache_path, "w") as f:
            json.dump(attribute_schema, f, indent=2, default=str)

    return attribute_schema


def get_data(
    dataset_name: str,
    task_name: str,
    cache_path: str,
    entity_table_only: bool = False,
    aggregate_neighbors: bool = False,
):
    dataset = get_dataset(dataset_name)
    task = get_task(dataset_name, task_name)
    if isinstance(task, CTUBaseEntityTask):
        db = task.get_sanitized_db(upto_test_timestamp=False)
    else:
        db = dataset.get_db(upto_test_timestamp=False)

    convert_timedelta(db)
    attribute_schema = get_attribute_schema(
        f"{cache_path}/attribute_schema.json",
        db,
        sql_schema=dataset.get_schema() if isinstance(dataset, DBDataset) else None,
    )
    standardize_db_dt(db, attribute_schema)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
        ),
        cache_dir=f"{cache_path}/materialized",
    )

    if entity_table_only and aggregate_neighbors:
        edge_dict = data.collect("edge_index")
        node_tf = data[task.entity_table].tf
        for (src, edge_name, dst), edge_index in edge_dict.items():
            if (
                src == task.entity_table
                and edge_index[0].unique(return_counts=True)[1].max() == 1
            ):
                prefix = f"{edge_name}_"
                node_tf = merge_tf(
                    left_tf=node_tf,
                    right_tf=data[dst].tf,
                    left_idx=edge_index[0],
                    right_idx=edge_index[1],
                    right_prefix=prefix,
                )
                col_stats_dict[task.entity_table].update(
                    {f"{prefix}{k}": v for k, v in col_stats_dict[dst].items()}
                )
        data[task.entity_table].tf = node_tf

    if entity_table_only:
        return (
            task,
            HeteroData({task.entity_table: data[task.entity_table]}),
            {task.entity_table: col_stats_dict[task.entity_table]},
        )

    return task, data, col_stats_dict


__all__ = [
    "get_text_embedder",
    "get_cache_path",
    "get_metrics",
    "get_tune_metric",
    "get_loss",
    "get_attribute_schema",
    "get_data",
]
