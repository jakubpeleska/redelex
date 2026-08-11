from typing import Dict, Optional, Any, Union


import json
from pathlib import Path

from mlflow.tracking import MlflowClient
from mlflow.entities import Run

import numpy as np

import pandas as pd

import torch

from torch_frame import stype


from relbench.base import Database, EntityTask, TaskType, Table
from relbench.modeling.graph import NodeTrainTableInput
from relbench.datasets import get_dataset

from redelex.data import (
    guess_schema,
    make_pkey_fkey_graph,
    TextEmbedder,
    GloveTextEmbedder,
    PotionTextEmbedder,
)
from redelex.db import DBSchema
from redelex.tasks import mixins, is_temporal_task
from redelex.transforms import AttachTargetTransform

from redelex.utils.datetime import to_unix_time


def get_potato_client():
    return MlflowClient(tracking_uri="http://potato.felk.cvut.cz:2222")


def get_experiment_runs(client: MlflowClient, experiment_name: str, filter_string: str = "status != 'FAILED'") -> list[Run]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    next_token = -1
    all_runs = []
    while next_token is not None:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=1000,
            page_token=next_token if next_token != -1 else None,
        )
        next_token = runs.token
        all_runs.extend(runs.to_list())
    return all_runs


def get_experiment_runs_df(client: MlflowClient, experiment_name: str, filter_string: Optional[str] = None) -> pd.DataFrame:
    all_runs = get_experiment_runs(client, experiment_name, filter_string=filter_string)
    runs_dict = []
    for r in all_runs:
        r_info = r.info.__dict__
        r_data = {k: v for d in r.data.to_dictionary().values() for k, v in d.items()}
        runs_dict.append({**r_info, **r_data})

    df = pd.DataFrame(runs_dict)
    return df


def get_run_metrics(client: MlflowClient, run_id: str, metrics: list[str]) -> pd.DataFrame:
    metrics_dict = {}
    for metric in metrics:
        metric_history = client.get_metric_history(run_id, metric)
        for m in metric_history:
            if m.step not in metrics_dict:
                metrics_dict[m.step] = {}
            metrics_dict[m.step][m.key] = m.value

    df: pd.DataFrame = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "step"}, inplace=True)
    df["run_id"] = run_id
    df = df[["run_id", "step"] + metrics]
    return df


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
        elif isinstance(value, pd.Timestamp):
            hyperparams_logging[key] = value.strftime("%Y-%m-%d %H:%M:%S")
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
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )
