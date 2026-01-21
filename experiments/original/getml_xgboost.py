from typing import Dict, Optional, Union

import os
import random
import sys
import json
from pathlib import Path
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

from argparse import ArgumentParser

from datetime import datetime
from timeit import default_timer as timer

import ray
from ray import tune, train as ray_train
from ray.tune.logger.aim import AimLoggerCallback
from ray.tune.logger.mlflow import MLflowLoggerCallback

import numpy as np
import pandas as pd

import torch

from torch_frame import stype
from torch_frame.data import StatType

import getml
from getml.feature_learning import loss_functions

from relbench.base import Database, EntityTask, TaskType, Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task

sys.path.append(".")

from redelex.tasks import CTUBaseEntityTask, CTUEntityTaskTemporal
from redelex.datasets import DBDataset
from redelex.utils import guess_schema, convert_timedelta, to_unix_time


from experiments.utils import (
    get_cache_path,
    get_metrics,
    get_tune_metric,
)


def set_getml_roles(df: getml.data.DataFrame, table: Table, col_to_stype: Dict[str, stype]):
    role = None
    for col in df.columns:
        st = col_to_stype.get(col, None)
        if col == table.pkey_col or col in table.fkey_col_to_pkey_table:
            role = getml.data.roles.join_key
        elif col == table.time_col or st == stype.timestamp:
            role = getml.data.roles.time_stamp
        elif st == stype.categorical:
            role = getml.data.roles.categorical
        elif st == stype.multicategorical:
            role = getml.data.roles.text
        elif st == stype.numerical:
            role = getml.data.roles.numerical
        elif st == stype.text_embedded:
            role = getml.data.roles.text

        if role is not None:
            df.set_role(col, role)

    return df


def build_task_df(
    task: EntityTask, split: str, return_task_table: bool = False
) -> Union[getml.data.DataFrame, tuple[getml.data.DataFrame, Table]]:
    task_table = task.get_table(split, mask_input_cols=False)
    task_df = task_table.df

    target_cols = []

    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        unique_values = task.stats[StatType.COUNT][0]

        for i, label in enumerate(unique_values):
            col = (task_df[task.target_col] == i).astype(int)
            name = task.target_col + "=" + str(label)
            task_df[name] = col
            target_cols.append(name)
    else:
        target_cols.append(task.target_col)

    name = f"__task_df_{split}__"

    task_getml_df = getml.data.DataFrame.from_pandas(task_df, name=name)

    for col in target_cols:
        task_getml_df.set_role(col, getml.data.roles.target)

    if task_table.pkey_col is not None:
        task_getml_df.set_role(task_table.pkey_col, getml.data.roles.join_key)

    for fk in task_table.fkey_col_to_pkey_table:
        task_getml_df.set_role(fk, getml.data.roles.join_key)

    if task_table.time_col is not None:
        task_getml_df.set_role(task_table.time_col, getml.data.roles.time_stamp)

    if return_task_table:
        return task_getml_df, task_table

    return task_getml_df


def build_getml_task_data(
    db: Database,
    task: Union[CTUBaseEntityTask, EntityTask],
    col_to_stype_dict: Dict[str, Dict[str, stype]],
) -> Dict[str, getml.data.DataFrame]:
    df_dict: Dict[str, getml.data.DataFrame] = {}

    for table_name, table in db.table_dict.items():
        df = table.df
        if table_name == task.entity_table:
            if task.target_col in df.columns:
                df = df.drop(columns=[task.target_col])

        df_dict[table_name] = getml.data.DataFrame.from_pandas(df, name=table_name)

    # add proper roles based on schema
    for table_name, table in db.table_dict.items():
        assert table_name in df_dict
        df = df_dict[table_name]
        col_to_stype = col_to_stype_dict[table_name]
        df = set_getml_roles(df, table, col_to_stype)

    return df_dict


def max_multiplicity(df: pd.DataFrame, fk_col: str):
    mltp = df[fk_col].dropna().value_counts().max()
    return mltp if pd.notna(mltp) else 0


def build_getml_datamodel(
    db: Database,
    df_dict: Dict[str, getml.data.DataFrame],
    task: Union[CTUBaseEntityTask, EntityTask],
) -> getml.data.DataModel:
    train_df, train_table = build_task_df(task, "train", return_task_table=True)
    task_table_name = "__task_table__"
    dm = getml.data.DataModel(train_df.to_placeholder(task_table_name))
    for table_name, df in df_dict.items():
        dm.add(df_dict[table_name].to_placeholder(table_name))

    table_dict: Dict[str, Table] = {task_table_name: train_table, **db.table_dict}
    links = defaultdict(list)
    for table_name, table in table_dict.items():
        for fk, ref_table in table.fkey_col_to_pkey_table.items():
            ref_pk = table_dict[ref_table].pkey_col

            mltp = max_multiplicity(table.df, fk)
            if mltp <= 1:
                rel = getml.data.relationship.one_to_one
                rev_rel = getml.data.relationship.one_to_many
            else:
                rel = getml.data.relationship.many_to_one
                rev_rel = getml.data.relationship.propositionalization

            links[table_name].append((table_name, (fk, rel, ref_pk), ref_table))
            links[ref_table].append((ref_table, (ref_pk, rev_rel, fk), table_name))

    is_temporal = isinstance(task, (CTUEntityTaskTemporal, EntityTask))
    open = [task_table_name]
    closed = []
    while len(open) > 0:
        src_table = open.pop()
        closed.append(src_table)
        for link in links[src_table]:
            dst_table = link[2]
            l_col, rel, r_col = link[1]

            if dst_table in closed:
                continue

            if dst_table not in open:
                open.append(dst_table)

            p_left: getml.data.Placeholder = getattr(dm, src_table)
            p_right: getml.data.Placeholder = getattr(dm, dst_table)

            time_cols = None
            if (
                is_temporal
                and table_dict[src_table].time_col is not None
                and table_dict[dst_table].time_col is not None
            ):
                time_cols = (
                    table_dict[src_table].time_col,
                    table_dict[dst_table].time_col,
                )

            p_left.join(p_right, on=(l_col, r_col), time_stamps=time_cols, relationship=rel)

    return dm


def run_experiment(
    config: tune.TuneConfig,
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
):
    context = ray_train.get_context()

    dataset_name: int = config["dataset_name"]
    task_name: int = config["task_name"]
    random_seed: int = config["seed"]

    random.seed(random_seed)
    np.random.seed(random_seed)

    resources = context.get_trial_resources().required_resources
    print(f"Resources: {resources}")

    getml.engine.launch()
    getml.engine.set_project(f"xgboost_{dataset_name}_{task_name}_{random_seed}")

    task = get_task(dataset_name, task_name)

    df_dict = build_getml_task_data(db, task, col_to_stype_dict)
    datamodel = build_getml_datamodel(db, df_dict, task)

    val_df = build_task_df(task, "val")
    test_df = build_task_df(task, "test")
    train_df = build_task_df(task, "train")

    container = getml.data.Container(
        train=train_df, validation=val_df, test=test_df, peripheral=df_dict
    )
    container.freeze()

    if task.task_type in [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    ]:
        predictor = getml.predictors.XGBoostClassifier()
        fast_prop = getml.feature_learning.FastProp(
            loss_function=loss_functions.CrossEntropyLoss,
        )

    elif task.task_type == TaskType.REGRESSION:
        predictor = getml.predictors.XGBoostRegressor()
        fast_prop = getml.feature_learning.FastProp(
            loss_function=loss_functions.SquareLoss,
        )
    else:
        raise ValueError("unsupported task type")

    pipe = getml.pipeline.Pipeline(
        data_model=datamodel,
        # preprocessors=[mapping],
        feature_learners=[fast_prop],
        # feature_selectors=[feature_selector],
        predictors=[predictor],
        share_selected_features=0.5,
        include_categorical=True,
    )

    metrics = get_metrics(dataset_name, task_name)

    start = timer()
    pipe = pipe.fit(container.train)
    end = timer()

    training_time = end - start

    val_pred = pipe.predict(container.validation)
    val_metrics = task.evaluate(val_pred, task.get_table("val"), metrics=metrics)

    test_pred = pipe.predict(container.test)
    test_metrics = task.evaluate(test_pred, metrics=metrics)

    metrics_dict = {
        "epoch": 1,
        "train_time": training_time,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    metrics_dict.update({f"best_val_{k}": v for k, v in val_metrics.items()})
    metrics_dict.update({f"test_{k}": v for k, v in test_metrics.items()})

    ray_train.report(metrics_dict)


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "pelesjak_test_experiment",
    aim_repo: Optional[str] = None,
    num_samples: Optional[int] = 1,
    num_gpus: int = 0,
    num_cpus: int = 1,
    random_seed: int = 42,
    cache_dir: str = ".cache",
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if num_gpus > 0 and ray_address == "local":
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        nvmlInit()
        free_memory = [
            int(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free)
            for i in range(torch.cuda.device_count())
        ]
        device_idx = np.argsort(free_memory)[::-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_idx[:num_gpus].astype(str))
        print("Free memory:", free_memory, os.environ["CUDA_VISIBLE_DEVICES"])

    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        log_to_driver=False,
        include_dashboard=False,
        num_cpus=num_cpus if ray_address == "local" else None,
        num_gpus=num_gpus if ray_address == "local" else None,
    )

    config = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "seed": tune.randint(0, 1000),
    }
    # scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    scheduler = None

    if ray_experiment_name is None:
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ray_experiment_name = f"resnet_sage_{time}_{dataset_name}_{task_name}"

    metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    tune_metric = f"val_{metric}"
    metric_mode = "max" if higher_is_better else "min"

    resources = ray.available_resources()

    gpus_used = 0
    cpus_used = 1
    if "GPU" in resources:
        batch_model_size = 4e9
        gpu_memory = max(
            [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            ]
        )
        gpus_used = batch_model_size / gpu_memory

    ray_callbacks = []
    if mlflow_uri is not None:
        ray_callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=mlflow_uri,
                experiment_name=mlflow_experiment,
            )
        )
    if aim_repo is not None:
        ray_callbacks.append(
            AimLoggerCallback(repo=aim_repo, experiment_name=ray_experiment_name)
        )

    cache_path = get_cache_path(dataset_name, task_name, cache_dir)

    dataset = get_dataset(dataset_name)
    task = get_task(dataset_name, task_name)
    if isinstance(task, CTUBaseEntityTask):
        db = task.get_sanitized_db(upto_test_timestamp=False)
    else:
        db = dataset.get_db(upto_test_timestamp=False)

    convert_timedelta(db)

    stypes_cache_path = Path(f"{cache_path}/attribute-schema.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for tname, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                if isinstance(stype_str, str):
                    col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        if isinstance(dataset, DBDataset):
            col_to_stype_dict = guess_schema(db, dataset.get_schema())
        else:
            col_to_stype_dict = guess_schema(db)
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    for tname, col_to_stype in col_to_stype_dict.items():
        for col, st in col_to_stype.items():
            if st == stype.timestamp:
                db.table_dict[tname].df[col] = to_unix_time(db.table_dict[tname].df[col])

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_experiment, db=db, col_to_stype_dict=col_to_stype_dict
            ),
            resources={"CPU": cpus_used, "GPU": gpus_used},
        ),
        run_config=ray_train.RunConfig(
            callbacks=ray_callbacks,
            name=ray_experiment_name,
            storage_path=ray_storage_path,
            stop={"time_total_s": 3600 * 4},
            log_to_file=True,
        ),
        tune_config=tune.TuneConfig(
            metric=tune_metric,
            mode=metric_mode,
            scheduler=scheduler,
            num_samples=num_samples,
            trial_name_creator=lambda trial: f"{dataset_name}_{task_name}_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
        ),
        param_space=config,
    )
    results = tuner.fit()

    try:
        best_result = results.get_best_result(tune_metric, metric_mode)

        print("Best trial config: {}".format(best_result.config))
        print("Best trial metrics: {}".format(best_result.metrics))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--aim_repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_cpus", type=int, default=1)

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    task_name = args.task

    task: CTUBaseEntityTask = get_task(dataset_name, task_name)
    if task.task_type in [
        TaskType.LINK_PREDICTION,
        TaskType.MULTILABEL_CLASSIFICATION,
    ]:
        print(f"Skipping {dataset_name} - {task_name}...")

    else:
        print(f"Processing {dataset_name} - {task_name}...")

        run_ray_tuner(
            dataset_name,
            task_name,
            ray_address=args.ray_address,
            ray_storage_path=(
                os.path.realpath(args.ray_storage)
                if args.ray_storage is not None
                else os.path.realpath(".results")
            ),
            ray_experiment_name=args.run_name,
            mlflow_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_experiment,
            aim_repo=args.aim_repo,
            random_seed=args.seed,
            num_samples=args.num_samples,
            num_gpus=args.num_gpus,
            num_cpus=args.num_cpus,
        )
