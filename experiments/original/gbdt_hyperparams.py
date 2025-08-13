from typing import Dict, Optional

import copy
import os
import random
import sys


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

import sklearn.metrics as skm
import torch

import torch_frame
from torch_frame import Metric, TensorFrame, stype
from torch_frame.data import StatType
from torch_frame.gbdt import LightGBM

from torch_geometric.data import HeteroData

from relbench.base import BaseTask, EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input
from relbench.tasks import get_task
from relbench.metrics import (
    average_precision,
    f1,
    mae,
    mse,
    r2,
    roc_auc,
)


sys.path.append(".")

from redelex.tasks import CTUBaseEntityTask, CTUEntityTaskTemporal
from redelex.utils import standardize_table_dt


from experiments.utils import (
    get_cache_path,
    get_data,
    get_tune_metric,
)


def get_metrics(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.REGRESSION:
        return [mae, mse, r2]

    elif task.task_type == TaskType.BINARY_CLASSIFICATION:

        def accuracy(true, pred) -> float:
            label = pred > 0.5
            return skm.accuracy_score(true, label)

        return [accuracy, average_precision, f1, roc_auc]

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def macro_f1(true, label) -> float:
            return skm.f1_score(true, label, average="macro")

        def micro_f1(true, label) -> float:
            return skm.f1_score(true, label, average="micro")

        def accuracy(true, label) -> float:
            return skm.accuracy_score(true, label)

        return [accuracy, macro_f1, micro_f1]
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def run_experiment(
    config: tune.TuneConfig,
    data: HeteroData,
    task: BaseTask,
):
    context = ray_train.get_context()
    # experiment_dir = context.get_trial_dir()

    dataset_name: int = config["dataset_name"]
    task_name: int = config["task_name"]
    random_seed: int = config["seed"]
    num_trials: int = config["num_trials"]

    # enable_reproducibility(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")

    resources = context.get_trial_resources().required_resources
    print(f"Resources: {resources}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_num_threads(1)
    print("Device:", device)

    num_classes = None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        tune_metric = Metric.ROCAUC
        task_type = torch_frame.TaskType.BINARY_CLASSIFICATION
    elif task.task_type == TaskType.REGRESSION:
        torch
        tune_metric = Metric.MAE
        task_type = torch_frame.TaskType.REGRESSION

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        num_classes = len(task.stats[StatType.COUNT][0])
        tune_metric = Metric.ACCURACY
        task_type = torch_frame.TaskType.MULTICLASS_CLASSIFICATION

    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")

    is_temporal = (
        isinstance(task, CTUEntityTaskTemporal) or isinstance(task, EntityTask)
    ) and hasattr(data[task.entity_table], "time")

    tf_dict: Dict[str, TensorFrame] = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        standardize_table_dt(table)
        table_input = get_node_train_table_input(table=table, task=task)
        tf: TensorFrame = copy.deepcopy(data[task.entity_table].tf[table_input.nodes[1]])
        time_feat = tf.feat_dict.pop(stype.timestamp, None)
        time_feat = (
            time_feat.view((time_feat.shape[0], -1)) if time_feat is not None else None
        )
        in_time_feat = (
            table_input.time[:, None]
            if is_temporal and table_input.time is not None
            else None
        )

        tf.y = table_input.target

        if time_feat is not None:
            if tf.feat_dict.get(stype.numerical) is not None:
                tf.feat_dict[stype.numerical] = torch.concat(
                    [tf.feat_dict[stype.numerical], time_feat], dim=1
                )
            else:
                tf.feat_dict[stype.numerical] = time_feat

        if in_time_feat is not None:
            if tf.feat_dict.get(stype.numerical) is not None:
                tf.feat_dict[stype.numerical] = torch.concat(
                    [tf.feat_dict[stype.numerical], in_time_feat], dim=1
                )
            else:
                tf.feat_dict[stype.numerical] = in_time_feat

        tf_dict[split] = tf

    metrics = get_metrics(dataset_name, task_name)

    model = LightGBM(task_type=task_type, metric=tune_metric, num_classes=num_classes)

    start = timer()
    model.tune(tf_train=tf_dict["train"], tf_val=tf_dict["val"], num_trials=num_trials)
    end = timer()

    training_time = end - start

    val_pred = model.predict(tf_test=tf_dict["val"]).numpy()
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        val_pred = val_pred[:, None]
    val_metrics = task.evaluate(val_pred, task.get_table("val"), metrics=metrics)

    test_pred = model.predict(tf_test=tf_dict["test"]).numpy()
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        test_pred = test_pred[:, None]
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
    aggregate_neighbors: bool = False,
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
        # object_store_memory=80e9,
        num_cpus=num_cpus if ray_address == "local" else None,
        num_gpus=num_gpus if ray_address == "local" else None,
        # _temp_dir=os.path.join(os.path.abspath("."), ".tmp"),
    )

    config = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "seed": tune.randint(0, 1000),
        "num_trials": 5,
    }
    # scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    scheduler = None

    if ray_experiment_name is None:
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ray_experiment_name = f"resnet_sage_{time}_{dataset_name}_{task_name}"

    metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    tune_metric = f"val_{metric}"
    metric_mode = "max" if higher_is_better else "min"

    cache_path = get_cache_path(dataset_name, task_name, cache_dir)

    task, data, col_stats_dict = get_data(
        dataset_name,
        task_name,
        cache_path,
        entity_table_only=True,
        aggregate_neighbors=aggregate_neighbors,
    )

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

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run_experiment, data=data, task=task),
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
    parser.add_argument("--aggregate_neighbors", default=False, action="store_true")

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
            aggregate_neighbors=args.aggregate_neighbors,
        )
