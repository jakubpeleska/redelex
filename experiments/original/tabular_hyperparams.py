from typing import Any, Dict, Optional

import math
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

from tqdm import tqdm

import torch

from torch_frame.data import StatType

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


from relbench.base import BaseTask, EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input
from relbench.tasks import get_task

sys.path.append(".")

from redelex.tasks import CTUBaseEntityTask, CTUEntityTaskTemporal
from redelex.utils import standardize_table_dt
from redelex.nn.models.tabular import TabularModel

from experiments.utils import (
    get_cache_path,
    get_data,
    get_loss,
    get_metrics,
    get_tune_metric,
)


def run_experiment(
    config: tune.TuneConfig,
    data: HeteroData,
    task: BaseTask,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
):
    context = ray_train.get_context()
    # experiment_dir = context.get_trial_dir()

    dataset_name: int = config["dataset_name"]
    task_name: int = config["task_name"]
    random_seed: int = config["seed"]
    lr: float = config["lr"]
    min_epochs: int = config["min_epochs"]
    batch_size: int = config["batch_size"]
    channels: int = config["channels"]
    tabular_model: str = config["tabular_model"]
    max_steps_per_epoch: int = config["max_steps_per_epoch"]
    min_total_steps: int = config["min_total_steps"]
    mlp_norm: str = config["mlp_norm"]
    num_workers: int = 0

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

    loss_fn, out_channels = get_loss(dataset_name, task_name)
    tune_metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    metrics = get_metrics(dataset_name, task_name)

    is_temporal = (
        isinstance(task, CTUEntityTaskTemporal) or isinstance(task, EntityTask)
    ) and hasattr(data[task.entity_table], "time")

    loader_dict: Dict[str, NeighborLoader] = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        standardize_table_dt(table)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[],
            time_attr="time" if is_temporal else None,
            input_nodes=table_input.nodes,
            input_time=table_input.time if is_temporal else None,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    model = TabularModel(
        tf=data[task.entity_table].tf,
        col_stats=col_stats_dict[task.entity_table],
        tabular_model=tabular_model,
        channels=channels,
        out_channels=out_channels,
        norm=mlp_norm,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    def train(split: str = "train") -> float:
        model.train()

        loader = loader_dict[split]

        loss_accum = count_accum = 0
        steps = 0
        total_steps = min(len(loader), max_steps_per_epoch)
        for batch in tqdm(loader, total=total_steps):
            batch = batch.to(device)[task.entity_table]

            optimizer.zero_grad()
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                target = batch.y.long()
            else:
                target = batch.y.float()

            loss = loss_fn(pred.float(), target)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

            steps += 1
            if steps > max_steps_per_epoch:
                break

        return loss_accum / count_accum

    @torch.no_grad()
    def test(split: str) -> np.ndarray:
        loader = loader_dict[split]

        model.eval()

        pred_list = []
        for batch in tqdm(loader):
            batch = batch.to(device)[task.entity_table]
            pred = model(batch)

            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)

            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                pred = torch.softmax(pred, dim=1)

            pred = pred.view(-1) if pred.size(1) == 1 else pred

            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()

    training_time = 0
    best_val_metric = -math.inf if higher_is_better else math.inf
    # report initial values
    ray_train.report(
        {f"val_{tune_metric}": best_val_metric, f"test_{tune_metric}": best_val_metric}
    )

    epoch_steps = min(len(loader_dict["train"]), max_steps_per_epoch)
    n_epochs = max(math.ceil(min_total_steps / epoch_steps), min_epochs)

    val_table = task.get_table("val")

    # model_checkpoint = os.path.join(experiment_dir, "best_model.pth")

    for epoch in range(1, n_epochs + 1):
        start = timer()
        train_loss = train()
        end = timer()

        training_time += end - start

        val_pred = test("val")
        val_metrics = task.evaluate(val_pred, val_table, metrics=metrics)

        metrics_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_time": training_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] <= best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            metrics_dict.update({f"best_val_{k}": v for k, v in val_metrics.items()})
            # torch.save(model.state_dict(), model_checkpoint)

            test_pred = test("test")
            test_metrics = task.evaluate(test_pred, metrics=metrics)
            metrics_dict.update({f"test_{k}": v for k, v in test_metrics.items()})

        ray_train.report(metrics_dict)


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    tabular_model: str,
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
    aggregate_neighbors: bool = False,
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
        # object_store_memory=80e9,
        num_cpus=num_cpus if ray_address == "local" else None,
        num_gpus=num_gpus if ray_address == "local" else None,
        # _temp_dir=os.path.join(os.path.abspath("."), ".tmp"),
    )

    config = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "seed": tune.randint(0, 1000),
        # training config
        "min_epochs": 10,
        "max_steps_per_epoch": 2000,
        "min_total_steps": 1000,
        "lr": 0.001,  # tune.choice([0.001, 0.005]),
        # sampling config
        "batch_size": 512,  # tune.choice([128, 256, 512]),
        # model config
        "tabular_model": tabular_model,  # tune.grid_search(["resnet", "linear"]),
        "channels": 64,  # tune.grid_search([16, 32, 64]),
        "mlp_norm": "batch_norm",  # tune.grid_search(["batch_norm", "layer_norm"]),
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
            tune.with_parameters(
                run_experiment, data=data, task=task, col_stats_dict=col_stats_dict
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
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], default=None)
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
            tabular_model=args.tabular_model,
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
