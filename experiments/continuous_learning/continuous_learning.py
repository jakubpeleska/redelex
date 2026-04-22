from typing import Any, Literal, Optional

import copy
from pathlib import Path

import traceback
import os
import random
from datetime import datetime, timedelta
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import numpy as np

import torch
from torchmetrics.aggregation import MeanMetric

import ray
from ray import tune, train as ray_train

import lightning as L
from lightning.pytorch import loggers, callbacks
from lightning.pytorch.utilities.model_summary import ModelSummary

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

import redelex.tasks.mixins as task_mixin
from redelex.data import make_pkey_fkey_graph
from redelex.loaders import ComposedLoader
from redelex.nn.train.lightning import LightningEntityTaskWrapper, SaveModelCallback

from experiments.continuous_learning.continuous_task import ContinuousWrapper

from experiments.continuous_learning.models import HeterogeneousSAGE

from experiments.continuous_learning.utils import (
    get_attribute_schema,
    get_hyperparams_logging,
    get_text_embedder,
    get_table_input,
)


def run_continuous_learning_experiment(
    config: dict[str, Any],
    with_ray: bool = True,
    with_mlflow: bool = True,
):
    random_seed: int = config["seed"]
    lr: float = config["lr"]

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")

    if with_ray:
        context = ray_train.get_context()
        trial_name = context.get_trial_name()
        resources = context.get_trial_resources().required_resources
        print(f"Resources: {resources}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.set_num_threads(1)
    else:
        allow_gpu = config.get("allow_gpu", False)
        if allow_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.set_num_threads(1)
        trial_name = f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("Device:", device)

    dataset_name: str = config["dataset_name"]
    task_name: str = config["task_name"]
    cache_path: str = config["cache_path"]

    model_save_dir = Path(config["model_save_dir"])
    model_save_dir = f"{model_save_dir}/{trial_name}"
    config["model_save_dir"] = model_save_dir

    gnn_channels = config["gnn_channels"]
    gnn_layers = config["gnn_layers"]
    gnn_aggr = config["gnn_aggr"]
    num_neighbors = config["num_neighbors"]

    batch_size = config["batch_size"]
    lr = config["lr"]
    max_training_steps: int = config["max_training_steps"]

    learning_mode: str = config["learning_mode"]
    weights_path: Optional[str] = config.get("weights_path", None)
    train_timestamp = config["train_timestamp"]
    val_timestamp = config["val_timestamp"]
    prev_train_timestamp: Optional[int] = config.get("prev_train_timestamp", None)

    assert learning_mode in ["from_scratch", "ft_full", "ft_upsample", "ft_newonly"]

    if learning_mode != "from_scratch":
        assert (
            weights_path is not None
        ), "weights_path must be provided for fine-tuning modes"

    if learning_mode in ["ft_upsample", "ft_newonly"]:
        assert (
            prev_train_timestamp is not None
        ), "prev_train_timestamp must be provided for ft_upsample and ft_newonly modes"

    dataset = get_dataset(dataset_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)

    text_embedder = get_text_embedder(
        config["text_embedder_name"], device=torch.device("cpu")
    )
    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
    )

    # create model
    model = HeterogeneousSAGE(
        data=data,
        col_stats_dict=col_stats_dict,
        gnn_channels=gnn_channels,
        gnn_layers=gnn_layers,
        gnn_aggr=gnn_aggr,
    )

    # optionally load weights from previous split
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    if learning_mode in ["from_scratch", "ft_full"]:
        train_start = db.min_timestamp
    elif learning_mode in ["ft_upsample", "ft_newonly"]:
        train_start = prev_train_timestamp

    # create train dataloader for current split
    train_table = wrapped_task.get_table(start=train_start, end=train_timestamp)
    train_input = get_table_input(train_table, task)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
        time_attr="time",
        input_nodes=train_input.nodes,
        input_time=train_input.time,
        transform=train_input.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=True,
    )
    if learning_mode == "ft_upsample":
        # for upsampling we create two loaders for new increment and
        # old data and sample from them with 0.5 probability each
        old_train_table = wrapped_task.get_table(
            start=db.min_timestamp, end=prev_train_timestamp
        )
        old_train_input = get_table_input(old_train_table, task)
        old_train_loader = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
            time_attr="time",
            input_nodes=old_train_input.nodes,
            input_time=old_train_input.time,
            transform=old_train_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=True,
        )
        train_loader = ComposedLoader([train_loader, old_train_loader], mode="rnd_uni")

    # create val dataloader for current split
    val_table = wrapped_task.get_table(start=train_timestamp, end=val_timestamp)
    val_input = get_table_input(val_table, task)
    val_loader = NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
        time_attr="time",
        input_nodes=val_input.nodes,
        input_time=val_input.time,
        transform=val_input.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=3
    )

    lightning_model = LightningEntityTaskWrapper(
        model=model,
        task=task,
        optimizer=optimizer,
        lr_scheduler_config={
            "scheduler": scheduler,
            "monitor": "val_loss_epoch",
            "mode": "min",
            "interval": "epoch",
            "frequency": 1,
        },
        metrics=({"loss": MeanMetric()}, "loss", False),
    )

    model_summary = ModelSummary(lightning_model, max_depth=2)

    config["model_parameters"] = model_summary.total_parameters
    config["model_size_MB"] = model_summary.model_size

    hyperparams_logging = get_hyperparams_logging(config)

    if with_mlflow:
        mlflow_experiment: str = config["mlflow_experiment"]
        mlflow_uri: str = config["mlflow_uri"]
        logger = loggers.MLFlowLogger(
            experiment_name=mlflow_experiment,
            run_name=trial_name,
            tracking_uri=mlflow_uri,
        )
        logger.log_hyperparams(hyperparams_logging)
    else:
        experiment_dir = config["experiment_dir"]
        logger = loggers.CSVLogger(save_dir=experiment_dir, name=trial_name)
        logger.log_hyperparams(hyperparams_logging)

    save_model_callback = SaveModelCallback(
        save_dir=model_save_dir,
        monitor="val_loss_epoch",
        mode="min",
        save_every_epoch=True,
    )
    trainer = L.Trainer(
        max_steps=max_training_steps,
        max_epochs=config.get("max_epochs", None),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        accelerator=device.type,
        devices=1,
        logger=logger,
        callbacks=[save_model_callback],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        max_time=timedelta(hours=2),
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
    )
    try:
        trainer.fit(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=None,
        )

        if with_ray:
            best_val_loss = trainer.callback_metrics.get("val_loss_epoch")
            if best_val_loss is not None:
                best_val_loss = best_val_loss.item()
            else:
                best_val_loss = float("inf")
            ray_train.report({"val_loss_epoch": best_val_loss, "model_save_dir": str(model_save_dir)})


    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        stack_trace = traceback.format_exc()
        logger.log_hyperparams({"stack_trace": stack_trace})
        print(stack_trace)
        logger.finalize("failed")


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    learning_mode: str,
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "pelesjak_test_experiment",
    num_samples: Optional[int] = 1,
    num_gpus: int = 0,
    num_cpus: int = 1,
    random_seed: int = 42,
    cache_dir: str = ".cache",
    model_save_dir: str = "./models",
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

    resources = ray.available_resources()
    print(f"Ray resources: {resources}")

    gpus_used = 0
    cpus_used = 2
    if "GPU" in resources:
        gpus_used = 1

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)

    splits = copy.deepcopy(wrapped_task.get_splits())
    best_weights_path = None

    for i in range(1, len(splits) - 1):
        train_timestamp = splits[i]
        val_timestamp = splits[i+1]
        prev_train_timestamp = splits[i-1] if i > 1 else None
        
        current_learning_mode = "from_scratch" if i == 1 else learning_mode

        tuner = tune.Tuner(
            tune.with_resources(
                run_continuous_learning_experiment,
                resources={"CPU": cpus_used, "GPU": gpus_used},
            ),
            run_config=ray_train.RunConfig(
                name=ray_experiment_name,
                storage_path=ray_storage_path,
                stop={"time_total_s": 3600 * 4},
                log_to_file=True,
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                trial_name_creator=lambda trial: (
                    f"{dataset_name}_{task_name}_{trial.trial_id}"
                ),
                trial_dirname_creator=lambda trial: trial.trial_id,
                max_concurrent_trials=num_cpus,
            ),
            param_space={
                "dataset_name": dataset_name,
                "task_name": task_name,
                "learning_mode": current_learning_mode,
                "seed": tune.randint(0, 1000),
                "text_embedder_name": "glove",
                "mlflow_experiment": mlflow_experiment,
                "mlflow_uri": mlflow_uri,
                "max_training_steps": 2000,
                "limit_train_batches": 100,
                "train_timestamp": train_timestamp,
                "val_timestamp": val_timestamp,
                "prev_train_timestamp": prev_train_timestamp,
                "weights_path": best_weights_path,
                "lr": 0.001,
                "batch_size": 128,
                "num_neighbors": 32,
                "gnn_channels": 128,
                "gnn_layers": 2,
                "gnn_aggr": "sum",
                "head_norm": "batch_norm",
                "cache_path": Path(cache_dir).absolute(),
                "model_save_dir": (Path(model_save_dir).absolute())
            },
        )
        results = tuner.fit()
        
        if results.errors:
            print(f"Errors encountered in split {i}. Stopping continuous learning.")
            break

        best_result = results.get_best_result(metric="val_loss_epoch", mode="min")
        if best_result is None or "model_save_dir" not in best_result.metrics:
            print(f"Failed to find the best result in split {i}. Stopping.")
            break
            
        best_weights_path = f"{best_result.metrics['model_save_dir']}/best_model.pt"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--learning_mode", type=str, choices=["from_scratch", "ft_full", "ft_upsample", "ft_newonly"], default="from_scratch")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    task_name = args.task

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
        random_seed=args.seed,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
        learning_mode=args.learning_mode,
    )
