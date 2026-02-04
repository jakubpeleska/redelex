import traceback
from typing import Optional, Any

import copy
import os
import random
from datetime import datetime, timedelta
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

from argparse import ArgumentParser

import ray
from ray import tune, train as ray_train

import numpy as np

import torch

import lightning as L
from lightning.pytorch import callbacks, loggers
from lightning.pytorch.utilities.model_summary import ModelSummary

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import NodeType

from torch_frame import stype
from torch_frame.data import StatType

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input
from relbench.tasks import get_task

sys.path.append(".")

from redelex.data import (
    TensorStatType,
    make_pkey_fkey_graph,
    make_tensor_stats_dict,
)
from redelex.nn.train import LightningEntityTaskWrapper
from redelex.transforms import AttachDictTransform

from experiments.utils import (
    get_attribute_schema,
    get_hyperparams_logging,
    get_text_embedder,
)

from .models import (
    HeteroSAGEModel,
    UniversalSAGEModel,
    UniversalHomogeneousSAGEModel,
    UniversalHomogeneousGANModel,
)


def get_model(model_name: str, config: dict[str, Any], data: HeteroData) -> torch.nn.Module:
    if model_name == "hetero_sage":
        return HeteroSAGEModel(
            col_channels=config["col_channels"],
            gnn_channels=config["gnn_channels"],
            out_channels=config["out_channels"],
            col_names_dict={nt: tf.col_names_dict for nt, tf in data.tf_dict.items()},
            col_stats_dict=config["col_stats_dict"],
            node_types=data.node_types,
            edge_types=data.edge_types,
            tabular_encoder_model=config["tabular_encoder_model"],
            tabular_encoder_layers=config["tabular_encoder_layers"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
            head_norm=config["head_norm"],
        )
    if model_name == "universal_sage":
        return UniversalSAGEModel(
            gnn_channels=config["gnn_channels"],
            col_channels=config["col_channels"],
            out_channels=config["out_channels"],
            text_embedder=get_text_embedder(
                config["text_embedder_name"], device=torch.device("cpu")
            ),
            node_types=data.node_types,
            edge_types=data.edge_types,
            tabular_encoder_layers=config["tabular_encoder_layers"],
            tabular_encoder_heads=config["tabular_encoder_heads"],
            tabular_encoder_dropout=config["tabular_encoder_dropout"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
            head_norm=config["head_norm"],
        )
    elif model_name == "universal_homogeneous_sage":
        return UniversalHomogeneousSAGEModel(
            gnn_channels=config["gnn_channels"],
            col_channels=config["col_channels"],
            out_channels=config["out_channels"],
            text_embedder=get_text_embedder(
                config["text_embedder_name"], device=torch.device("cpu")
            ),
            tabular_encoder_layers=config["tabular_encoder_layers"],
            tabular_encoder_heads=config["tabular_encoder_heads"],
            tabular_encoder_dropout=config["tabular_encoder_dropout"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
            head_norm=config["head_norm"],
        )

    elif model_name == "universal_homogeneous_gan":
        return UniversalHomogeneousGANModel(
            gnn_channels=config["gnn_channels"],
            col_channels=config["col_channels"],
            out_channels=config["out_channels"],
            text_embedder=get_text_embedder(
                config["text_embedder_name"], device=torch.device("cpu")
            ),
            tabular_encoder_layers=config["tabular_encoder_layers"],
            tabular_encoder_heads=config["tabular_encoder_heads"],
            tabular_encoder_dropout=config["tabular_encoder_dropout"],
            gnn_layers=config["gnn_layers"],
            gnn_heads=config["gnn_heads"],
            gnn_dropout=config["gnn_dropout"],
            head_norm=config["head_norm"],
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_task_experiment(
    config: dict[str, Any],
    data: HeteroData,
    col_stats_dict: dict[NodeType, dict[str, dict[StatType, Any]]],
    tensor_stats_dict: dict[NodeType, dict[stype, dict[TensorStatType, torch.Tensor]]],
    name_embeddings_dict: dict[NodeType, dict[str, torch.Tensor]],
    with_ray: bool = True,
    with_mlflow: bool = True,
):
    dataset_name: str = config["dataset_name"]
    task_name: str = config["task_name"]
    random_seed: int = config["seed"]

    lr: float = config["lr"]
    batch_size: int = config["batch_size"]
    num_neighbors: int = config["num_neighbors"]

    gnn_layers: int = config["gnn_layers"]

    tensor_stats_dict = copy.deepcopy(tensor_stats_dict)
    name_embeddings_dict = copy.deepcopy(name_embeddings_dict)

    config["col_stats_dict"] = col_stats_dict

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
        trial_name = (
            f"{dataset_name}_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    print("Device:", device)

    # Move stats to device
    for tname in tensor_stats_dict:
        for st, stats in tensor_stats_dict[tname].items():
            for stat_name, stat_value in stats.items():
                tensor_stats_dict[tname][st][stat_name] = stat_value.to(device)
        for name, embedding in name_embeddings_dict[tname].items():
            name_embeddings_dict[tname][name] = embedding.to(device)

    task: EntityTask = get_task(dataset_name, task_name)

    if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:
        out_channels = 1
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = len(task.stats()[StatType.COUNT][0])

    config["out_channels"] = out_channels
    model = get_model(config["model_name"], config, data)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    lightning_model = LightningEntityTaskWrapper(
        model=model, optimizer=optimizer, task=task
    )

    model_summary = ModelSummary(lightning_model, max_depth=2)

    config["model_parameters"] = model_summary.total_parameters
    config["model_size_MB"] = model_summary.model_size

    loader_dict: dict[str, NeighborLoader] = {}

    dict_transform = AttachDictTransform(
        [("name_embeddings", name_embeddings_dict), ("tensor_stats", tensor_stats_dict)]
    )

    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
            time_attr="time",
            # temporal_strategy="last",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=T.Compose([table_input.transform, dict_transform]),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
        )

    val_check_interval = min(
        len(loader_dict["val"]) + len(loader_dict["test"]),
        len(loader_dict["train"]),
    )
    if val_check_interval < 100:
        val_check_interval = 100
    if val_check_interval > len(loader_dict["train"]) // 2:
        val_check_interval = len(loader_dict["train"])

    if val_check_interval < len(loader_dict["train"]) // 3:
        val_check_interval = len(loader_dict["train"]) // 3

    config["val_check_interval"] = val_check_interval

    max_training_steps: int = config["max_training_steps"]
    max_training_steps = max(
        max_training_steps, len(loader_dict["train"]), val_check_interval * 30
    )
    config["max_training_steps"] = max_training_steps

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

    trainer = L.Trainer(
        max_steps=max_training_steps,
        accelerator=device.type,
        devices=1,
        logger=logger,
        callbacks=[
            callbacks.EarlyStopping(
                monitor=f"val_{lightning_model.tune_metric}",
                mode="max" if lightning_model.higher_is_better else "min",
                patience=10,
            )
        ],
        num_sanity_val_steps=0,
        val_check_interval=val_check_interval,
        enable_checkpointing=False,
        max_time=timedelta(hours=4),
        use_distributed_sampler=False,
    )
    try:
        trainer.fit(
            lightning_model,
            train_dataloaders=loader_dict["train"],
            val_dataloaders=[loader_dict["val"], loader_dict["test"]],
            ckpt_path=None,
        )
    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        stack_trace = traceback.format_exc()
        logger.log_hyperparams({"stack_trace": stack_trace})
        print(stack_trace)
        logger.finalize("failed")


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    model_name: str,
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
        # _temp_dir=os.path.join(os.getcwd(), ".tmp")
        # if len(os.path.join(os.getcwd(), ".tmp")) < 107
        # else None,
    )

    cache_path = f"{cache_dir}/{dataset_name}"

    dataset = get_dataset(dataset_name)

    db = dataset.get_db(upto_test_timestamp=False)

    schema_cache_path = f"{cache_path}/attribute-schema.json"
    attribute_schema = get_attribute_schema(schema_cache_path, db)

    text_embedder_name = "glove"
    text_embedder = get_text_embedder(text_embedder_name, device=torch.device("cpu"))

    materialized_cache_dir = f"{cache_path}/materialized"
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        attribute_schema,
        text_embedder=text_embedder,
        cache_dir=materialized_cache_dir,
    )
    del db

    tensor_stats_dict = {}
    name_embeddings_dict = {}
    for tname in data.node_types:
        tensor_stats_dict[tname] = make_tensor_stats_dict(
            col_stats_dict=col_stats_dict[tname],
            col_names_dict=data[tname].tf.col_names_dict,
            text_embedder=text_embedder,
            device=torch.device("cpu"),
        )
        name_embeddings_dict[tname] = {
            s: text_embedder(s) for s in [tname, *col_stats_dict[tname].keys()]
        }

    resources = ray.available_resources()
    print(f"Ray resources: {resources}")

    gpus_used = 0
    cpus_used = 2
    if "GPU" in resources:
        gpus_used = 1

    def get_model_specific_params(model_name: str) -> int:
        if model_name == "hetero_sage":
            return {
                "tabular_encoder_model": "resnet",
                "gnn_aggr": "sum",
            }
        elif model_name in ["universal_sage", "universal_homogeneous_sage"]:
            return {
                "tabular_encoder_heads": 8,
                "tabular_encoder_dropout": 0.1,
                "gnn_aggr": "sum",
            }
        elif model_name == "universal_homogeneous_gan":
            return {
                "tabular_encoder_heads": 8,
                "tabular_encoder_dropout": 0.1,
                "gnn_heads": 8,
                "gnn_dropout": 0.1,
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_task_experiment,
                data=data,
                col_stats_dict=col_stats_dict,
                tensor_stats_dict=tensor_stats_dict,
                name_embeddings_dict=name_embeddings_dict,
            ),
            resources={"CPU": cpus_used, "GPU": gpus_used},
        ),
        run_config=ray_train.RunConfig(
            name=ray_experiment_name,
            storage_path=ray_storage_path,
            stop={"time_total_s": 3600 * 24},
            log_to_file=True,
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            trial_name_creator=lambda trial: f"{dataset_name}_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
            max_concurrent_trials=num_cpus,
        ),
        param_space={
            "dataset_name": dataset_name,
            "task_name": task_name,
            "model_name": model_name,
            "seed": tune.randint(0, 1000),
            "text_embedder_name": text_embedder_name,
            # logging config
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            # training config
            "max_training_steps": 30000,
            "lr": 0.0001,
            # sampling config
            "batch_size": 128,
            "num_neighbors": 16,
            # model config
            "col_channels": 512,
            "gnn_channels": 128,
            "tabular_encoder_layers": tune.grid_search([1, 2, 4, 8]),
            "gnn_layers": 2,
            "head_norm": "batch_norm",
            **get_model_specific_params(model_name),
        },
    )
    tuner.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "hetero_sage",
            "universal_sage",
            "universal_homogeneous_sage",
            "universal_homogeneous_gan",
        ],
    )
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_cpus", type=int, default=1)

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    task_name = args.task
    model_name = args.model_name

    run_ray_tuner(
        dataset_name,
        task_name,
        model_name,
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
    )
