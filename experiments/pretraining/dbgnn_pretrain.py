from typing import Any, Dict, Literal, Optional

import os
import random
import sys
from datetime import datetime, timedelta

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

from argparse import ArgumentParser

import ray
from ray import tune, train as ray_train

import numpy as np

import torch

import lightning as L
from lightning.pytorch import callbacks, loggers

from torch_frame.data import StatType

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.nn import MLP

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input

sys.path.append(".")
from redelex.datasets import get_dataset_info
from redelex.nn.models.rdl_model import RDLModel
from redelex.nn.corruptors import DBResampleCorruptor
from redelex.nn.models.pretrain_wrappers import (
    PretrainingModel,
    LightningPretraining,
    LightningEntityTaskModel,
)

from experiments.utils import (
    get_attribute_schema,
    get_text_embedder,
)


def get_backbone(
    data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    channels: int,
    tabular_model: Literal["resnet", "linear"],
    rgnn_model: Literal["sage", "dbformer"],
    rgnn_layers: int,
    rgnn_aggr: str,
):
    return RDLModel(
        data=data,
        col_stats_dict=col_stats_dict,
        tabular_model=tabular_model,
        tabular_channels=channels,
        rgnn_model=rgnn_model,
        rgnn_channels=channels,
        rgnn_layers=rgnn_layers,
        rgnn_aggr=rgnn_aggr,
    )


def run_task_experiment(
    config: Dict[str, Any],
    full_data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    with_ray: bool = True,
    with_mlflow: bool = True,
    with_pretrained: bool = True,
):
    dataset_name: str = config["dataset_name"]
    task_name: str = config["task_name"]
    random_seed: int = config["seed"]

    lr: float = config["lr"]
    batch_size: int = config["batch_size"]
    num_neighbors: int = config["num_neighbors"]
    max_training_steps: int = config["max_training_steps"]
    finetune_backbone: bool = config["finetune_backbone"]

    channels: int = config["channels"]
    tabular_model: str = config["tabular_model"]
    rgnn_model: str = config["rgnn_model"]
    rgnn_layers: int = config["rgnn_layers"]
    rgnn_aggr: str = config["rgnn_aggr"]

    head_layers = config["head_layers"]
    head_channels = config["head_channels"]
    head_norm = config["head_norm"]
    head_dropout = config["head_dropout"]

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
        print("Device:", device)
    else:
        trial_name = (
            f"{dataset_name}_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    task = get_task(dataset_name, task_name)

    backbone = get_backbone(
        data=full_data,
        col_stats_dict=col_stats_dict,
        channels=channels,
        tabular_model=tabular_model,
        rgnn_model=rgnn_model,
        rgnn_layers=rgnn_layers,
        rgnn_aggr=rgnn_aggr,
    )
    if with_pretrained:
        backbone_model_path: str = config["backbone_model_path"]
        backbone.load_state_dict(torch.load(backbone_model_path, map_location=device))

    task_head = MLP(
        in_channels=channels,
        hidden_channels=head_channels,
        out_channels=1,
        num_layers=head_layers,
        norm=head_norm,
        act="relu",
        dropout=head_dropout,
    )

    if finetune_backbone:
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(task_head.parameters()), lr=lr
        )
    else:
        optimizer = torch.optim.Adam(task_head.parameters(), lr=lr)

    ligtning_model = LightningEntityTaskModel(
        backbone,
        task_head,
        optimizer,
        dataset_name=dataset_name,
        task_name=task_name,
        finetune_backbone=finetune_backbone,
    )

    loader_dict: Dict[str, NeighborLoader] = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            full_data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(rgnn_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            shuffle=split == "train",
        )
    if with_mlflow:
        mlflow_experiment: str = config["mlflow_experiment"]
        mlflow_uri: str = config["mlflow_uri"]
        logger = loggers.MLFlowLogger(
            experiment_name=mlflow_experiment,
            run_name=trial_name,
            tracking_uri=mlflow_uri,
        )

        logger.log_hyperparams(
            {
                "dataset": dataset_name,
                "task": task_name,
                "batch_size": batch_size,
                "lr": lr,
                "finetune_backbone": finetune_backbone,
                "tabular_model": tabular_model,
                "tabular_channels": channels,
                "rgnn_model": rgnn_model,
                "rgnn_channels": channels,
                "rgnn_layers": rgnn_layers,
                "rgnn_aggr": rgnn_aggr,
                "max_training_steps": max_training_steps,
                "head_channels": head_channels,
                "head_layers": head_layers,
                "head_norm": head_norm,
                "head_dropout": head_dropout,
            }
        )
    else:
        experiment_dir = config["experiment_dir"]
        logger = loggers.CSVLogger(save_dir=experiment_dir, name=trial_name)

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[
            callbacks.EarlyStopping(
                monitor=f"val_{ligtning_model.tune_metric}",
                mode="max" if ligtning_model.higher_is_better else "min",
                patience=10,
            ),
            callbacks.TQDMProgressBar(leave=True),
        ],
        num_sanity_val_steps=0,
        val_check_interval=min(100, len(loader_dict["train"])),
        enable_checkpointing=False,
        max_time=timedelta(hours=2),
    )
    try:
        trainer.fit(
            ligtning_model,
            train_dataloaders=loader_dict["train"],
            val_dataloaders=[loader_dict["val"], loader_dict["test"]],
            ckpt_path=None,
        )
    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        logger.finalize("failed")
        print(e)


def run_pretraining_experiment(
    config: Dict[str, Any],
    full_data: HeteroData,
    train_data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    with_ray: bool = True,
    with_mlflow: bool = True,
):
    dataset_name: str = config["dataset_name"]

    random_seed: int = config["seed"]
    lr: float = config["lr"]
    temperature: float = config["temperature"]
    max_training_steps: int = config["max_training_steps"]
    corrupt_prob: float = config["corrupt_prob"]
    schema_diameter: int = config["schema_diameter"]
    batch_size: float = config["batch_size"]
    with_neighbor_pretrain: bool = config["with_neighbor_pretrain"]

    channels: int = config["channels"]
    tabular_model: str = config["tabular_model"]
    rgnn_model: str = config["rgnn_model"]
    rgnn_layers: int = config["rgnn_layers"]
    rgnn_aggr: str = config["rgnn_aggr"]

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")

    if with_ray:
        context = ray_train.get_context()
        trial_name = context.get_trial_name()
        experiment_dir = context.get_trial_dir()

        resources = context.get_trial_resources().required_resources
        print(f"Resources: {resources}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.set_num_threads(1)
    else:
        experiment_dir = config["experiment_dir"]
        trial_name = f"{dataset_name}_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("Device:", device)

    corruptor = DBResampleCorruptor(
        train_data, corrupt_prob=corrupt_prob, distribution="uniform"
    )

    sample_tables = {
        "rel-amazon": "review",
        "rel-avito": "SearchInfo",
        "rel-f1": "results",
        "rel-stack": "posts",
        "rel-trial": "outcomes",
    }

    def get_loader(data: HeteroData, loader_type: Literal["hgt", "neighbor"]):
        if loader_type == "hgt":
            return HGTLoader(
                data,
                num_samples=[batch_size] * max(schema_diameter, 3),
                input_nodes=sample_tables[dataset_name],
                transform=corruptor.corrupt_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
        elif loader_type == "neighbor":
            return NeighborLoader(
                data,
                num_neighbors=[
                    min(batch_size, 16) // 2 ** (i + 1)
                    for i in range(max(schema_diameter, 3))
                ],
                input_nodes=sample_tables[dataset_name],
                input_time=data[sample_tables[dataset_name]].time,
                transform=corruptor.corrupt_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                time_attr="time",
                num_workers=1,
            )
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")

    pretrain_loaders = [get_loader(train_data, "hgt")]
    if with_neighbor_pretrain:
        pretrain_loaders.append(get_loader(train_data, "neighbor"))

    pretrain_val_loader = get_loader(full_data, "hgt")

    backbone = get_backbone(
        data=full_data,
        col_stats_dict=col_stats_dict,
        channels=channels,
        tabular_model=tabular_model,
        rgnn_model=rgnn_model,
        rgnn_layers=rgnn_layers,
        rgnn_aggr=rgnn_aggr,
    )

    pretrain_model = PretrainingModel(backbone, channels, temperature=temperature)

    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr)

    backbone_model_path = os.path.join(experiment_dir, f"best_{trial_name}.pt")

    ligtning_pretrain = LightningPretraining(
        pretrain_model,
        optimizer,
        model_save_path=backbone_model_path,
        with_neightbor_loader=with_neighbor_pretrain,
    )

    if with_mlflow:
        mlflow_experiment: str = config["mlflow_experiment"]
        mlflow_uri: str = config["mlflow_uri"]
        pretrain_run_name = f"{trial_name}_pretrain"

        logger = loggers.MLFlowLogger(
            experiment_name=mlflow_experiment,
            run_name=pretrain_run_name,
            tracking_uri=mlflow_uri,
        )

        logger.log_hyperparams(
            {
                "dataset": dataset_name,
                "batch_size": batch_size,
                "corrupt_prob": corrupt_prob,
                "temperature": temperature,
                "lr": lr,
                "sample_depth": max(schema_diameter, 3),
                "tabular_model": tabular_model,
                "tabular_channels": channels,
                "rgnn_model": rgnn_model,
                "rgnn_channels": channels,
                "rgnn_layers": rgnn_layers,
                "rgnn_aggr": rgnn_aggr,
                "max_training_steps": max_training_steps,
                "with_neighbor_pretrain": with_neighbor_pretrain,
            }
        )
    else:
        logger = loggers.CSVLogger(experiment_dir)

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, stopping_threshold=0.05
    )
    trainer = L.Trainer(
        max_steps=max_training_steps,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[early_stopping, callbacks.TQDMProgressBar(leave=True)],
        log_every_n_steps=15,
        num_sanity_val_steps=0,
        val_check_interval=min(50, len(pretrain_loaders[0])),
        limit_val_batches=50,
        enable_checkpointing=False,
        max_time=timedelta(hours=4),
    )
    try:
        if not with_neighbor_pretrain:
            pretrain_loaders = pretrain_loaders[0]

        trainer.fit(
            ligtning_pretrain,
            train_dataloaders=pretrain_loaders,
            val_dataloaders=pretrain_val_loader,
            ckpt_path=None,
        )
    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        logger.finalize("failed")
        print(e)

    if not os.path.exists(backbone_model_path):
        torch.save(ligtning_pretrain.model.backbone.state_dict(), backbone_model_path)

    if not with_ray:
        return

    task_names = get_task_names(dataset_name)
    task_names = [
        task_name
        for task_name in task_names
        if get_task(dataset_name, task_name).task_type != TaskType.LINK_PREDICTION
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_task_experiment,
                full_data=full_data,
                col_stats_dict=col_stats_dict,
            ),
            resources={"CPU": 1},
        ),
        run_config=ray_train.RunConfig(
            name=f"{trial_name}_tasks",
            storage_path=experiment_dir,
            stop={"time_total_s": 3600 * 4},
            log_to_file=True,
        ),
        param_space={
            "dataset_name": dataset_name,
            "task_name": tune.grid_search(task_names),
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            "seed": random_seed,
            # training config
            "max_training_steps": 2000,
            "lr": 0.005 if dataset_name != "rel-trial" else 0.0001,
            "finetune_backbone": tune.grid_search([False, True]),
            # sampling config
            "batch_size": 512,
            "num_neighbors": 128,
            # model config
            "backbone_model_path": backbone_model_path,
            "channels": channels,
            "tabular_model": tabular_model,
            "rgnn_model": rgnn_model,
            "rgnn_layers": rgnn_layers,
            "rgnn_aggr": rgnn_aggr,
            "head_layers": 2,
            "head_channels": 128,
            "head_norm": "batch_norm",
            "head_dropout": 0.0,
        },
    )
    tuner.fit()


def run_ray_tuner(
    dataset_name: str,
    rgnn_model: str,
    tabular_model: str,
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

    train_db = db.upto(dataset.val_timestamp)
    dataset.validate_and_correct_db(train_db)

    schema_cache_path = f"{cache_path}/attribute-schema.json"
    attribute_schema = get_attribute_schema(schema_cache_path, db)

    materialized_cache_dir = f"{cache_path}/materialized"
    full_data, col_stats_dict = make_pkey_fkey_graph(
        db,
        attribute_schema,
        text_embedder_cfg=get_text_embedder(),
        cache_dir=materialized_cache_dir + "/full",
    )
    train_data, _ = make_pkey_fkey_graph(
        train_db,
        attribute_schema,
        text_embedder_cfg=get_text_embedder(),
        cache_dir=materialized_cache_dir + "/train",
    )

    # resources = ray.available_resources()

    gpus_used = 0
    cpus_used = 1
    # if "GPU" in resources:
    #     batch_model_size = 4e9
    #     gpu_memory = max(
    #         [
    #             torch.cuda.get_device_properties(i).total_memory
    #             for i in range(torch.cuda.device_count())
    #         ]
    #     )
    #     gpus_used = batch_model_size / gpu_memory

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_pretraining_experiment,
                full_data=full_data,
                train_data=train_data,
                col_stats_dict=col_stats_dict,
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
            max_concurrent_trials=num_cpus // 2,
        ),
        param_space={
            "dataset_name": dataset_name,
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            "schema_diameter": get_dataset_info(dataset_name).schema_diameter.item(),
            "seed": tune.randint(0, 1000),
            # training config
            "max_training_steps": 2000,
            "lr": 0.001,  # tune.choice([0.001, 0.005]),
            "temperature": 1.0,  # tune.grid_search([1.0]),
            "corrupt_prob": tune.grid_search([0.2, 0.4, 0.6]),
            "with_neighbor_pretrain": tune.grid_search([False, True]),
            # sampling config
            "batch_size": 64,
            # model config
            "channels": 128,
            "tabular_model": tabular_model,  # tune.grid_search(["resnet", "linear"]),
            "rgnn_model": rgnn_model,
            "rgnn_layers": tune.grid_search([2, 3]),
            "rgnn_aggr": tune.grid_search(["mean", "sum"]),
        },
    )
    tuner.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--rgnn_model", choices=["sage", "dbformer"], default="sage")
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], default=None)
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

    run_ray_tuner(
        dataset_name,
        rgnn_model=args.rgnn_model,
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
        random_seed=args.seed,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
    )
