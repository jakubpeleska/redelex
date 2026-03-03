from redelex.transforms import AttachValuesTransform
from torch_geometric.data import HeteroData
from typing import Optional, Any

import traceback
import os
import random
from datetime import datetime, timedelta

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

from argparse import ArgumentParser

import ray
from ray import tune, train as ray_train

import numpy as np

import torch

import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.utilities.model_summary import ModelSummary

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_frame.data import StatType

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from redelex.data import make_pkey_fkey_graph, make_tensor_stats_dict
from redelex.transforms import AttachDictTransform
from redelex.loaders import ComposedLoader
from redelex.tasks import mixins, is_temporal_task

from experiments.universal_encoder.utils import (
    get_attribute_schema,
    get_hyperparams_logging,
    get_text_embedder,
    get_table_input,
)

from experiments.universal_encoder.models import (
    RowEncoder,
    HeterogeneousGNN,
    HomogeneousGNN,
    HeterogeneousTaskHead,
    HomogeneousTaskHead,
    TaskGNNAndHead,
)
from experiments.universal_encoder.lightning_wrappers import (
    LightningMultiTaskWrapper,
    SavePretrainedCallback,
)

PRETRAIN_TASKS = {
    "rel-amazon": ["user-churn", "user-ltv", "item-churn", "item-ltv"],
    "rel-avito": ["ad-ctr", "user-visits", "user-clicks"],
    "rel-f1": ["driver-dnf", "driver-top3", "driver-position"],
    "rel-stack": ["user-engagement", "user-badge", "post-votes"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success"],
    "ctu-adventureworks": ["adventureworks-temporal"],
    "ctu-employee": ["employee-temporal"],
    "ctu-ergastf1": ["ergastf1-original"],
    "ctu-expenditures": ["expenditures-original"],
    "ctu-fnhk": ["fnhk-temporal"],
    "ctu-gosales": ["gosales-temporal"],
    "ctu-grants": ["grants-temporal"],
    "ctu-lahman": ["lahman-temporal"],
    "ctu-movielens": ["movielens-original"],
    "ctu-restbase": ["restbase-original"],
    "ctu-sakila": ["sakila-temporal"],
    "ctu-sales": ["sales-original"],
    "ctu-sap": ["sap-sales-temporal"],
    "ctu-seznam": ["seznam-temporal"],
}


def get_dataset_data(
    dataset_name: str, cache_path: str, text_embedder, device: torch.device
):
    dataset = get_dataset(dataset_name)
    db = dataset.get_db.__wrapped__(dataset, False)

    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
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
    for split in ["train", "val", "test"]:
        table = task.get_table.__wrapped__(task, split, mask_input_cols=False)
        if task.task_type == TaskType.REGRESSION:
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
            shuffle=split == "train",
            num_workers=0,
            drop_last=split == "train",
        )
    return task, loader_dict


def run_task_experiment(
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

    text_embedder = get_text_embedder(
        config["text_embedder_name"], device=torch.device("cpu")
    )

    loaders = {"train": {}, "val": {}, "test": {}}
    heads = torch.nn.ModuleDict()

    shared_gnn = config.get("shared_gnn", False)
    gnn_type = config.get("gnn_type", "heterogeneous")

    leave_out_dataset = config.get("leave_out_dataset", [])
    if not isinstance(leave_out_dataset, list):
        leave_out_dataset = [leave_out_dataset]

    for dataset_name, task_names in PRETRAIN_TASKS.items():
        if dataset_name in leave_out_dataset:
            continue

        data, col_stats_dict, tensor_stats_dict, name_embeddings_dict = get_dataset_data(
            dataset_name,
            f"{config['cache_path']}/{dataset_name}",
            text_embedder,
            device,
        )

        for task_name in task_names:
            task = get_task(dataset_name, task_name)
            name = f"{dataset_name}-{task_name}"
            task, loader_dict = get_task_data(
                task,
                name,
                data,
                name_embeddings_dict,
                tensor_stats_dict,
                col_stats_dict,
                config,
            )
            loaders["train"][name] = loader_dict["train"]
            loaders["val"][name] = loader_dict["val"]
            loaders["test"][name] = loader_dict["test"]

            if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:
                out_channels = 1
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                out_channels = len(task.stats()[StatType.COUNT][0])

            if shared_gnn:
                if gnn_type == "heterogeneous":
                    task_head = HeterogeneousTaskHead(
                        in_channels=config["gnn_channels"],
                        out_channels=out_channels,
                        head_norm=config["head_norm"],
                    )
                else:
                    task_head = HomogeneousTaskHead(
                        in_channels=config["gnn_channels"],
                        out_channels=out_channels,
                        head_norm=config["head_norm"],
                    )
                heads[name] = task_head
            else:
                if gnn_type == "heterogeneous":
                    task_gnn = HeterogeneousGNN(
                        node_types=data.node_types,
                        edge_types=data.edge_types,
                        gnn_channels=config["gnn_channels"],
                        gnn_layers=config["gnn_layers"],
                        gnn_aggr=config["gnn_aggr"],
                    )
                    task_head = HeterogeneousTaskHead(
                        in_channels=config["gnn_channels"],
                        out_channels=out_channels,
                        head_norm=config["head_norm"],
                    )
                else:
                    task_gnn = HomogeneousGNN(
                        gnn_channels=config["gnn_channels"],
                        gnn_layers=config["gnn_layers"],
                        gnn_aggr=config["gnn_aggr"],
                    )
                    task_head = HomogeneousTaskHead(
                        in_channels=config["gnn_channels"],
                        out_channels=out_channels,
                        head_norm=config["head_norm"],
                    )
                heads[name] = TaskGNNAndHead(
                    gnn=task_gnn, head=task_head, gnn_type=gnn_type
                )

    loader_dict = {
        split: ComposedLoader(loaders[split], mode="minimum")
        for split in ["train", "val", "test"]
    }

    tabular_encoder_config = {
        "col_channels": config["col_channels"],
        "out_channels": config["gnn_channels"],
        "embedding_dim": text_embedder.embedding_dim,
        "tabular_encoder_heads": config["tabular_encoder_heads"],
        "tabular_encoder_layers": config["tabular_encoder_layers"],
        "tabular_encoder_dropout": config["tabular_encoder_dropout"],
        "use_stype_emb": config.get("use_stype_emb", True),
        "use_name_emb": config.get("use_name_emb", True),
        "use_stats_emb": config.get("use_stats_emb", True),
    }

    tabular_encoder = RowEncoder(**tabular_encoder_config)

    if shared_gnn:
        if gnn_type == "heterogeneous":
            global_gnn = HeterogeneousGNN(
                node_types=data.node_types,  # Assumes node_types match or is handled (ideal for homogeneous, tricky for heterogeneous shared)
                edge_types=data.edge_types,
                gnn_channels=config["gnn_channels"],
                gnn_layers=config["gnn_layers"],
                gnn_aggr=config["gnn_aggr"],
            )
        else:
            global_gnn = HomogeneousGNN(
                gnn_channels=config["gnn_channels"],
                gnn_layers=config["gnn_layers"],
                gnn_aggr=config["gnn_aggr"],
            )
    else:
        global_gnn = None

    tabular_encoder = tabular_encoder.to(device)
    heads = heads.to(device)
    if global_gnn is not None:
        global_gnn = global_gnn.to(device)

    optim_params = [*tabular_encoder.parameters(), *heads.parameters()]
    if global_gnn is not None:
        optim_params.extend(list(global_gnn.parameters()))

    optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=0.1)

    gnn_config = None
    if global_gnn is not None:
        gnn_config = {
            "gnn_channels": config["gnn_channels"],
            "gnn_layers": config["gnn_layers"],
            "gnn_aggr": config["gnn_aggr"],
            "gnn_type": gnn_type,
        }

    lightning_model = LightningMultiTaskWrapper(
        tabular_encoder=tabular_encoder,
        gnn=global_gnn,
        heads=heads,
        optimizer=optimizer,
        gnn_type=gnn_type,
        tabular_encoder_config=tabular_encoder_config,
        gnn_config=gnn_config,
    )

    model_summary = ModelSummary(lightning_model, max_depth=2)

    config["model_parameters"] = model_summary.total_parameters
    config["model_size_MB"] = model_summary.model_size

    max_training_steps: int = config["max_training_steps"]
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

    save_pretrained_callback = SavePretrainedCallback(
        save_path=config["model_save_path"],
        monitor="val_loss_epoch",
        mode="min",
    )

    trainer = L.Trainer(
        max_steps=max_training_steps,
        max_epochs=config.get("max_epochs", None),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        accelerator=device.type,
        devices=1,
        logger=logger,
        callbacks=[save_pretrained_callback],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        max_time=timedelta(hours=12),
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
    )
    try:
        trainer.fit(
            lightning_model,
            train_dataloaders=loader_dict["train"],
            val_dataloaders=loader_dict["val"],
            ckpt_path=None,
        )

        # Fallback to make sure the latest model is saved if no validation happened yet
        lightning_model.save_pretrained(config["model_save_path"])
        print(f"Force saved pretrained checkpoint to: {config['model_save_path']}")

    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        stack_trace = traceback.format_exc()
        logger.log_hyperparams({"stack_trace": stack_trace})
        print(stack_trace)
        logger.finalize("failed")


def run_ray_tuner(
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "pelesjak_test_experiment",
    num_samples: Optional[int] = 1,
    num_gpus: int = 0,
    num_cpus: int = 1,
    random_seed: int = 42,
    model_save_path: str = "pretrained_model.pt",
    cache_dir: str = ".cache",
    leave_out_dataset: Optional[str] = None,
    use_stype_emb: bool = True,
    use_name_emb: bool = True,
    use_stats_emb: bool = True,
    shared_gnn: bool = False,
    gnn_type: str = "heterogeneous",
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

    tuner = tune.Tuner(
        tune.with_resources(
            run_task_experiment,
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
            trial_name_creator=lambda trial: f"pretrain_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
            max_concurrent_trials=num_cpus,
        ),
        param_space={
            "seed": tune.randint(0, 1000),
            "text_embedder_name": "glove",
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            "max_training_steps": 30000,
            "lr": 0.0001,
            "batch_size": 128,
            "num_neighbors": 16,
            "col_channels": 512,
            "gnn_channels": 128,
            "tabular_encoder_layers": tune.grid_search([1, 2, 4, 8]),
            "tabular_encoder_heads": 8,
            "tabular_encoder_dropout": 0.1,
            "gnn_layers": 2,
            "gnn_aggr": "sum",
            "head_norm": "batch_norm",
            "cache_path": cache_dir,
            "model_save_path": model_save_path,
            "leave_out_dataset": leave_out_dataset,
            "use_stype_emb": use_stype_emb,
            "use_name_emb": use_name_emb,
            "use_stats_emb": use_stats_emb,
            "shared_gnn": shared_gnn,
            "gnn_type": gnn_type,
        },
    )
    tuner.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--model_save_path", type=str, default="pretrained_model.pt")
    parser.add_argument(
        "--leave_out_dataset",
        type=str,
        default=None,
        help="Dataset to leave out during pretraining",
    )

    parser.add_argument(
        "--shared_gnn",
        action="store_true",
        help="If set, uses a shared GNN backbone across all tasks, and only simple Task Head per task.",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="heterogeneous",
        choices=["heterogeneous", "homogeneous"],
        help="Type of GNN to instantiate (either shared or task specific).",
    )

    parser.add_argument(
        "--no_stype_emb", action="store_true", help="Ablate stype embeddings"
    )
    parser.add_argument("--no_name_emb", action="store_true", help="Ablate name embeddings")
    parser.add_argument(
        "--no_stats_emb", action="store_true", help="Ablate stats embeddings"
    )

    args = parser.parse_args()
    print(args)

    run_ray_tuner(
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
        cache_dir=args.cache_dir,
        model_save_path=args.model_save_path,
        leave_out_dataset=args.leave_out_dataset,
        use_stype_emb=not args.no_stype_emb,
        use_name_emb=not args.no_name_emb,
        use_stats_emb=not args.no_stats_emb,
        shared_gnn=args.shared_gnn,
        gnn_type=args.gnn_type,
    )
