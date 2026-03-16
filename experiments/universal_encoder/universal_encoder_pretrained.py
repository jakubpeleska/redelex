from typing import Optional, Any
from pathlib import Path

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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.utilities.model_summary import ModelSummary

from torch_frame.data import StatType

from relbench.base import TaskType
from relbench.tasks import get_task

from redelex.loaders import ComposedLoader
from redelex.tasks import mixins

from experiments.universal_encoder.utils import (
    get_dataset_data,
    get_task_data,
    get_hyperparams_logging,
    get_text_embedder,
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
    # "ctu-adventureworks": ["adventureworks-temporal"],
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

    model_save_dir = Path(config["model_save_dir"])
    model_save_dir = f"{model_save_dir}/{trial_name}"
    config["model_save_dir"] = model_save_dir

    text_embedder = get_text_embedder(
        config["text_embedder_name"], device=torch.device("cpu")
    )

    loaders = {"train": {}, "val": {}}
    heads = torch.nn.ModuleDict()

    shared_gnn = config.get("shared_gnn", True)
    config["shared_gnn"] = shared_gnn
    if shared_gnn:
        config["gnn_type"] = "homogeneous"

    gnn_type = config.get("gnn_type", "homogeneous")

    leave_out_dataset = config.get("leave_out_dataset", [])
    if not isinstance(leave_out_dataset, list):
        leave_out_dataset = [leave_out_dataset]

    for dataset_name, task_names in PRETRAIN_TASKS.items():
        if dataset_name in leave_out_dataset:
            continue

        if "rel-all" in leave_out_dataset and dataset_name.startswith("rel-"):
            continue

        if "ctu-all" in leave_out_dataset and dataset_name.startswith("ctu-"):
            continue

        target = None
        if len(task_names) == 1:
            task = get_task(dataset_name, task_names[0])
            if isinstance(task, mixins.ImputeEntityTaskMixin):
                target = (task.entity_table, task.target_col)

        data, col_stats_dict, tensor_stats_dict, name_embeddings_dict = get_dataset_data(
            dataset_name,
            f"{config['cache_path']}/{dataset_name}",
            text_embedder,
            device,
            target=target,
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

            if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:
                out_channels = 1
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                out_channels = len(task.stats()[StatType.COUNT][0])

            if shared_gnn:
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
        split: ComposedLoader(loaders[split], mode="minimum") for split in ["train", "val"]
    }

    row_encoder_config = {
        "col_channels": config["col_channels"],
        "out_channels": config["gnn_channels"],
        "embedding_dim": text_embedder.embedding_dim,
        "encoder_heads": config["row_encoder_heads"],
        "encoder_layers": config["row_encoder_layers"],
        "encoder_dropout": config["row_encoder_dropout"],
        "use_stype_emb": config.get("use_stype_emb", True),
        "use_name_emb": config.get("use_name_emb", True),
        "use_stats_emb": config.get("use_stats_emb", True),
    }

    row_encoder = RowEncoder(**row_encoder_config)

    if shared_gnn:
        global_gnn = HomogeneousGNN(
            gnn_channels=config["gnn_channels"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
        )
    else:
        global_gnn = None

    row_encoder = row_encoder.to(device)
    heads = heads.to(device)
    if global_gnn is not None:
        global_gnn = global_gnn.to(device)

    optim_params = [*row_encoder.parameters(), *heads.parameters()]
    if global_gnn is not None:
        optim_params.extend(list(global_gnn.parameters()))

    optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=0.1)

    max_training_steps: int = config["max_training_steps"]
    # Linear warmup for 10% of training, then cosine decay
    warmup_sch = LinearLR(
        optimizer, start_factor=0.01, total_iters=max_training_steps // 10
    )
    decay_sch = CosineAnnealingLR(
        optimizer, T_max=max_training_steps - max_training_steps // 10
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sch, decay_sch], milestones=[max_training_steps // 10]
    )

    gnn_config = None
    if global_gnn is not None:
        gnn_config = {
            "gnn_channels": config["gnn_channels"],
            "gnn_layers": config["gnn_layers"],
            "gnn_aggr": config["gnn_aggr"],
        }

    lightning_model = LightningMultiTaskWrapper(
        row_encoder=row_encoder,
        gnn=global_gnn,
        heads=heads,
        optimizer=optimizer,
        gnn_type=gnn_type,
        row_encoder_config=row_encoder_config,
        gnn_config=gnn_config,
        scheduler=scheduler,
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

    save_pretrained_callback = SavePretrainedCallback(
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
        lightning_model.save_pretrained(f"{model_save_dir}/final_model.pt")
        print(f"Force saved pretrained checkpoint to: {model_save_dir}/final_model.pt")

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
    model_save_dir: str = "./models",
    cache_dir: str = ".cache",
    leave_out_dataset: str = "none",
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
            trial_name_creator=lambda trial: (
                f"leaveout_{leave_out_dataset}_tablayers_{trial.config['row_encoder_layers']}_{trial.trial_id}"
            ),
            trial_dirname_creator=lambda trial: trial.trial_id,
            max_concurrent_trials=num_cpus,
        ),
        param_space={
            "seed": tune.randint(0, 1000),
            "text_embedder_name": "glove",
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            "max_training_steps": 50000,
            "limit_train_batches": 500,
            "limit_val_batches": 500,
            "lr": 0.001,
            "batch_size": 128,
            "num_neighbors": 16,
            "col_channels": 512,
            "gnn_channels": 512
            if shared_gnn
            else (128 if gnn_type == "heterogeneous" else 256),
            "row_encoder_layers": tune.grid_search([1, 4, 2]),
            "row_encoder_heads": 8,
            "row_encoder_dropout": 0.1,
            "gnn_layers": 2,
            "gnn_aggr": "sum",
            "head_norm": "batch_norm",
            "cache_path": Path(cache_dir).absolute(),
            "model_save_dir": (Path(model_save_dir).absolute()),
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
    parser.add_argument("--model_save_dir", type=str, default="./models")
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
        model_save_dir=args.model_save_dir,
        leave_out_dataset=args.leave_out_dataset,
        use_stype_emb=not args.no_stype_emb,
        use_name_emb=not args.no_name_emb,
        use_stats_emb=not args.no_stats_emb,
        shared_gnn=args.shared_gnn,
        gnn_type=args.gnn_type,
    )
