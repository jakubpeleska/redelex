import traceback
from typing import Optional, Any

import os
import random
from datetime import datetime, timedelta
import sys
from pathlib import Path

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

from torch_geometric.data import HeteroData

from torch_frame.data import StatType

from relbench.base import TaskType
from relbench.tasks import get_task

sys.path.append(".")

from redelex.nn.train.lightning import LightningEntityTaskWrapper
from redelex.tasks import mixins

from experiments.universal_encoder.utils import (
    get_dataset_data,
    get_task_data,
    get_hyperparams_logging,
    get_text_embedder,
)

from experiments.universal_encoder.models import (
    RowEncoder,
    HomogeneousGNN,
    HeterogeneousGNN,
    HomogeneousTaskHead,
    HeterogeneousTaskHead,
    TaskGNNAndHead,
)


# class DynamicValidationCallback(L.Callback):
#     def __init__(self, val_check_interval: int = 100, val_check_list: list[int] = []):
#         super().__init__()
#         self.val_check_interval = val_check_interval
#         self.val_check_list = val_check_list

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if (
#             trainer.global_step % self.val_check_interval == 0
#             or trainer.global_step in self.val_check_list
#         ) and trainer.global_step > 0:
#             trainer.validate(model=pl_module, dataloaders=trainer.val_dataloaders, ckpt_path=None)


class SupervisedWrapper(torch.nn.Module):
    def __init__(
        self, row_encoder: RowEncoder, task_gnn_and_head: TaskGNNAndHead, entity_table: str
    ):
        super().__init__()
        self.row_encoder = row_encoder
        self.task_gnn_and_head = task_gnn_and_head
        self.entity_table = entity_table

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.row_encoder.parameters()):
            self.row_encoder.eval()
        if (
            hasattr(self.task_gnn_and_head, "gnn")
            and self.task_gnn_and_head.gnn is not None
        ):
            if not any(p.requires_grad for p in self.task_gnn_and_head.gnn.parameters()):
                self.task_gnn_and_head.gnn.eval()
        return self

    def forward(self, batch: HeteroData, entity_table: Optional[str] = None):
        target_table = entity_table or self.entity_table
        x_dict = self.row_encoder(batch)
        return self.task_gnn_and_head(x_dict, batch, target_table)


def run_task_experiment(
    config: dict[str, Any],
    with_ray: bool = True,
    with_mlflow: bool = True,
):
    pretrained_checkpoint: Optional[str] = config.get("pretrained_checkpoint", None)
    use_pretrained_row_encoder = pretrained_checkpoint is not None and config.get(
        "pretrained_row_encoder", False
    )
    use_pretrained_gnn = pretrained_checkpoint is not None and config.get(
        "pretrained_gnn", False
    )
    finetune_pretrained = pretrained_checkpoint is not None and config.get(
        "finetune_pretrained", False
    )

    dataset_name: str = config["dataset_name"]
    task_name: str = config["task_name"]
    random_seed: int = config["seed"]

    lr: float = config["lr"]
    cache_path: str = f"{config['cache_dir']}/{dataset_name}"

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

    task: mixins.BaseTask = get_task(dataset_name, task_name)

    if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:
        out_channels = 1
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = len(task.stats()[StatType.COUNT][0])

    config["out_channels"] = out_channels

    text_embedder = get_text_embedder(
        config["text_embedder_name"], device=torch.device("cpu")
    )

    target = None
    if isinstance(task, mixins.ImputeEntityTaskMixin):
        target = (task.entity_table, task.target_col)

    data, col_stats_dict, tensor_stats_dict, name_embeddings_dict = get_dataset_data(
        dataset_name, cache_path, text_embedder, device, target=target
    )
    task, loader_dict = get_task_data(
        task,
        f"{dataset_name}-{task_name}",
        data,
        name_embeddings_dict,
        tensor_stats_dict,
        col_stats_dict,
        config,
        splits=["train", "val", "test"],
        normalize_target=False,
    )

    row_encoder_config = None
    gnn_config = None
    checkpoint = None
    row_encoder_state_dict = None
    gnn_state_dict = None

    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        if use_pretrained_row_encoder:
            row_encoder_config = checkpoint.get("row_encoder_config", None)
            row_encoder_state_dict = checkpoint.get("row_encoder_state_dict", None)
            config["col_channels"] = row_encoder_config["col_channels"]
            config["gnn_channels"] = row_encoder_config["out_channels"]
            config["tabular_encoder_heads"] = row_encoder_config["encoder_heads"]
            config["tabular_encoder_layers"] = row_encoder_config["encoder_layers"]
            config["tabular_encoder_dropout"] = row_encoder_config["encoder_dropout"]
            config["use_stype_emb"] = row_encoder_config["use_stype_emb"]
            config["use_name_emb"] = row_encoder_config["use_name_emb"]
            config["use_stats_emb"] = row_encoder_config["use_stats_emb"]
            assert row_encoder_state_dict is not None, (
                "Pretrained row encoder state dict not found"
            )
        if use_pretrained_gnn:
            gnn_config = checkpoint.get("gnn_config", None)
            gnn_state_dict = checkpoint.get("gnn_state_dict", None)
            config["gnn_channels"] = gnn_config["gnn_channels"]
            config["gnn_layers"] = gnn_config["gnn_layers"]
            config["gnn_aggr"] = gnn_config["gnn_aggr"]
            assert gnn_state_dict is not None, "Pretrained GNN state dict not found"

    if use_pretrained_row_encoder:
        print("Using row_encoder_config from checkpoint")
        row_encoder = RowEncoder(**row_encoder_config)
        row_encoder.load_state_dict(row_encoder_state_dict, strict=True)
        if not finetune_pretrained:
            row_encoder.requires_grad_(False)
    else:
        row_encoder = RowEncoder(
            col_channels=config["col_channels"],
            out_channels=config["gnn_channels"],
            embedding_dim=text_embedder.embedding_dim,
            encoder_heads=config.get("tabular_encoder_heads", 4),
            encoder_layers=config.get("tabular_encoder_layers", 2),
            encoder_dropout=config.get("tabular_encoder_dropout", 0.1),
            use_stype_emb=config.get("use_stype_emb", True),
            use_name_emb=config.get("use_name_emb", True),
            use_stats_emb=config.get("use_stats_emb", True),
        )

    if use_pretrained_gnn:
        gnn = HomogeneousGNN(
            gnn_channels=config["gnn_channels"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
        )
        gnn.load_state_dict(gnn_state_dict, strict=True)
        if not finetune_pretrained:
            gnn.requires_grad_(False)
        head = HomogeneousTaskHead(
            in_channels=gnn_config["gnn_channels"],
            out_channels=config["out_channels"],
            head_norm=config["head_norm"],
        )
        task_gnn_and_head = TaskGNNAndHead(gnn=gnn, head=head, gnn_type="homogeneous")
    else:
        task_gnn = HeterogeneousGNN(
            node_types=data.node_types,
            edge_types=data.edge_types,
            gnn_channels=config["gnn_channels"],
            gnn_layers=config["gnn_layers"],
            gnn_aggr=config["gnn_aggr"],
        )
        head = HeterogeneousTaskHead(
            in_channels=config["gnn_channels"],
            out_channels=config["out_channels"],
            head_norm=config["head_norm"],
        )
        task_gnn_and_head = TaskGNNAndHead(
            gnn=task_gnn, head=head, gnn_type="heterogeneous"
        )

    model = SupervisedWrapper(row_encoder, task_gnn_and_head, task.entity_table)
    model = model.to(device)

    optimizable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizable_params, lr=lr, weight_decay=0.1)

    scheduler = None
    boosted_training = config.get("boosted_training", False)
    config["boosted_training"] = boosted_training
    if boosted_training:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=10.0, end_factor=1.0, total_iters=100
        )

    lightning_model = LightningEntityTaskWrapper(
        model=model, optimizer=optimizer, task=task, scheduler=scheduler
    )

    model_summary = ModelSummary(lightning_model, max_depth=2)

    config["model_parameters"] = model_summary.total_parameters
    config["model_size_MB"] = model_summary.model_size

    max_training_steps: int = config.get("max_training_steps", 10000)
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
        max_steps=config.get("max_training_steps", 10000),
        max_epochs=config.get("max_epochs", None),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        accelerator=device.type,
        # callbacks=[
        #     DynamicValidationCallback(val_check_interval=100, val_check_list=[1, 10, 50])
        # ],
        devices=1,
        logger=logger,
        num_sanity_val_steps=0,
        val_check_interval=min(100, len(loader_dict["train"]) // 2),
        enable_checkpointing=False,
        max_time=timedelta(hours=2),
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
    pretrained_checkpoint: Optional[list[str]] = None,
    pretrained_row_encoder: bool = False,
    pretrained_gnn: bool = False,
    finetune_pretrained: bool = False,
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
            trial_name_creator=lambda trial: f"{dataset_name}_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
            max_concurrent_trials=num_cpus,
        ),
        param_space={
            "dataset_name": dataset_name,
            "task_name": task_name,
            "seed": tune.randint(0, 1000),
            "text_embedder_name": "glove",
            # logging config
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            # training config
            "max_training_steps": 5000,
            "lr": 0.001,
            "boosted_training": False,
            # sampling config
            "batch_size": 128,
            "num_neighbors": 16,
            # pretraining checkpoint
            "pretrained_checkpoint": tune.grid_search(
                [str(Path(p).absolute()) for p in pretrained_checkpoint]
            )
            if pretrained_checkpoint
            else None,
            "pretrained_row_encoder": True,
            "pretrained_gnn": False,
            "finetune_pretrained": False,
            "cache_dir": str(Path(cache_dir).absolute()),
            # model config
            "col_channels": 512,
            "gnn_channels": 128,
            "tabular_encoder_heads": 8,
            "tabular_encoder_dropout": 0.1,
            "tabular_encoder_layers": 1,
            "gnn_layers": 2,
            "gnn_aggr": "sum",
            "head_norm": "batch_norm",
        },
    )
    tuner.fit()


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
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        nargs="+",
        default=None,
        help="Path to pretrained model checkpoint(s)",
    )
    parser.add_argument("--pretrained_row_encoder", action="store_true")
    parser.add_argument("--pretrained_gnn", action="store_true")
    parser.add_argument("--finetune_pretrained", action="store_true")

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
        pretrained_checkpoint=args.pretrained_checkpoint,
        pretrained_row_encoder=args.pretrained_row_encoder,
        pretrained_gnn=args.pretrained_gnn,
        finetune_pretrained=args.finetune_pretrained,
    )
