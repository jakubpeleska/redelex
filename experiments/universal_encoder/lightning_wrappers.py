from pathlib import Path
from typing import Optional, Any
from collections import defaultdict

import torch

from torch_geometric.data import HeteroData


import lightning as L

from torchmetrics.aggregation import MinMetric, MeanMetric

from relbench.base import TaskType

from redelex.nn.train.utils import get_loss

from .models import RowEncoder


class LightningMultiTaskWrapper(L.LightningModule):
    def __init__(
        self,
        tabular_encoder: RowEncoder,
        gnn: Optional[torch.nn.Module],
        heads: torch.nn.ModuleDict,
        optimizer: torch.optim.Optimizer,
        gnn_type: str = "homogeneous",
        tabular_encoder_config: Optional[dict[str, Any]] = None,
        gnn_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        self.tabular_encoder = tabular_encoder
        self.gnn = gnn
        self.heads = heads
        self.optimizer = optimizer
        self.gnn_type = gnn_type

        self.tabular_encoder_config = tabular_encoder_config
        self.gnn_config = gnn_config

        self.loss_fn_dict = {
            TaskType.BINARY_CLASSIFICATION: get_loss(TaskType.BINARY_CLASSIFICATION),
            TaskType.MULTICLASS_CLASSIFICATION: get_loss(
                TaskType.MULTICLASS_CLASSIFICATION
            ),
            TaskType.REGRESSION: get_loss(TaskType.REGRESSION),
        }

        self.train_loss_dict = defaultdict(MeanMetric)
        self.val_loss_dict = defaultdict(MeanMetric)
        self.val_best_loss = MinMetric()
        self.val_best_loss.update(float("inf"))

    def forward(self, batch: HeteroData) -> torch.Tensor:
        task_name = batch["task_name"]
        task_type = batch["task_type"]
        entity_table = batch["entity_table"]

        seed_time_size = (
            batch[entity_table].seed_time.size(0)
            if hasattr(batch[entity_table], "seed_time")
            else batch[entity_table].batch_size
        )

        x_dict = self.tabular_encoder(batch)

        if self.gnn is not None:
            if self.gnn_type == "homogeneous":
                x, node_slices, edge_slices = self.gnn(x_dict, batch, entity_table)
                pred = self.heads[task_name](x, node_slices, entity_table, seed_time_size)
            else:
                x_dict = self.gnn(x_dict, batch, entity_table)
                pred = self.heads[task_name](x_dict, entity_table, seed_time_size)
        else:
            # If no shared GNN is provided, the task head encapsulates the task-specific GNN
            pred = self.heads[task_name](x_dict, batch, entity_table, seed_time_size)

        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target = batch[entity_table].y.long()
        else:
            target = batch[entity_table].y.float()
            pred = pred.view(-1)

        loss_fn = self.loss_fn_dict[task_type]

        loss = loss_fn(pred, target)
        return loss

    def training_step(self, batch: HeteroData, batch_idx):
        loss = self(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss is NaN or Inf, check the model and data.")
        loss_value = loss.detach().cpu()
        self.train_loss_dict["train_loss"].update(loss_value)
        self.train_loss_dict[f"train_loss_{batch['task_name']}"].update(loss_value)
        metrics_dict = {
            "train_loss": self.train_loss_dict["train_loss"].compute().item(),
            f"train_loss_{batch['task_name']}": self.train_loss_dict[
                f"train_loss_{batch['task_name']}"
            ]
            .compute()
            .item(),
        }

        self.log_dict(metrics_dict, prog_bar=True, batch_size=1)

        return loss

    def on_train_epoch_end(self):
        train_loss_metrics: dict[str, float] = {}
        for k, v in self.train_loss_dict.items():
            train_loss_metrics[f"{k}_epoch"] = v.compute().item()
            v.reset()
        self.log_dict(train_loss_metrics, prog_bar=True, logger=True)

    def validation_step(self, batch: HeteroData, batch_idx):
        loss = self(batch)
        loss_value = loss.detach().cpu()
        self.val_loss_dict["val_loss"].update(loss_value)
        self.val_loss_dict[f"val_loss_{batch['task_name']}"].update(loss_value)
        metrics_dict = {
            "val_loss": self.val_loss_dict["val_loss"].compute().item(),
            f"val_loss_{batch['task_name']}": self.val_loss_dict[
                f"val_loss_{batch['task_name']}"
            ]
            .compute()
            .item(),
        }

        self.log_dict(metrics_dict, prog_bar=True, batch_size=1)

        return loss

    def on_validation_epoch_end(self):
        val_loss_metrics: dict[str, float] = {}
        for k, v in self.val_loss_dict.items():
            val_loss_metrics[f"{k}_epoch"] = v.compute().item()
            v.reset()
        self.log_dict(val_loss_metrics, prog_bar=True, logger=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint["state_dict"].keys())
        for k in keys:
            if k.startswith("heads."):
                del checkpoint["state_dict"][k]

    def configure_optimizers(self):
        return self.optimizer

    def save_pretrained(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "tabular_encoder_state_dict": self.tabular_encoder.state_dict(),
            "tabular_encoder_config": self.tabular_encoder_config,
        }
        if self.gnn is not None:
            save_dict["gnn_state_dict"] = self.gnn.state_dict()
            save_dict["gnn_config"] = self.gnn_config

        torch.save(save_dict, path)


class SavePretrainedCallback(L.Callback):
    def __init__(self, save_path: str, monitor: str = "val_loss_epoch", mode: str = "min"):
        super().__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningMultiTaskWrapper
    ):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            # If the metric is not present yet (e.g. sanity check), just return
            return

        current_score = (
            current_score.item()
            if isinstance(current_score, torch.Tensor)
            else current_score
        )

        is_best = (self.mode == "min" and current_score < self.best_score) or (
            self.mode == "max" and current_score > self.best_score
        )

        if is_best:
            self.best_score = current_score
            pl_module.save_pretrained(self.save_path)
