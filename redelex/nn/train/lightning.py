import copy
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from relbench.base import EntityTask, TaskType
from torchmetrics.aggregation import MaxMetric, MeanMetric, MinMetric

from .utils import get_loss, get_metrics


class LightningEntityTaskWrapper(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        task: EntityTask,
        lr_scheduler_config: Optional[dict] = None,
    ):
        super().__init__()

        self.model = model
        self.task = task
        self.loss_fn = get_loss(self.task.task_type)
        self.val_metrics, self.tune_metric, self.higher_is_better = get_metrics(
            self.task.task_type
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)
        self.optimizer = optimizer
        self.lr_scheduler_config = lr_scheduler_config
        self.scheduler = (
            lr_scheduler_config["scheduler"] if lr_scheduler_config is not None else None
        )

        self.train_loss = MeanMetric().requires_grad_(False)
        self.best_tune_metric = MaxMetric() if self.higher_is_better else MinMetric()
        self.best_tune_metric.requires_grad_(False)
        if self.higher_is_better:
            self.best_tune_metric.update(float("-inf"))
        else:
            self.best_tune_metric.update(float("inf"))

        self.first_val = True

    def forward(self, batch):
        pred = self.model(batch, self.task.entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if pred.size(0) != batch[self.task.entity_table].batch_size:
            pred = pred[: batch[self.task.entity_table].batch_size]

        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target = batch[self.task.entity_table].y.long()
        else:
            target = batch[self.task.entity_table].y.float()
        return pred, target

    def training_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = self.loss_fn(pred.float(), target)
        batch_size = pred.size(0)

        self.train_loss.update(loss.detach(), batch_size)

        self.log(
            "train_loss",
            self.train_loss.compute(),
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log_dict({"train_loss_epoch": train_loss}, prog_bar=True, logger=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int):
        pred, target = self(batch)

        pred = pred.detach().cpu()
        target = target.detach().cpu()

        mode = "val" if dataloader_idx == 0 else "test"
        metrics = self.val_metrics if mode == "val" else self.test_metrics

        for _, m in metrics.items():
            m.update(pred, target)

    def on_validation_epoch_end(self):
        val_metrics: dict[str, float] = {}

        tune_metric = self.val_metrics[self.tune_metric].compute()
        best_tune_metric = self.best_tune_metric.compute()
        self.best_tune_metric.update(tune_metric)

        for metrics, mode in [
            (self.val_metrics, "val"),
            (self.test_metrics, "test"),
        ]:
            for k, m in metrics.items():
                val_metrics[f"{mode}_{k}"] = m.compute()
                m.reset()

                if self.first_val:
                    val_metrics[f"first_{mode}_{k}"] = val_metrics[f"{mode}_{k}"]

                if (self.higher_is_better and tune_metric > best_tune_metric) or (
                    not self.higher_is_better and tune_metric < best_tune_metric
                ):
                    val_metrics[f"best_{mode}_{k}"] = val_metrics[f"{mode}_{k}"]

        if (self.higher_is_better and tune_metric > best_tune_metric) or (
            not self.higher_is_better and tune_metric < best_tune_metric
        ):
            val_metrics["best_step"] = self.trainer.global_step

        if self.first_val:
            val_metrics["first_step"] = self.trainer.global_step

        self.first_val = False

        self.log_dict(val_metrics, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.lr_scheduler_config is not None:
            return {
                "optimizer": self.optimizer,
                **self.lr_scheduler_config,
            }
        return self.optimizer


class SaveModelCallback(L.Callback):
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss_epoch",
        mode: str = "min",
        save_every_epoch: bool = False,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_every_epoch = save_every_epoch
        self.best_score = float("inf") if mode == "min" else float("-inf")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningEntityTaskWrapper
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

        if self.save_every_epoch:
            torch.save(
                pl_module.model.state_dict(),
                f"{self.save_dir}/epoch_{trainer.current_epoch}_{self.monitor}_{current_score:.3f}.pt",
            )

        if (self.mode == "min" and current_score < self.best_score) or (
            self.mode == "max" and current_score > self.best_score
        ):
            self.best_score = current_score
            torch.save(pl_module.model.state_dict(), f"{self.save_dir}/best_model.pt")
