from typing import Optional
import copy

import torch

import lightning as L

from torchmetrics.aggregation import MaxMetric, MinMetric, MeanMetric

from relbench.base import TaskType, EntityTask

from .utils import get_loss, get_metrics


class LightningEntityTaskWrapper(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        task: EntityTask,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
        self.scheduler = scheduler

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
            "train_loss", self.train_loss.compute(), prog_bar=True, batch_size=batch_size
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

        for mname, m in metrics.items():
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
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return self.optimizer
