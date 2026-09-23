import copy
from typing import Optional, Union

import lightning as L
import torch
from relbench.base import EntityTask, TaskType
from torchmetrics.aggregation import MaxMetric, MeanMetric, MinMetric

from redelex.tasks.mixins import EntityTaskMixin

from .utils import get_loss, get_metrics


class LightningEntityTaskWrapper(L.LightningModule):
    r"""Trains a model on an entity task, logging first/best metrics.

    The metrics follow from ``task.task_type``; for multiclass classification the
    number of classes is taken from ``task.num_classes``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        task: Union[EntityTaskMixin, EntityTask],
        lr_scheduler_config: Optional[dict] = None,
        modes: tuple[str, ...] = ("val", "test"),
    ):
        super().__init__()

        self.model = model
        self.task = task
        self.loss_fn = get_loss(self.task.task_type)
        self.val_metrics, self.tune_metric, self.higher_is_better = get_metrics(
            self.task.task_type, num_classes=getattr(task, "num_classes", None)
        )
        self.modes = modes
        self.test_metrics = (
            copy.deepcopy(self.val_metrics) if "test" in self.modes else None
        )
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
    def _active_metrics_with_mode(self):
        pairs = [(self.val_metrics, "val"), (self.test_metrics, "test")]
        return [(m, mode) for m, mode in pairs if m is not None]

    def _active_metrics(self):
        return [m for m, _ in self._active_metrics_with_mode()]

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pred, target = self(batch)

        pred = pred.detach().cpu()
        target = target.detach().cpu()

        mode = "val" if dataloader_idx == 0 else "test"
        metrics = self.val_metrics if mode == "val" else self.test_metrics
        if metrics is None:
            return

        for _, m in metrics.items():
            m.update(pred, target)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            # Discard metric updates from the sanity-check batches so they do
            # not pollute first/best metrics of the real epochs.
            for metrics in self._active_metrics():
                for m in metrics.values():
                    m.reset()
            return

        val_metrics: dict[str, float] = {}

        tune_metric = self.val_metrics[self.tune_metric].compute()
        best_tune_metric = self.best_tune_metric.compute()
        self.best_tune_metric.update(tune_metric)

        for metrics, mode in self._active_metrics_with_mode():
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

        self._step_plateau_scheduler(tune_metric)

    def _step_plateau_scheduler(self, tune_metric) -> None:
        """Advance a ReduceLROnPlateau once per validation.

        Stepping from the hook that produced the metric makes ``patience`` mean
        "validations" for every mode and every dataset.
        """
        if self.scheduler is None:
            return
        value = tune_metric.item() if hasattr(tune_metric, "item") else float(tune_metric)
        self.scheduler.step(value)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=False, logger=True)

    def configure_optimizers(self):
        # The scheduler is deliberately NOT handed to Lightning: it is stepped in
        # `_step_plateau_scheduler` instead, once per validation. See there.
        return self.optimizer


__all__ = ["LightningEntityTaskWrapper"]
