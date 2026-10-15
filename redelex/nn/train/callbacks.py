import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Sequence

import lightning as L
import torch

from .entity_wrapper import LightningEntityTaskWrapper


class SaveModelCallback(L.Callback):
    r"""Saves the wrapped model's state dict on tune-metric improvement.

    Args:
        save_dir: Directory to save checkpoints to.
        monitor: The logged metric to monitor. If None, ``val_<tune_metric>``
            of the :class:`LightningEntityTaskWrapper` is used together with
            its improvement direction.
        mode: "min" or "max". Only used with an explicit ``monitor``
            (defaults to "min").
        save_every_epoch: If True, additionally save a checkpoint every epoch.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_every_epoch: bool = False,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_every_epoch = save_every_epoch
        self.best_score: Optional[float] = None
        self._warned_missing_monitor = False
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _resolve_monitor(self, pl_module: LightningEntityTaskWrapper) -> tuple[str, str]:
        if self.monitor is not None:
            return self.monitor, self.mode
        monitor = f"val_{pl_module.tune_metric}"
        mode = "max" if pl_module.higher_is_better else "min"
        return monitor, mode

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningEntityTaskWrapper
    ):
        if trainer.sanity_checking:
            return

        monitor, mode = self._resolve_monitor(pl_module)

        current_score = trainer.callback_metrics.get(monitor)
        if current_score is None:
            if not self._warned_missing_monitor:
                warnings.warn(
                    f"SaveModelCallback: monitored metric '{monitor}' has not "
                    "been logged; no checkpoint is saved.",
                    stacklevel=2,
                )
                self._warned_missing_monitor = True
            return

        current_score = (
            current_score.item()
            if isinstance(current_score, torch.Tensor)
            else current_score
        )

        if self.save_every_epoch:
            torch.save(
                pl_module.model.state_dict(),
                f"{self.save_dir}/epoch_{trainer.current_epoch}_{monitor}_{current_score:.3f}.pt",
            )

        if (
            self.best_score is None
            or (mode == "min" and current_score < self.best_score)
            or (mode == "max" and current_score > self.best_score)
        ):
            self.best_score = current_score
            torch.save(pl_module.model.state_dict(), f"{self.save_dir}/best_model.pt")


def _state_dict(pl_module: L.LightningModule) -> dict:
    model = getattr(pl_module, "model", pl_module)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _scalar(value) -> float:
    return value.item() if isinstance(value, torch.Tensor) else float(value)


class ConvergenceCallback(L.Callback):
    r"""Stops training once a monitored validation value stops improving.

    The value is read after every validation (schedule it with the Trainer's
    ``val_check_interval``). Several patience values are tracked in ONE run:
    under a constant learning rate the trajectory does not depend on when
    training would have stopped, so the run that waits for the largest
    patience also contains the exact outcome of every smaller one.

    For each ``P`` the callback records the step at which ``P`` consecutive
    checks without improvement were reached, the best step before it, and a
    copy of the best weights as ``best_P{P}.pt``. ``best_model.pt`` is the copy
    for ``primary`` (default: the smallest patience), which is what a chain
    continues from. Training stops when the largest patience fires; a run that
    reaches the Trainer's ``max_steps`` first is flagged ``hit_cap``.
    """

    def __init__(
        self,
        save_dir: str,
        patiences: Sequence[int] = (5,),
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        primary: Optional[int] = None,
        also_record: Sequence[str] = (),
    ):
        super().__init__()
        if not patiences or any(p < 1 for p in patiences):
            raise ValueError(f"patiences must be positive integers, got {patiences}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patiences = tuple(sorted(set(int(p) for p in patiences)))
        self.primary = int(primary) if primary is not None else self.patiences[0]
        if self.primary not in self.patiences:
            raise ValueError(f"primary patience {self.primary} is not among {self.patiences}")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)

        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None
        self.checks = 0
        self.since_best = 0
        self.history: list[tuple[int, float]] = []
        self.also_record = tuple(also_record)
        self.extra_history: dict[str, list[tuple[int, float]]] = {k: [] for k in self.also_record}
        self.stopped: dict[int, dict] = {}

    @property
    def _best_path(self) -> Path:
        return self.save_dir / "best_running.pt"

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def _freeze(self, patience: int, step: int, hit_cap: bool) -> None:
        target = self.save_dir / f"best_P{patience}.pt"
        if self._best_path.exists():
            shutil.copyfile(self._best_path, target)
        self.stopped[patience] = {
            "stop_step": int(step),
            "best_step": None if self.best_step is None else int(self.best_step),
            "best_value": self.best_value,
            "checks": self.checks,
            "hit_cap": bool(hit_cap),
        }
        if patience == self.primary and target.exists():
            shutil.copyfile(target, self.save_dir / "best_model.pt")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            raise KeyError(
                f"ConvergenceCallback: '{self.monitor}' was not logged by the validation loop"
            )
        value = _scalar(value)
        step = int(trainer.global_step)
        self.checks += 1
        self.history.append((step, value))
        for key in self.also_record:
            other = trainer.callback_metrics.get(key)
            if other is not None:
                self.extra_history[key].append((step, _scalar(other)))

        if self._improved(value):
            self.best_value, self.best_step, self.since_best = value, step, 0
            torch.save(_state_dict(pl_module), self._best_path)
        else:
            self.since_best += 1

        for p in self.patiences:
            if p not in self.stopped and self.since_best >= p:
                self._freeze(p, step, hit_cap=False)
        if len(self.stopped) == len(self.patiences):
            trainer.should_stop = True

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Any patience still waiting ran into the step cap (or was interrupted).
        for p in self.patiences:
            if p not in self.stopped:
                self._freeze(p, int(trainer.global_step), hit_cap=True)

    def summary(self) -> dict:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "primary": self.primary,
            "checks": self.checks,
            "patience": {str(p): self.stopped.get(p) for p in self.patiences},
            # Every check as (step, value): enough to replay any other stopping
            # rule offline (calibration of N and P).
            "history": [[int(s), float(v)] for s, v in self.history],
            "history_extra": {k: [[int(s), float(v)] for s, v in h] for k, h in self.extra_history.items()},
        }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CostMeterCallback(L.Callback):
    r"""Splits a fit's wall-clock into training, data, validation and other work.

    Buckets, in seconds (every boundary is CUDA-synchronised):

    * ``train_compute``: forward, backward and optimiser step of each batch;
    * ``train_data``: the gap between two batches, i.e. neighbour sampling and
      host-to-device copies, with any validation or measured work removed;
    * ``val``: validation loops;
    * anything timed through :meth:`measure` (e.g. ``anytime``).

    ``train_compute + train_data`` is the reported GPU time of an update,
    validation excluded. It also counts optimiser steps and seed samples, and,
    given ``source_counts`` (``MixtureLoader.source_counts``), samples per source.
    """

    def __init__(
        self,
        entity_table: Optional[str] = None,
        source_counts: Optional[Callable] = None,
    ):
        super().__init__()
        self.entity_table = entity_table
        self.source_counts = source_counts
        self.seconds: dict[str, float] = {"train_compute": 0.0, "train_data": 0.0, "val": 0.0}
        self.samples = 0
        self.samples_by_source: dict[str, int] = {}
        self.steps = 0
        self._gap_start: Optional[float] = None
        self._excluded = 0.0
        self._batch_start: Optional[float] = None
        self._val_start: Optional[float] = None
        self._fit_start: Optional[float] = None
        self._fit_ended = False
        self.wall = 0.0
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )

    def add(self, bucket: str, seconds: float) -> None:
        self.seconds[bucket] = self.seconds.get(bucket, 0.0) + float(seconds)

    @contextmanager
    def measure(self, bucket: str):
        """Time a block into ``bucket`` and keep it out of ``train_data``."""
        _sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            now = time.perf_counter()
            elapsed = now - start
            self.add(bucket, elapsed)
            self._excluded += elapsed
            if self._fit_ended and self._fit_start is not None:
                # Work measured after fit end (a final evaluation) still belongs
                # to this update: stretch the wall clock over it.
                self.wall = now - self._fit_start

    def on_fit_start(self, trainer, pl_module):
        _sync()
        self._fit_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _sync()
        now = time.perf_counter()
        if self._gap_start is not None:
            self.add("train_data", max(0.0, now - self._gap_start - self._excluded))
        self._excluded = 0.0
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _sync()
        now = time.perf_counter()
        if self._batch_start is not None:
            self.add("train_compute", now - self._batch_start)
        self._batch_start = None
        self._gap_start = now
        self._excluded = 0.0
        self.steps = int(trainer.global_step)
        if self.entity_table is not None:
            self.samples += int(batch[self.entity_table].batch_size)
            if self.source_counts is not None:
                for name, n in self.source_counts(batch).items():
                    self.samples_by_source[name] = self.samples_by_source.get(name, 0) + n

    def on_validation_start(self, trainer, pl_module):
        _sync()
        self._val_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        _sync()
        if self._val_start is not None:
            elapsed = time.perf_counter() - self._val_start
            self.add("val", elapsed)
            self._excluded += elapsed
        self._val_start = None

    def on_fit_end(self, trainer, pl_module):
        _sync()
        if self._fit_start is not None:
            self.wall = time.perf_counter() - self._fit_start
        self._fit_ended = True
        self.steps = int(trainer.global_step)

    def summary(self) -> dict:
        train = self.seconds["train_compute"] + self.seconds["train_data"]
        return {
            "device": self.device_name,
            "steps": self.steps,
            "samples": self.samples,
            "samples_by_source": dict(self.samples_by_source),
            "seconds": dict(self.seconds),
            "train_seconds": train,
            "fit_wall_seconds": self.wall,
        }


__all__ = ["ConvergenceCallback", "CostMeterCallback", "SaveModelCallback"]
