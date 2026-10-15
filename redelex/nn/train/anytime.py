r"""Score a model at fixed optimiser steps while it trains, without disturbing it.

An anytime curve answers "how good is the model if the update stops after
``s`` steps?". Step 0 is the model as the update found it (the previous
model when warm-starting, a random one when training from scratch).

Evaluation must not change training. It runs in eval mode under
``torch.no_grad`` inside a forked torch RNG, with numpy and python RNG states
restored afterwards, so the training run draws exactly the same samples with
or without this callback. Its time goes to the cost meter's ``anytime``
bucket, never into training time. List the cost meter BEFORE this callback,
so that the meter has closed the batch when the evaluation starts.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Callable, Optional, Sequence

import lightning as L
import numpy as np
import torch

DEFAULT_SCHEDULE = (0, 10, 25, 50, 100, 250, 500, 1000, 2000)


def anytime_schedule(step: int, base: Sequence[int] = DEFAULT_SCHEDULE) -> bool:
    """``True`` at the base steps, then at every doubling of the last one."""
    if step in base:
        return True
    last = base[-1]
    if step <= last or last <= 0:
        return False
    ratio = step / last
    return ratio == int(ratio) and (int(ratio) & (int(ratio) - 1)) == 0


class AnytimeEvalCallback(L.Callback):
    r"""Calls ``evaluate(model) -> {name: value}`` at scheduled steps.

    Records ``{"step": s, **metrics}`` in :attr:`records` and logs each value as
    ``anytime_<name>`` at step ``s``. The final step of the fit is always
    recorded, flagged ``final``.
    """

    def __init__(
        self,
        evaluate: Callable[[torch.nn.Module], dict],
        base_schedule: Sequence[int] = DEFAULT_SCHEDULE,
        meter=None,
        eval_seed: int = 0,
        log_prefix: str = "anytime_",
    ):
        super().__init__()
        self.evaluate = evaluate
        self.base_schedule = tuple(base_schedule)
        self.meter = meter
        self.eval_seed = int(eval_seed)
        self.log_prefix = log_prefix
        self.records: list[dict] = []
        self._done: set[int] = set()

    def _model(self, pl_module):
        return getattr(pl_module, "model", pl_module)

    def _run(self, trainer: L.Trainer, pl_module: L.LightningModule, step: int, final: bool = False):
        if step in self._done:
            if final:
                for r in self.records:
                    if r["step"] == step:
                        r["final"] = True
            return
        self._done.add(step)
        model = self._model(pl_module)
        timer = self.meter.measure("anytime") if self.meter is not None else nullcontext()
        with timer:
            was_training = model.training
            np_state, py_state = np.random.get_state(), random.getstate()
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.eval_seed)
                np.random.seed(self.eval_seed)
                random.seed(self.eval_seed)
                model.eval()
                with torch.no_grad():
                    metrics = dict(self.evaluate(model))
            np.random.set_state(np_state)
            random.setstate(py_state)
            model.train(was_training)

        record = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        if final:
            record["final"] = True
        self.records.append(record)
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {f"{self.log_prefix}{k}": float(v) for k, v in metrics.items()}, step=int(step)
            )

    def on_train_start(self, trainer, pl_module):
        self._run(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if anytime_schedule(step, self.base_schedule):
            self._run(trainer, pl_module, step)

    def on_fit_end(self, trainer, pl_module):
        self._run(trainer, pl_module, int(trainer.global_step), final=True)


__all__ = ["AnytimeEvalCallback", "DEFAULT_SCHEDULE", "anytime_schedule"]
