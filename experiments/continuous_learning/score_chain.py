r"""Score task rows with an updated model, inline after each update.

Replaces the old separate prediction pass, which re-scored every checkpoint on
the whole timeline (cost growing with episodes squared) and rewrote one wide
CSV after every checkpoint. Here each update writes one small
``preds.parquet`` (``__row__``, ``pred``) for exactly the rows the analysis
needs, and the chain writes one ``rows.parquet`` that says what every row is:
entity, anchor time, target, window index and held-out flag.

Predictions are raw model outputs: logits for binary tasks, values for
regression. AUROC is invariant to the logistic link, so nothing is lost.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from relbench.base import EntityTask, Table, TaskType
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from redelex.continual.anchors import HOLDOUT_COL, ROW_COL
from redelex.continual.protocol import EpisodePlan

from experiments.continuous_learning.utils import get_table_input


def metric_name(task_type: TaskType) -> str:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc"
    if task_type == TaskType.REGRESSION:
        return "mae"
    raise ValueError(f"unsupported task type {task_type}")


def metric_value(task_type: TaskType, target, pred) -> float:
    """AUROC for binary tasks (nan with one class present), MAE for regression."""
    target, pred = np.asarray(target), np.asarray(pred)
    if len(target) == 0:
        return float("nan")
    if task_type == TaskType.BINARY_CLASSIFICATION:
        if len(np.unique(target)) < 2:
            return float("nan")
        return float(roc_auc_score(target, pred))
    if task_type == TaskType.REGRESSION:
        return float(mean_absolute_error(target, pred))
    raise ValueError(f"unsupported task type {task_type}")


def make_eval_loader(
    data: HeteroData,
    task: EntityTask,
    table: Table,
    num_neighbors: list[int],
    batch_size: int,
) -> NeighborLoader:
    inp = get_table_input(table, task)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=inp.nodes,
        input_time=inp.time,
        transform=inp.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=False,
    )


@torch.no_grad()
def predict_table(
    model: torch.nn.Module,
    data: HeteroData,
    task: EntityTask,
    table: Table,
    num_neighbors: list[int],
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> np.ndarray:
    """Raw outputs for every row of ``table``, in table order.

    Neighbour sampling is seeded inside a forked RNG, so the same model scores
    the same rows identically and the caller's random state is untouched.
    """
    if len(table.df) == 0:
        return np.zeros(0, dtype=np.float32)
    loader = make_eval_loader(data, task, table, num_neighbors, batch_size)
    was_training = model.training
    model.eval()
    out = []
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        for batch in loader:
            batch = batch.to(device)
            n = batch[task.entity_table].batch_size
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            out.append(pred[:n].float().cpu())
    np.random.set_state(np_state)
    random.setstate(py_state)
    model.train(was_training)
    return torch.cat(out).numpy()


def rows_frame(full_table: Table, task: EntityTask, plan: EpisodePlan) -> pd.DataFrame:
    """One row per task row: id, entity, anchor time, target, window, held-out."""
    df = full_table.df
    return pd.DataFrame(
        {
            ROW_COL: df[ROW_COL].to_numpy(),
            "entity": df[task.entity_col].to_numpy(),
            "time": pd.to_datetime(df[task.time_col]).to_numpy(),
            "target": df[task.target_col].to_numpy(),
            "window": plan.assign_windows(df[task.time_col]),
            "holdout": df[HOLDOUT_COL].to_numpy(),
        }
    )


def write_rows_table(path: str | Path, rows: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    rows.to_parquet(tmp, index=False)
    tmp.replace(path)


def needed_rows(rows: pd.DataFrame, episode_index: int, stale_windows: Optional[int]) -> np.ndarray:
    """Rows an update's model must be scored on.

    * every row of the deploy window ``e + 1`` and of the next ``stale_windows``
      windows (all of them when ``stale_windows`` is None): next-window score and
      staleness;
    * held-out rows of windows ``1 .. e``: forgetting (BWT) on rows never trained on.
    """
    window = rows["window"].to_numpy()
    deploy = episode_index + 1
    hi = np.inf if stale_windows is None else deploy + stale_windows
    future = (window >= deploy) & (window <= hi)
    past_holdout = (window >= 1) & (window <= episode_index) & rows["holdout"].to_numpy()
    return future | past_holdout


__all__ = [
    "make_eval_loader",
    "metric_name",
    "metric_value",
    "needed_rows",
    "predict_table",
    "rows_frame",
    "write_rows_table",
]
