from typing import Any, Literal, Optional

import copy
from pathlib import Path

import sys
import traceback
import os
import random
from datetime import datetime, timedelta
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import pandas as pd

import numpy as np

import torch

import ray
from ray import tune, train as ray_train

import lightning as L
from lightning.pytorch import loggers, callbacks
from lightning.pytorch.utilities.model_summary import ModelSummary

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T

from relbench.base import Table, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

import redelex.tasks.mixins as task_mixin
from redelex.data import make_pkey_fkey_graph
from redelex.loaders import ComposedLoader
from redelex.nn.train import (
    LightningEntityTaskWrapper,
    PhaseTimerCallback,
    SaveModelCallback,
)
from redelex.nn.train.utils import get_metrics

from experiments.continuous_learning.continuous_task import ContinuousWrapper

from experiments.continuous_learning.cl_modes import (
    DEFAULT_ROSTER,
    MODES,
    MODE_ALIASES,
    AttachAuxTransform,
    CLState,
    buffer_to_table,
    make_der_penalty,
    make_ewc_penalty,
    make_lwf_penalty,
    resolve_mode,
)

from redelex.continual.adapters import (
    AdapterStack,
    freeze_module,
    parameter_counts,
)
from redelex.continual import (
    ParameterAnchor,
    ReservoirBuffer,
    fisher_diagonal,
    frozen_teacher,
)

from experiments.continuous_learning.models import HeterogeneousSAGE

from redelex.utils.datetime import to_unix_time

from experiments.continuous_learning.utils import (
    get_attribute_schema,
    get_hyperparams_logging,
    get_text_embedder,
    get_table_input,
    get_potato_client,
    get_experiment_runs_df,
    subsample_val_table,
)

def get_resume_state_from_mlflow(
    mlflow_experiment: str,
    dataset_name: str,
    task_name: str,
    val_metric: str,
    higher_is_better: bool,
    mlflow_uri: Optional[str] = None,
    expected_trials: int = 5,
    chain_id: Optional[str] = None,
) -> tuple[int, Optional[str], Optional[str]]:
    """Last fully completed increment, and the weights and CL state to continue from.

    Returns ``(next_increment, weights_path, cl_state_path)``. The CL state is
    returned alongside the weights and from the SAME trial: a chain that resumes
    with weights but without its buffer or anchor silently runs one episode with
    the method disabled, and permanently loses the pre-resume history that made
    the reservoir uniform over the stream.

    Args:
        expected_trials: How many trials an increment must have to count as
            complete. Must be the number of seeds actually run -- a hardcoded
            value larger than that makes every increment look incomplete and
            resume silently restarts from episode 1.
        chain_id: When given, only runs from this chain are considered. Without
            it, unrelated runs in the same MLflow experiment -- including the
            published grid, whose checkpoints live on a machine this one cannot
            reach -- can satisfy the quorum.
    """
    try:
        mlflow_client = get_potato_client(mlflow_uri)
        filters = [
            f"params.dataset_name = '{dataset_name}'",
            f"params.task_name = '{task_name}'",
            "attributes.status = 'FINISHED'",
        ]
        if chain_id:
            filters.append(f"params.chain_id = '{chain_id}'")
        runs = get_experiment_runs_df(
            mlflow_client, mlflow_experiment, filter_string=" and ".join(filters)
        )
        if runs.empty:
            print("No finished runs found; starting from the first increment.")
            return 1, None, None

        best_metric_col = f"best_val_{val_metric}"
        runs["increment"] = runs["increment"].astype(int)
        runs[best_metric_col] = pd.to_numeric(runs[best_metric_col], errors="coerce")

        # Walk back to the newest increment that is genuinely complete. A partially
        # finished increment must not be resumed past -- its best trial is a
        # sample of the seeds, not the best of them.
        max_inc, inc_runs = runs["increment"].max(), None
        while max_inc >= 1:
            candidate = runs[runs["increment"] == max_inc]
            if len(candidate) >= expected_trials:
                inc_runs = candidate
                break
            if len(candidate) > 0:
                print(
                    f"Increment {max_inc} has {len(candidate)}/{expected_trials} "
                    f"trials; treating it as incomplete and looking further back."
                )
            max_inc -= 1

        if inc_runs is None or inc_runs.empty:
            print("No complete increment found; starting from the first increment.")
            return 1, None, None

        idx = (
            inc_runs[best_metric_col].idxmax()
            if higher_is_better
            else inc_runs[best_metric_col].idxmin()
        )
        best_run = inc_runs.loc[idx]

        if "model_save_dir" not in best_run or pd.isna(best_run["model_save_dir"]):
            raise ValueError(f"best run for increment {max_inc} has no model_save_dir")

        save_dir = Path(best_run["model_save_dir"])
        weights_path = save_dir / "best_model.pt"
        if not weights_path.exists():
            # The published grid records paths on a cluster this box cannot see.
            raise FileNotFoundError(
                f"checkpoint for increment {max_inc} is not on this filesystem: "
                f"{weights_path}"
            )

        state_path = save_dir / CLState.FILENAME
        print(
            f"Resuming after increment {max_inc} "
            f"({len(inc_runs)} trials), CL state "
            f"{'found' if state_path.exists() else 'ABSENT'}."
        )
        return (
            max_inc + 1,
            str(weights_path),
            str(state_path) if state_path.exists() else None,
        )

    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"Could not determine resume state ({type(e).__name__}: {e}); "
              f"starting from the first increment.")

    return 1, None, None


def _update_chain_state(
    *,
    cl_state,
    chain_spec,
    config,
    model,
    lightning_model,
    train_loader,
    wrapped_task,
    task,
    data,
    train_start,
    train_timestamp,
    device,
):
    """Refresh the state a continual-learning chain carries into the next episode.

    Weights already travel via ``best_model.pt``. Everything else a method needs --
    the replay buffer, the EWC anchor -- has to be updated here and saved beside
    them, or the method silently restarts every episode.

    Driven by the CHAIN's mode rather than this episode's, because episode 1 runs
    as ``from_scratch`` for every method and would otherwise leave the next episode
    with nothing to replay or anchor to.
    """
    # Lightning leaves the model on CPU after `fit`, so everything below would run
    # there -- a full inference pass over a 105k-row increment on rel-hm. Put it
    # back on the training device first; the helpers then read the device off the
    # model itself, so batches and parameters cannot disagree.
    model.to(device)

    # Measure the chain state at the weights the chain actually continues from.
    # After `fit` the model holds the FINAL optimiser step, but the next episode
    # warm-starts from `best_model.pt`, so an anchor or a stored logit taken here
    # describes a point the chain never occupies. On the smoke artefacts the two
    # differ by ||theta_final - theta_best|| = 22.7.
    best_checkpoint = Path(config["model_save_dir"]) / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        model.to(device)
    else:
        print(
            f"WARNING: {best_checkpoint} missing; chain state will be measured at "
            f"the final weights rather than the selected ones.",
            flush=True,
        )

    if chain_spec.uses_buffer:
        buffer = cl_state.buffer
        if buffer is None:
            buffer = ReservoirBuffer(
                capacity=int(config.get("buffer_size", 10_000)),
                seed=int(config["seed"]),
            )

        # Offer this episode's increment to the buffer. Reservoir sampling keeps a
        # uniform sample of the whole stream, so early episodes stay represented as
        # history grows -- which is exactly what proportional mixing did not do.
        increment = wrapped_task.get_table(start=train_start, end=train_timestamp)
        df = increment.df
        if len(df) > 0:
            node_ids = df[task.entity_col].astype("int64").to_numpy()
            timestamps = to_unix_time(df[increment.time_col])
            targets = df[task.target_col].to_numpy(dtype="float64")

            logits = None
            if chain_spec.distil_stored_logits:
                logits = _predict_logits(
                    model=model,
                    table=increment,
                    task=task,
                    data=data,
                    config=config,
                    device=device,
                )

            buffer.add(
                node_ids, timestamps=timestamps, targets=targets, logits=logits
            )
        cl_state.buffer = buffer
        config["replay_buffer_size"] = len(buffer)
        config["replay_buffer_seen"] = buffer.seen
        print(
            f"Replay buffer: {len(buffer)}/{buffer.capacity} held, {buffer.seen} seen",
            flush=True,
        )

    if chain_spec.freeze_backbone and getattr(model, "adapters", None) is not None:
        cl_state.adapters = model.adapters.state_dict()

    if chain_spec.uses_anchor:
        anchor = cl_state.anchor
        if anchor is None:
            anchor = ParameterAnchor(
                lam=float(config.get("ewc_lambda", 100.0)),
                gamma=float(config.get("ewc_gamma", 0.9)),
            )
        fisher = fisher_diagonal(
            model,
            batches=_batches_on(train_loader, next(model.parameters()).device),
            loss_fn=lightning_model.loss_fn,
            forward_fn=lambda m, b: lightning_model(b)[0].float(),
            target_fn=lambda b: lightning_model(b)[1],
            max_batches=int(config.get("fisher_batches", 64)),
        )
        anchor.consolidate(model, fisher)
        cl_state.anchor = anchor
        print(
            f"EWC anchor consolidated over {len(anchor)} tensors "
            f"({anchor.episodes} episode(s))",
            flush=True,
        )


def _batches_on(loader, device):
    """Yield batches on `device`.

    Lightning moves batches during `fit`, but anything run afterwards -- the Fisher
    pass, the DER++ logit capture -- gets raw loader output and has to move them
    itself, or the forward hits "at least two devices, cuda:0 and cpu".
    """
    for batch in loader:
        yield batch.to(device)


@torch.no_grad()
def _predict_logits(*, model, table, task, data, config, device):
    """Model outputs for every row of a table, for DER++ to store alongside it.

    DER++ distils against the logit the model produced *when the exemplar was
    stored*, so these must be captured at the end of the episode that saw them.
    """
    was_training = model.training
    # Take the device from the model rather than the caller: after `trainer.fit`
    # the model is wherever Lightning left it, which is not necessarily `device`.
    device = next(model.parameters()).device
    model.eval()
    try:
        table_input = get_table_input(table, task)
        gnn_layers = config["gnn_layers"]
        num_neighbors = config["num_neighbors"]
        loader = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=config["batch_size"],
            temporal_strategy="uniform",
            shuffle=False,
        )
        out = []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(-1) == 1 else pred
            out.append(pred[: batch[task.entity_table].batch_size].detach().cpu())
        return torch.cat(out).numpy().astype("float64") if out else None
    finally:
        if was_training:
            model.train()


def run_continuous_learning_experiment(
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

    # One thread per trial, ALWAYS -- not only when a GPU is present. This used to
    # sit inside the `torch.cuda.is_available()` branch, so a CPU run silently got
    # torch's default of one thread per core (64 here). That is the worst setting
    # available: measured on this box, 8 threads buy only 2.2x on a training step
    # (per-core efficiency 0.27) while 32 single-threaded processes hold 0.74-0.80
    # efficiency. Oversubscribing threads across concurrent trials is strictly
    # worse than running more trials.
    torch.set_num_threads(int(config.get("torch_threads", 1)))

    if with_ray:
        context = ray_train.get_context()
        trial_name = context.get_trial_name()
        resources = context.get_trial_resources().required_resources
        print(f"Resources: {resources}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        allow_gpu = config.get("allow_gpu", False)
        if allow_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        trial_name = f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Device: {device} | torch threads: {torch.get_num_threads()}")

    dataset_name: str = config["dataset_name"]
    task_name: str = config["task_name"]
    cache_path: str = config["cache_path"]

    model_save_dir = Path(config["model_save_dir"])
    model_save_dir = f"{model_save_dir}/{trial_name}"
    config["model_save_dir"] = model_save_dir

    gnn_channels = config["gnn_channels"]
    gnn_layers = config["gnn_layers"]
    gnn_aggr = config["gnn_aggr"]
    num_neighbors = config["num_neighbors"]

    batch_size = config["batch_size"]
    lr = config["lr"]
    max_training_steps: int = config["max_training_steps"]

    learning_mode: str = config["learning_mode"]
    weights_path: Optional[str] = config.get("weights_path", None)
    train_timestamp = config["train_timestamp"]
    val_timestamp = config["val_timestamp"]
    prev_train_timestamp: Optional[pd.Timestamp] = config.get("prev_train_timestamp", None)

    # `learning_mode` is what THIS episode runs (episode 1 is always from_scratch);
    # `chain_learning_mode` is what the chain is, and it decides which state has to
    # be carried forward -- a replay buffer must be filled during episode 1 even
    # though episode 1 itself trains from scratch, or episode 2 starts empty.
    chain_learning_mode: str = config.get("chain_learning_mode", learning_mode)
    spec = resolve_mode(learning_mode)
    chain_spec = resolve_mode(chain_learning_mode)
    cl_state_path: Optional[str] = config.get("cl_state_path", None)

    if spec.warm_start:
        assert (
            weights_path is not None
        ), f"weights_path must be provided for mode {spec.name}"

    if spec.needs_prev_timestamp:
        assert (
            prev_train_timestamp is not None
        ), f"prev_train_timestamp must be provided for mode {spec.name}"

    cl_state = CLState()
    if cl_state_path is not None and Path(cl_state_path).exists():
        cl_state = CLState.load(cl_state_path)
        print(f"Loaded CL state from {cl_state_path}", flush=True)

    dataset = get_dataset(dataset_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)

    text_embedder = get_text_embedder(
        config["text_embedder_name"], device=torch.device("cpu")
    )
    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
    )

    # The head width is fixed by the task, not a free choice. Every task in the
    # current grid is binary or regression, so this is 1 throughout -- but the
    # model defaulted to 1 unconditionally, which would have silently produced a
    # single logit for a multiclass task.
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = task.num_classes
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = len(task.stats()["num_labels"]) if hasattr(task, "stats") else None
        assert out_channels, "multilabel tasks need an explicit label count"
    else:
        out_channels = 1
    config["out_channels"] = out_channels

    # create model
    model = HeterogeneousSAGE(
        data=data,
        col_stats_dict=col_stats_dict,
        gnn_channels=gnn_channels,
        gnn_layers=gnn_layers,
        gnn_aggr=gnn_aggr,
        out_channels=out_channels,
        # `head_norm` was in param_space and logged to MLflow, but was never
        # passed here -- the model always used its own default. Same value, but
        # the logged "hyperparameter" was inert.
        norm=config["head_norm"],
    )

    # Parameter isolation: rebuild the adapter stack this chain has accumulated, so
    # the checkpoint's adapter tensors have somewhere to land.
    adapter_stack = None
    if chain_spec.freeze_backbone:
        adapter_stack = AdapterStack(
            channels=gnn_channels, rank=int(config.get("adapter_rank", 16))
        )
        if cl_state.adapters is not None:
            adapter_stack.load_state_dict(cl_state.adapters)
        model.adapters = adapter_stack

    # optionally load weights from previous split
    if weights_path is not None:
        # strict=False for freeze_extend only: the adapter stack grows by one module
        # per episode, so a checkpoint never has exactly the keys of the model that
        # is about to extend it.
        model.load_state_dict(
            torch.load(weights_path), strict=not chain_spec.freeze_backbone
        )

    if spec.freeze_backbone:
        # Freeze everything learned so far and train only newly added capacity.
        # This is what makes the mode forget nothing by construction.
        for module in (
            model.row_encoder,
            model.temporal_encoder,
            model.gnn,
            model.head,
        ):
            # freeze_module returns a count of TENSORS; every number reported here
            # is a count of ELEMENTS, so its return value is deliberately discarded
            # rather than logged next to element counts as if comparable.
            freeze_module(module)

        new_adapter = adapter_stack.add_adapter()
        counts = parameter_counts(model)
        added = sum(p.numel() for p in new_adapter.parameters())
        config["frozen_parameters"] = counts["frozen"]
        config["trainable_parameters"] = counts["trainable"]
        config["adapter_parameters"] = added
        config["n_adapters"] = adapter_stack.n_adapters
        print(
            f"freeze_extend: adapter #{adapter_stack.n_adapters} adds {added:,} "
            f"parameters; {counts['trainable']:,}/{counts['total']:,} trainable, "
            f"{counts['frozen']:,} frozen",
            flush=True,
        )

    train_start = (
        db.min_timestamp if spec.train_window == "full" else prev_train_timestamp
    )
    config["train_start"] = train_start

    # create train dataloader for current split
    train_table = wrapped_task.get_table(start=train_start, end=train_timestamp)
    train_input = get_table_input(train_table, task)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
        time_attr="time",
        input_nodes=train_input.nodes,
        input_time=train_input.time,
        transform=train_input.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=True,
    )
    def _make_loader(table_input, transform=None):
        return NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=transform if transform is not None else table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=True,
        )

    replay_ratio: float = config.get("replay_ratio", 0.5)

    if spec.legacy and spec.name == "ft_upsample":
        # Reproduction path only. `rnd_uni` mixes proportionally to loader size, so
        # the new-data share decays from ~50% to a few percent as history grows --
        # see analysis/upsampling-ratio-finding.md. Kept to reproduce the submitted
        # paper, never used by the roster.
        old_train_table = wrapped_task.get_table(
            start=db.min_timestamp, end=prev_train_timestamp
        )
        old_train_loader = _make_loader(get_table_input(old_train_table, task))
        train_loader = ComposedLoader(
            {"new": train_loader, "old": old_train_loader}, mode="rnd_uni"
        )

    elif spec.uses_buffer and cl_state.buffer is not None and len(cl_state.buffer) > 0:
        # Bounded replay: the buffer holds a uniform sample of everything seen so
        # far, capped at `buffer_size`. Unlike ft_upsample the ratio is explicit and
        # honoured, because `weighted` draws by configured proportion rather than by
        # loader size.
        replay_table, replay_order = buffer_to_table(
            cl_state.buffer,
            entity_col=task.entity_col,
            time_col=wrapped_task.full_table.time_col,
            target_col=task.target_col,
            template=wrapped_task.full_table,
        )
        replay_input = get_table_input(replay_table, task)

        replay_transform = replay_input.transform
        if spec.distil_stored_logits:
            # DER++ distils against the logit recorded when each exemplar was
            # stored, so the logit must follow its own row through the sort.
            # `replay_order` is the exact permutation the table was built with.
            ordered_logits = torch.as_tensor(
                cl_state.buffer.logits[replay_order], dtype=torch.float32
            )
            replay_transform = T.Compose(
                [
                    replay_input.transform,
                    AttachAuxTransform(
                        task.entity_table, "teacher_logit", ordered_logits
                    ),
                ]
            )

        replay_loader = _make_loader(replay_input, transform=replay_transform)
        train_loader = ComposedLoader(
            {"new": train_loader, "old": replay_loader},
            mode="weighted",
            weights={"new": replay_ratio, "old": 1.0 - replay_ratio},
        )
        config["replay_buffer_used"] = len(cl_state.buffer)

    # create val dataloader for current split
    val_table = wrapped_task.get_table(start=train_timestamp, end=val_timestamp)

    # Bound the cost of model selection. Validation runs `max_training_steps /
    # val_check_interval` times per run over the whole window, which on the large
    # datasets costs several times the training it is evaluating. Subsample the
    # *table* rather than using `limit_val_batches`: the loader is unshuffled, so
    # capping batches would keep only the temporally earliest rows.
    #
    # The seed depends on (dataset, task, increment) and deliberately NOT on the
    # trial seed, so every method and every seed selects against the identical
    # evaluation set. Reported metrics are unaffected either way -- they come from
    # run_predictions.py re-scoring checkpoints over the full timeline.
    val_max_rows: Optional[int] = config.get("val_max_rows", None)
    config["val_rows_full"] = int(len(val_table.df))
    val_table, subsample_seed = subsample_val_table(
        val_table, val_max_rows, dataset_name, task_name, config["increment"]
    )
    if subsample_seed is not None:
        config["val_subsample_seed"] = subsample_seed
    config["val_rows_used"] = int(len(val_table.df))

    val_input = get_table_input(val_table, task)
    val_loader = NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
        time_attr="time",
        input_nodes=val_input.nodes,
        input_time=val_input.time,
        transform=val_input.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=False,
    )

    # Only trainable parameters: freeze_extend leaves most of the model frozen, and
    # handing Adam frozen tensors would build optimiser state for parameters that
    # never move.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    val_check_interval: Optional[int] = config.get("val_check_interval", None)

    _, val_metric, higher_is_better = get_metrics(
        task.task_type, num_classes=getattr(task, "num_classes", None)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max" if higher_is_better else "min", factor=0.5, patience=3
    )
    
    lightning_model = LightningEntityTaskWrapper(
        model=model,
        task=task,
        optimizer=optimizer,
        lr_scheduler_config={
            "scheduler": scheduler,
            "monitor": f"val_{val_metric}",
            "mode": "max" if higher_is_better else "min",
            # Step on the same cadence as validation, so `patience` means the same
            # number of optimiser steps in every run. Under the old epoch-based
            # setting an "epoch" was min(limit_train_batches, len(loader)), so short
            # early episodes decayed the LR ~15x more aggressively per step.
            "interval": "step" if val_check_interval else "epoch",
            "frequency": val_check_interval or 1,
        },
        modes=("val",)
    )

    # --- continual-learning penalty terms -----------------------------------
    # Attached to the wrapper rather than passed to its constructor, so the
    # wrapper stays usable by every other experiment without knowing about CL.
    if spec.uses_anchor and cl_state.anchor is not None:
        lightning_model.cl_penalty = make_ewc_penalty(cl_state.anchor)
        print(
            f"EWC active: lambda={cl_state.anchor.lam}, "
            f"{len(cl_state.anchor)} anchored tensors from "
            f"{cl_state.anchor.episodes} episode(s)",
            flush=True,
        )
    elif spec.uses_teacher and weights_path is not None:
        # LwF distils from the previous episode's model on current data. The
        # teacher is a frozen copy loaded from the same checkpoint the student
        # warm-started from, so it is exactly "the model before this episode".
        teacher = frozen_teacher(model).to(device)
        lightning_model.cl_penalty = make_lwf_penalty(
            teacher, task.entity_table, weight=config.get("lwf_alpha", 1.0)
        )
        print("LwF active: distilling from the previous episode's model", flush=True)
    elif spec.distil_stored_logits:
        lightning_model.cl_penalty = make_der_penalty(
            task.entity_table, alpha=config.get("der_alpha", 0.5)
        )
        print("DER++ active: distilling against stored logits", flush=True)

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

    save_model_callback = SaveModelCallback(
        save_dir=model_save_dir,
        monitor=f"val_{val_metric}",
        mode="max" if higher_is_better else "min",
        save_every_epoch=False,
    )
    phase_timer = PhaseTimerCallback()
    trainer = L.Trainer(
        max_steps=max_training_steps,
        max_epochs=config.get("max_epochs", None),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        # An integer `val_check_interval` counts global steps and may exceed the
        # epoch length only when `check_val_every_n_epoch` is None (see Lightning
        # fit_loop.py). That combination is what makes the number of validation
        # passes identical across datasets and episode sizes.
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None if val_check_interval else 1,
        accelerator=device.type,
        devices=1,
        logger=logger,
        callbacks=[
            save_model_callback,
            phase_timer,
            # The LR was logged nowhere, which is why its episode-dependent
            # collapse under the old epoch-based schedule went unnoticed.
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        max_time=timedelta(hours=2),
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
    )
    try:
        trainer.fit(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=None,
        )

        # --- carry state to the next episode --------------------------------
        # Follows the CHAIN's mode, not this episode's: episode 1 trains from
        # scratch for every method, but an ER chain still has to fill its buffer
        # there or episode 2 replays nothing.
        _update_chain_state(
            cl_state=cl_state,
            chain_spec=chain_spec,
            config=config,
            model=model,
            lightning_model=lightning_model,
            train_loader=train_loader,
            wrapped_task=wrapped_task,
            task=task,
            data=data,
            train_start=train_start,
            train_timestamp=train_timestamp,
            device=device,
        )
        # A NaN parameter is silent: the forward nan_to_num's it away, the metric
        # still looks healthy, and the dead column is simply gone. Refuse to write
        # chain state built on one, so a poisoned Fisher or buffer cannot travel.
        nonfinite = [
            n for n, p in model.named_parameters() if not torch.isfinite(p).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"non-finite parameters after training: {nonfinite[:5]} "
                f"({len(nonfinite)} total) -- refusing to save chain state"
            )

        cl_state.save(Path(model_save_dir) / CLState.FILENAME)

        if with_ray:
            best_val_metric = trainer.callback_metrics.get(f"best_val_{val_metric}")
            if best_val_metric is not None:
                best_val_metric = best_val_metric.item()
                ray_train.report({f"val_{val_metric}": best_val_metric, "model_save_dir": str(model_save_dir)})


    except Exception as e:
        logger.log_hyperparams({"error": str(e)})
        stack_trace = traceback.format_exc()
        logger.log_hyperparams({"stack_trace": stack_trace})
        print(stack_trace)
        logger.finalize("failed")


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    learning_mode: str,
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "pelesjak_test_experiment",
    num_samples: Optional[int] = 1,
    num_gpus: int = 0,
    num_cpus: int = 1,
    gpu_ids: Optional[list[int]] = None,
    random_seed: int = 42,
    seeds: Optional[list[int]] = None,
    max_increments: Optional[int] = None,
    max_training_steps: int = 2000,
    limit_train_batches: Optional[int] = 100,
    max_epochs: Optional[int] = None,
    val_check_interval: Optional[int] = 100,
    val_max_rows: Optional[int] = 25_000,
    val_delta_days: Optional[float] = None,
    cpus_per_trial: int = 2,
    torch_threads: int = 1,
    buffer_size: int = 10_000,
    replay_ratio: float = 0.5,
    der_alpha: float = 0.5,
    ewc_lambda: float = 100.0,
    ewc_gamma: float = 0.9,
    lwf_alpha: float = 1.0,
    fisher_batches: int = 64,
    cache_dir: str = ".cache",
    model_save_dir: str = "./models",
    resume: bool = False,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Per-trial seeds are fixed up front rather than drawn lazily by
    # `tune.randint` inside the episode loop. The values are identical to the
    # legacy draw (seed 42 -> [102, 435, 860, 270, 106]), but pinning them here
    # means every episode uses the same seed set and `--resume` cannot shift the
    # sequence, which previously gave resumed runs different seeds.
    if seeds is None:
        seeds = [int(x) for x in np.random.randint(0, 1000, num_samples)]
    seeds = list(seeds)
    print(f"Per-trial seeds ({len(seeds)}): {seeds}", flush=True)

    # Keep the script runnable without an MLflow server configured: an explicit
    # None would otherwise reach MLflow as the literal experiment name "None".
    if not mlflow_experiment:
        mlflow_experiment = f"cl_{learning_mode}"

    if gpu_ids:
        # Explicit pinning, for an external scheduler that owns the allocation
        # (see scripts/run_grid.py). Skips the auto-selection below, which would
        # otherwise renumber CUDA_VISIBLE_DEVICES relative to the visible set and
        # land the job on the wrong physical device.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        num_gpus = min(num_gpus, len(gpu_ids)) or len(gpu_ids)
        print(f"Pinned to GPUs {os.environ['CUDA_VISIBLE_DEVICES']}")
    elif num_gpus > 0 and ray_address == "local":
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
    # CPU-only runs are process-parallel, one core each: threads scale badly here
    # (8 threads -> 2.2x) while processes scale well (32-way -> 0.74-0.80
    # efficiency), and there is a hard throughput cliff past ~32 concurrent
    # trials on this 64-physical-core box.
    cpus_used = int(cpus_per_trial)
    if "GPU" in resources:
        gpus_used = 1

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)
    _, val_metric, higher_is_better = get_metrics(
        task.task_type, num_classes=getattr(task, "num_classes", None)
    )
    # Episode width. Defaults to the dataset's own validation window
    # (test_timestamp - val_timestamp), which is what every published run used and
    # what makes the increment size an accident of how RelBench happened to split
    # the data rather than a controlled variable. `--val_delta_days` makes it one.
    split_delta = (
        pd.Timedelta(days=val_delta_days) if val_delta_days is not None else None
    )
    splits = copy.deepcopy(wrapped_task.get_splits(val_delta=split_delta))
    if split_delta is not None:
        print(
            f"Episode width overridden to {val_delta_days} days: "
            f"{len(splits) - 2} episodes (default gives the val-window width)",
            flush=True,
        )
    del wrapped_task
    del task
    best_weights_path = None
    best_cl_state_path = None

    # Identifies this chain, so `--resume` cannot mistake another chain's runs --
    # notably the published grid, which shares dataset/task names but whose
    # checkpoints live on a machine this one cannot reach -- for its own.
    chain_id = f"{dataset_name}/{task_name}/{learning_mode}/{model_save_dir}"
    
    model_save_dir = Path(model_save_dir).absolute()
    
    cache_path = Path(cache_dir).absolute() / dataset_name
    
    start_inc = 1
    
    if resume:
        resume_inc, resume_weights_path, resume_state_path = (
            get_resume_state_from_mlflow(
                mlflow_experiment, dataset_name, task_name, val_metric,
                higher_is_better, mlflow_uri=mlflow_uri,
                expected_trials=len(seeds), chain_id=chain_id,
            )
        )
        if resume_inc > 1 and resume_weights_path is not None:
            chain_spec = resolve_mode(learning_mode)
            if chain_spec.needs_chain_state and resume_state_path is None:
                # Continuing a replay or EWC chain without its buffer or anchor
                # would run one episode with the method silently disabled and lose
                # the pre-resume history for good. Fail rather than warn: worker
                # stdout is swallowed by log_to_driver=False.
                raise RuntimeError(
                    f"cannot resume a {learning_mode!r} chain at increment "
                    f"{resume_inc}: its weights were found but {CLState.FILENAME} "
                    f"was not. Restart the chain instead of resuming it."
                )
            start_inc = resume_inc
            best_weights_path = resume_weights_path
            best_cl_state_path = resume_state_path
            print(
                f"Resuming from increment {resume_inc} with weights "
                f"{resume_weights_path} and CL state {resume_state_path}"
            )
        else:
            print("No valid resume state found. Starting from scratch.")
            
    if start_inc >= len(splits) - 1:
        print("All increments are already completed according to MLflow. Exiting.")
        return {"completed": 0, "expected": 0}

    last_inc = len(splits) - 1
    if max_increments is not None:
        last_inc = min(last_inc, start_inc + max_increments)
        print(f"Limiting to {max_increments} increment(s): {start_inc}..{last_inc - 1}")

    completed = 0
    for i in range(start_inc, last_inc):
        train_timestamp = splits[i]
        val_timestamp = splits[i+1]
        prev_train_timestamp = splits[i-1] if i > 1 else None
        
        current_learning_mode = "from_scratch" if i == 1 else learning_mode

        tuner = tune.Tuner(
            tune.with_resources(
                run_continuous_learning_experiment,
                resources={"CPU": cpus_used, "GPU": gpus_used},
            ),
            run_config=ray_train.RunConfig(
                name=ray_experiment_name,
                storage_path=ray_storage_path,
                stop={"time_total_s": 3600 * 4},
                log_to_file=True,
            ),
            tune_config=tune.TuneConfig(
                # seeds are a grid_search axis, so one sample per seed
                num_samples=1,
                trial_name_creator=lambda trial: (
                    f"{dataset_name}_{task_name}_{i}_{trial.trial_id}"
                ),
                trial_dirname_creator=lambda trial: trial.trial_id,
                max_concurrent_trials=num_cpus,
            ),
            param_space={
                "dataset_name": dataset_name,
                "task_name": task_name,
                "learning_mode": current_learning_mode,
                # what the chain is, versus what this episode runs -- episode 1 is
                # always from_scratch, but its buffer/anchor must still be built
                "chain_learning_mode": learning_mode,
                "chain_id": chain_id,
                "seed": tune.grid_search(seeds),
                "text_embedder_name": "glove",
                "mlflow_experiment": mlflow_experiment,
                "mlflow_uri": mlflow_uri,
                "max_training_steps": max_training_steps,
                # 100 batches is not an epoch: it is the step-budget protocol, where
                # cost is equalised across episodes by capping batches rather than
                # passes. The budget-sensitivity arm needs the other regime -- a real
                # pass over the episode's own data -- so both are now expressible.
                "limit_train_batches": limit_train_batches,
                "max_epochs": max_epochs,
                "val_check_interval": val_check_interval,
                "val_max_rows": val_max_rows,
                "val_delta_days": val_delta_days,
                "torch_threads": torch_threads,
                "increment": i,
                "train_timestamp": train_timestamp,
                "val_timestamp": val_timestamp,
                "prev_train_timestamp": prev_train_timestamp,
                "weights_path": best_weights_path if learning_mode != "from_scratch" else None,
                "cl_state_path": best_cl_state_path,
                # CL method hyperparameters, logged with every run
                "buffer_size": buffer_size,
                "replay_ratio": replay_ratio,
                "der_alpha": der_alpha,
                "ewc_lambda": ewc_lambda,
                "ewc_gamma": ewc_gamma,
                "lwf_alpha": lwf_alpha,
                "fisher_batches": fisher_batches,
                "lr": 0.001,
                "batch_size": 128,
                "num_neighbors": 32,
                "gnn_channels": 128,
                "gnn_layers": 2,
                "gnn_aggr": "sum",
                "head_norm": "batch_norm",
                "cache_path": cache_path,
                "model_save_dir": model_save_dir / f"increment_{i}",
            },
        )
        results = tuner.fit()

        # A single flaky trial must not kill the whole episode chain: a sweep can
        # run for days, and the chain only truly fails when no trial survived to
        # produce weights for the next episode.
        if results.errors:
            n_failed = sum(1 for e in results.errors if e is not None)
            print(
                f"Increment {i}: {n_failed}/{len(results)} trials failed; "
                f"continuing with the survivors."
            )
            if n_failed >= len(results):
                print(f"All trials failed in split {i}. Stopping continuous learning.")
                break

        best_result = results.get_best_result(metric=f"val_{val_metric}", mode="max" if higher_is_better else "min")
        if best_result is None or "model_save_dir" not in best_result.metrics:
            print(f"Failed to find the best result in split {i}. Stopping.")
            break
        
        completed += 1
        best_weights_path = f"{best_result.metrics['model_save_dir']}/best_model.pt"
        # The buffer/anchor of the *selected* trial travel forward with its weights,
        # so the chain state always matches the checkpoint it was produced with.
        candidate_state = Path(best_result.metrics["model_save_dir"]) / CLState.FILENAME
        best_cl_state_path = str(candidate_state) if candidate_state.exists() else None

    expected = max(last_inc - start_inc, 0)
    if completed < expected:
        print(
            f"INCOMPLETE: {completed}/{expected} increments finished for "
            f"{dataset_name}/{task_name} [{learning_mode}]",
            flush=True,
        )
    return {"completed": completed, "expected": expected}



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Explicit per-trial seeds. Defaults to num_samples values drawn "
             "deterministically from --seed, matching the original runs.",
    )
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--max_increments", type=int, default=None,
        help="Stop after this many increments. For calibration and smoke runs.",
    )
    parser.add_argument("--max_training_steps", type=int, default=2000)
    parser.add_argument(
        "--limit_train_batches", type=int, default=100,
        help="Cap each epoch at N batches. The default is the step-budget "
             "protocol. Pass 0 to let an epoch be a full pass over the episode's "
             "training set, which is what --max_epochs then counts.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Stop after N epochs. With --limit_train_batches 0 this is a real "
             "epoch budget, so later episodes with more history train longer -- "
             "the opposite of the fixed step budget, and the control for it.",
    )
    parser.add_argument(
        "--val_check_interval", type=int, default=100,
        help="Validate every N optimiser steps. 0 restores the old epoch-based "
             "behaviour, where an epoch was min(limit_train_batches, len(loader)) "
             "and short episodes validated far more often.",
    )
    parser.add_argument(
        "--val_max_rows", type=int, default=25_000,
        help="Cap the validation window by uniform subsample, seeded on "
             "(dataset, task, increment) so it is identical across methods and "
             "seeds. 0 disables the cap. Affects model selection only.",
    )
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument(
        "--gpu_ids", type=int, nargs="*", default=None,
        help="Physical GPU ids to pin this run to, bypassing free-memory "
             "auto-selection. Use when an external scheduler owns the allocation.",
    )
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument(
        "--learning_mode", type=str, default="from_scratch",
        choices=sorted(set(MODES) | set(MODE_ALIASES)),
        help="Roster: " + ", ".join(DEFAULT_ROSTER)
             + ". ft_full/ft_newonly are aliases of joint/naive; ft_upsample is "
               "retained only to reproduce the submitted paper.",
    )
    parser.add_argument(
        "--val_delta_days", type=float, default=None,
        help="Episode width in days. Default: the dataset's validation window "
             "(test_timestamp - val_timestamp). Set this to sweep increment size; "
             "it cannot go below the task's own timedelta, and the 10%% row filter "
             "will silently drop episodes that come out too small.",
    )
    parser.add_argument(
        "--cpus_per_trial", type=int, default=2,
        help="CPU cores Ray reserves per trial. Use 1 for CPU-only runs: threads "
             "scale badly (8 threads buy 2.2x) while processes scale well.",
    )
    parser.add_argument(
        "--torch_threads", type=int, default=1,
        help="torch intra-op threads per trial. Keep at 1 unless running a single "
             "trial alone; oversubscribing across concurrent trials is worse than "
             "running more trials.",
    )
    parser.add_argument("--buffer_size", type=int, default=10_000,
                        help="Replay buffer capacity for er/der_pp.")
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                        help="Share of each epoch drawn from the new increment.")
    parser.add_argument("--der_alpha", type=float, default=0.5)
    parser.add_argument("--ewc_lambda", type=float, default=100.0)
    parser.add_argument("--ewc_gamma", type=float, default=0.9)
    parser.add_argument("--lwf_alpha", type=float, default=1.0)
    parser.add_argument("--fisher_batches", type=int, default=64)

    parser.add_argument("--model_save_dir", type=str, default="./models")
    parser.add_argument("--resume", action="store_true", default=False)
    
    
    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    task_name = args.task

    summary = run_ray_tuner(
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
        seeds=args.seeds,
        max_increments=args.max_increments,
        max_training_steps=args.max_training_steps,
        limit_train_batches=(args.limit_train_batches or None),
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval or None,
        val_max_rows=args.val_max_rows or None,
        val_delta_days=args.val_delta_days,
        cpus_per_trial=args.cpus_per_trial,
        torch_threads=args.torch_threads,
        buffer_size=args.buffer_size,
        replay_ratio=args.replay_ratio,
        der_alpha=args.der_alpha,
        ewc_lambda=args.ewc_lambda,
        ewc_gamma=args.ewc_gamma,
        lwf_alpha=args.lwf_alpha,
        fisher_batches=args.fisher_batches,
        gpu_ids=args.gpu_ids,
        mlflow_experiment=args.mlflow_experiment,
        random_seed=args.seed,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
        learning_mode=args.learning_mode,
        model_save_dir=args.model_save_dir,
        resume=args.resume,
    )

    # A chain that stopped early must not look like success: scripts/run_grid.py
    # writes its "done" marker on exit code 0, so a silent early stop would be
    # recorded as a complete cell and never retried.
    if summary and summary["completed"] < summary["expected"]:
        sys.exit(1)
