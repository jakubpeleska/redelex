r"""Run one chain: one (dataset, task, mode, seed) through every update.

A chain is the unit of the rerun. Seed ``s`` continues its OWN previous model
from update to update (no best-of-k parent), so five seeds are five
independent chains. Each update ``e`` at time ``T_e``:

1. builds the model: fresh (seeded per update) for a cold regime, or this
   chain's previous model for a warm one. A cold regime never loads weights;
2. trains on rows labelled by ``T_e`` (``t <= T_e - Δw``) that are not held
   out, mixed as the regime says (:mod:`.modes`);
3. stops on the held-out entities' rows of the whole last episode (loss checked
   every N optimiser steps, patience P), never on the next window;
4. scores the rows the analysis needs and writes ``preds.parquet``;
5. writes ``episode.json`` last, atomically: the commit marker resume reads.

Layout::

    {out}/{chain_key}/rows.parquet          what every task row is (shared by seeds)
    {out}/{chain_key}/seed{s}/chain.json    config + protocol hash
    {out}/{chain_key}/seed{s}/ep{e:03d}/    model.pt, best_P*.pt, preds*.parquet, episode.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from relbench.base import Table

from redelex.continual.anchors import ROW_COL, grid_anchor_timestamps, labelled_anchor_range
from redelex.continual.protocol import EpisodePlan, Protocol, grid_update_times
from redelex.continual.regularization import ParameterAnchor, fisher_diagonal
from redelex.continual.replay import ReservoirBuffer
from redelex.continual.shrink_perturb import shrink_perturb_
from redelex.loaders import MixtureLoader, Source, epoch_steps
from redelex.nn.train import (
    AnytimeEvalCallback,
    ConvergenceCallback,
    CostMeterCallback,
    LightningEntityTaskWrapper,
)
from redelex.transforms import AttachTargetTransform

from experiments.continuous_learning.continuous_task import ContinuousWrapper, assert_no_holdout
from experiments.continuous_learning.models import HeterogeneousSAGE
from experiments.continuous_learning.modes import ModeSpec, get_mode
from experiments.continuous_learning.score_chain import (
    make_eval_loader,
    metric_name,
    metric_value,
    needed_rows,
    predict_table,
    rows_frame,
    write_rows_table,
)
from experiments.continuous_learning.utils import get_table_input

# relbench 2.1.1 workarounds (notes/DECISIONS.md): rel-stack's db.zip hash is
# stale upstream, and rel-amazon cannot be rebuilt from raw.
DOWNLOAD = {"rel-amazon": True, "rel-stack": False}

EPISODE_FILE = "episode.json"

# Every update is its own short Trainer run without a Lightning logger: these
# messages would repeat for each one and bury the output that matters.
warnings.filterwarnings("ignore", message=r".*does not have many workers.*")
warnings.filterwarnings("ignore", message=r".*but have no logger configured.*")
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


@dataclass
class ChainConfig:
    dataset: str
    task: str
    mode: str
    seed: int
    out_dir: str
    variant: str = "main"
    # Protocol. None = the task's native value: ΔI = test - val span, stride = Δw.
    # delta_i_days = 0 means "never update after the first" (ΔI = ∞).
    delta_i_days: Optional[float] = None
    # Start the update grid at this date (ISO) instead of the first update with
    # enough rows; it becomes the grid's reference, so every ΔI shares it (RQ4).
    first_update: Optional[str] = None
    # Adopt this model at the first update instead of training one (RQ4: every
    # configuration starts from the same model trained up to t0).
    init_model: Optional[str] = None
    anchor_stride_days: Optional[float] = None
    anchor_phase_days: float = 0.0
    holdout_frac: float = 0.1
    min_initial_rows: int = 256
    # Training.
    lr: float = 1e-3
    batch_size: int = 128
    gnn_channels: int = 128
    gnn_layers: int = 2
    num_neighbors: int = 32
    max_steps: int = 20_000
    # Stopping: check every `check_every` steps, or every `check_every_epochs`
    # of an epoch over the training rows (at least `check_min_steps`).
    check_every: Optional[int] = 50
    check_every_epochs: Optional[float] = None
    check_min_steps: int = 10
    patiences: tuple = (5, 10)
    monitor: str = "val_loss"
    # "convergence": stop on held-out loss. "budget": train exactly max_steps
    # steps, no validation, keep the final weights (RQ4's steps per update B).
    stop_rule: str = "convergence"
    min_selection_rows: int = 32
    selection_max_rows: Optional[int] = None
    # Reporting.
    anytime: bool = True
    anytime_max_rows: int = 25_000
    anytime_schedule: tuple = (0, 10, 25, 50, 100, 250, 500, 1000, 2000)
    # Calibration only: score the next window every `anytime_every` steps instead.
    anytime_every: Optional[int] = None
    # Calibration only: halve the LR after this many checks without improvement
    # of the validation metric (default: constant LR).
    plateau_patience: Optional[int] = None
    stale_windows: Optional[int] = None
    eval_batch_size: int = 512
    # Keep every patience's best weights (best_P*.pt) after their predictions are
    # written. Off by default: model.pt is all a chain needs to continue or rescore.
    keep_patience_models: bool = False
    # Runtime.
    device: str = "auto"
    torch_threads: Optional[int] = None
    cache_dir: str = ".cache"
    text_embedder: str = "glove"
    max_updates: Optional[int] = None
    deadline: Optional[float] = None
    deadline_margin_s: float = 120.0
    mlflow_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def chain_key(self) -> str:
        return f"{self.dataset}__{self.task}__{self.mode}__{self.variant}"

    def chain_root(self) -> Path:
        return Path(self.out_dir) / self.chain_key

    def seed_dir(self) -> Path:
        return self.chain_root() / f"seed{self.seed}"


@dataclass
class ExperimentContext:
    dataset_name: str
    task_name: str
    task: Any
    db: Any
    wrapper: ContinuousWrapper
    data: Any
    col_stats: dict
    plan: EpisodePlan
    protocol: Protocol
    rows: pd.DataFrame
    device: torch.device


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _days(value: Optional[float]) -> Optional[pd.Timedelta]:
    return None if value is None else pd.Timedelta(days=value)


def first_update_index(wrapper: ContinuousWrapper, times: list, min_rows: int) -> int:
    """First update with at least ``min_rows`` trainable (non-held-out) rows."""
    for i, t in enumerate(times):
        if len(wrapper.trainable(t).df) >= min_rows:
            return i
    raise ValueError(f"no update time has {min_rows} trainable rows")


def build_context(
    cfg: ChainConfig,
    *,
    dataset=None,
    task=None,
    text_embedder=None,
    col_to_stype: Optional[dict] = None,
) -> ExperimentContext:
    """Load data once: database, task table on the anchor grid, graph, plan.

    ``dataset``, ``task``, ``text_embedder`` and ``col_to_stype`` can be
    injected (tests, notebooks); otherwise they come from relbench and the cache.
    """
    from redelex.data import make_pkey_fkey_graph

    if dataset is None:
        from relbench.datasets import get_dataset

        dataset = get_dataset(cfg.dataset, download=DOWNLOAD.get(cfg.dataset, False))
    if task is None:
        from relbench.tasks import get_task

        task = get_task(cfg.dataset, cfg.task)
    db = dataset.get_db(upto_test_timestamp=False)
    cache = Path(cfg.cache_dir) / cfg.dataset

    dw = task.timedelta
    stride = _days(cfg.anchor_stride_days) or dw
    phase = pd.Timedelta(days=cfg.anchor_phase_days)
    start, end = labelled_anchor_range(task, db)
    anchors = grid_anchor_timestamps(dataset.val_timestamp, stride, start, end, phase=phase)
    salt = f"{cfg.dataset}/{task.entity_table}/v1"
    anchor_cache = (
        None
        if cfg.extra.get("no_anchor_cache")
        else cache / "anchors" / f"{cfg.task}__s{stride.days}d__p{phase.days}d.parquet"
    )
    wrapper = ContinuousWrapper.from_anchors(
        task, db, anchors, cfg.holdout_frac, salt, cache_path=anchor_cache
    )

    native = dataset.test_timestamp - dataset.val_timestamp
    if cfg.delta_i_days is None:
        delta_i = native
    elif cfg.delta_i_days == 0:
        delta_i = None  # ΔI = ∞: the first model serves the whole horizon
    else:
        delta_i = pd.Timedelta(days=cfg.delta_i_days)
    reference = pd.Timestamp(cfg.first_update) if cfg.first_update else dataset.val_timestamp
    times = grid_update_times(reference, delta_i or native, first=anchors.min(), last=anchors.max())
    if cfg.first_update:
        times = [t for t in times if t >= reference]
    times = times[first_update_index(wrapper, times, cfg.min_initial_rows):]
    if delta_i is None:
        times = times[:1]
    plan = EpisodePlan(tuple(times), horizon_end=anchors.max() + pd.Timedelta(days=1), delta_w=dw)

    protocol = Protocol(
        dataset=cfg.dataset,
        task=cfg.task,
        delta_w=dw,
        delta_i=delta_i,
        anchor_stride=stride,
        anchor_phase=phase,
        holdout_frac=cfg.holdout_frac,
        holdout_salt=salt,
        extra={"min_initial_rows": cfg.min_initial_rows, "first_update": times[0],
               "first_update_requested": cfg.first_update},
    )

    if col_to_stype is None:
        from experiments.continuous_learning.utils import get_attribute_schema

        col_to_stype = get_attribute_schema(str(cache / "attribute-schema.json"), db)
    if text_embedder is None:
        from experiments.continuous_learning.utils import get_text_embedder

        text_embedder = get_text_embedder(cfg.text_embedder, device=torch.device("cpu"))
    graph_cache = None if cfg.extra.get("no_graph_cache") else str(cache / "materialized")
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict={t: dict(c) for t, c in col_to_stype.items()},
        text_embedder=text_embedder,
        cache_dir=graph_cache,
    )
    rows = rows_frame(wrapper.full_table, task, plan)
    return ExperimentContext(
        cfg.dataset, cfg.task, task, db, wrapper, data, col_stats, plan, protocol, rows, _device(cfg.device)
    )


def update_seed(seed: int, episode: int) -> int:
    return zlib.crc32(f"{seed}/{episode}".encode()) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_model(ctx: ExperimentContext, cfg: ChainConfig) -> HeterogeneousSAGE:
    return HeterogeneousSAGE(
        data=ctx.data,
        col_stats_dict=ctx.col_stats,
        gnn_channels=cfg.gnn_channels,
        gnn_layers=cfg.gnn_layers,
    )


def training_sources(
    ctx: ExperimentContext, mode: ModeSpec, ep, buffer: Optional[ReservoirBuffer] = None
) -> dict[str, tuple[Table, float]]:
    """The tables an update trains on, with their sample shares."""
    w, T = ctx.wrapper, ep.update_time
    if mode.buffer_size is not None:
        # Bounded replay: the last episode plus the reservoir of earlier rows. The
        # buffer is read back through the held-out filter like every other table.
        held = buffer.row_ids if buffer is not None else []
        return {
            "new": (w.trainable(T, after=ep.new_after), 1.0 - mode.buffer_share),
            "buffer": (w.get_rows(held, part="train"), mode.buffer_share),
        }
    if mode.history == "all":
        if mode.new_share is None:
            return {"all": (w.trainable(T), 1.0)}
        # Older rows are those labelled by the previous update. At the first
        # update there are none, and the mixture gives their share to "new".
        old = w.trainable(ep.prev_update_time) if ep.prev_update_time is not None else w.get_table(T, T, part="train")
        return {
            "new": (w.trainable(T, after=ep.new_after), mode.new_share),
            "old": (old, 1.0 - mode.new_share),
        }
    if mode.history == "last":
        return {"new": (w.trainable(T, after=ep.new_after), 1.0)}
    # last_k: rows newly labelled over the last k updates.
    k = mode.window_episodes
    eps = ctx.plan.episodes
    oldest = ep.index - k  # 1-based index of the update whose cutoff bounds the window
    after = None if oldest < 1 else eps[oldest - 1].labelled_until
    return {"window": (w.trainable(T, after=after), 1.0)}


def _subsample(table: Table, max_rows: Optional[int], key: str) -> Table:
    if max_rows is None or len(table.df) <= max_rows:
        return table
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    keep = np.sort(rng.choice(len(table.df), size=max_rows, replace=False))
    return Table(
        df=table.df.iloc[keep].reset_index(drop=True),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def selection_table(ctx: ExperimentContext, cfg: ChainConfig, ep) -> tuple[Table, str]:
    """Held-out rows of the whole last episode; wider if that is too small."""
    w = ctx.wrapper
    sel = w.trainable(ep.update_time, after=ep.new_after, part="holdout")
    source = "last_episode"
    if len(sel.df) < cfg.min_selection_rows:
        sel = w.trainable(ep.update_time, part="holdout")
        source = "all_labelled_history"
    sel = _subsample(sel, cfg.selection_max_rows, f"{ctx.dataset_name}/{ctx.task_name}/{ep.index}/selection")
    return sel, source


def _neighbors(cfg: ChainConfig) -> list[int]:
    return [int(cfg.num_neighbors / 2**i) for i in range(cfg.gnn_layers)]


def _check_interval(cfg: ChainConfig, n_train_rows: int) -> int:
    if cfg.check_every_epochs is not None:
        return epoch_steps(n_train_rows, cfg.batch_size, cfg.check_every_epochs, minimum=cfg.check_min_steps)
    return int(cfg.check_every)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


class _SeedAudit(L.Callback):
    """Test hook: records the entity id of every training seed."""

    def __init__(self, entity_table: str, sink: list):
        self.entity_table, self.sink = entity_table, sink

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        n = batch[self.entity_table].batch_size
        self.sink.extend(batch[self.entity_table].n_id[:n].tolist())


def run_update(ctx: ExperimentContext, cfg: ChainConfig, mode: ModeSpec, ep, prev_model: Optional[Path], ep_dir: Path) -> dict:
    task, device = ctx.task, ctx.device
    seed = update_seed(cfg.seed, ep.index)
    seed_everything(seed)
    t_setup = time.perf_counter()

    model = new_model(ctx, cfg)
    loaded = False
    adopt = ep.index == 1 and cfg.init_model is not None
    if adopt and (mode.ewc_lambda is not None or mode.buffer_size is not None):
        raise NotImplementedError("init_model is supported for stateless regimes only")
    if adopt:
        model.load_state_dict(torch.load(cfg.init_model, map_location="cpu"))
        loaded = True
    elif mode.warm_start and prev_model is not None:
        fresh = new_model(ctx, cfg) if mode.shrink is not None else None
        model.load_state_dict(torch.load(prev_model, map_location="cpu"))
        loaded = True
        if fresh is not None:
            shrink_perturb_(model, fresh, mode.shrink, mode.perturb)
    elif mode.warm_start and ep.index > 1:
        raise RuntimeError(f"{mode.name} needs the previous model at update {ep.index}")

    # Chain state of the stateful ablations travels next to the previous model.
    state = {}
    if (mode.ewc_lambda is not None or mode.buffer_size is not None) and prev_model is not None:
        state = torch.load(prev_model.parent / "cl_state.pt", map_location="cpu", weights_only=False)
    buffer = anchor = None
    if mode.buffer_size is not None:
        buffer = (ReservoirBuffer.from_state_dict(state["buffer"]) if state.get("buffer")
                  else ReservoirBuffer(mode.buffer_size, seed=update_seed(cfg.seed, 0)))
    if mode.ewc_lambda is not None:
        anchor = (ParameterAnchor.from_state_dict(state["anchor"]) if state.get("anchor")
                  else ParameterAnchor(lam=mode.ewc_lambda, gamma=0.9))

    sources = {} if adopt else training_sources(ctx, mode, ep, buffer=buffer)
    for name, (table, _) in sources.items():
        assert_no_holdout(table)
    deploy = ctx.wrapper.get_table(*ep.deploy_span, part="all")
    if adopt:
        model.to(device)
        torch.save(model.state_dict(), ep_dir / "model.pt")
        return _score_and_record(ctx, cfg, mode, ep, ep_dir, model, deploy, seed, loaded, sources=sources,
                                 meter=CostMeterCallback(), convergence=None, anytime_cb=None,
                                 selection=None, selection_source=None, check_every=None,
                                 steps_per_epoch=None, anytime_rows=0, setup_s=time.perf_counter() - t_setup,
                                 optimizer=None, anchor=None, buffer=None, loss_fn=None)
    n_train = sum(len(t.df) for t, _ in sources.values())
    if n_train == 0:
        raise RuntimeError(f"update {ep.index}: no training rows for {mode.name}")
    loader_sources = {}
    for name, (table, share) in sources.items():
        inp = get_table_input(table, task)
        loader_sources[name] = Source(inp.nodes[1], inp.time, inp.target, share)
    steps_per_epoch = epoch_steps(n_train, cfg.batch_size)
    train_loader = MixtureLoader(
        ctx.data,
        task.entity_table,
        loader_sources,
        num_neighbors=_neighbors(cfg),
        batch_size=cfg.batch_size,
        num_samples=steps_per_epoch * cfg.batch_size,
        seed=seed,
        transform_factory=AttachTargetTransform,
    )

    budget = cfg.stop_rule == "budget"
    if cfg.stop_rule not in ("convergence", "budget"):
        raise ValueError(f"unknown stop_rule {cfg.stop_rule!r}")
    selection, selection_source = (None, None) if budget else selection_table(ctx, cfg, ep)
    val_loader = None if budget else make_eval_loader(ctx.data, task, selection, _neighbors(cfg), cfg.eval_batch_size)
    anytime_table = _subsample(deploy, cfg.anytime_max_rows, f"{ctx.dataset_name}/{ctx.task_name}/{ep.index}/anytime")

    meter = CostMeterCallback(entity_table=task.entity_table, source_counts=train_loader.source_counts)
    convergence = None if budget else ConvergenceCallback(
        str(ep_dir),
        patiences=tuple(cfg.patiences),
        monitor=cfg.monitor,
        mode="min" if cfg.monitor == "val_loss" or cfg.monitor.endswith("mae") else "max",
        also_record=(f"val_{metric_name(task.task_type)}", "val_loss"),
    )
    callbacks = [meter] + ([] if budget else [convergence])
    anytime_cb = None
    if cfg.anytime and len(anytime_table.df):
        target = anytime_table.df[task.target_col].to_numpy()

        def evaluate(m):
            pred = predict_table(m, ctx.data, task, anytime_table, _neighbors(cfg), cfg.eval_batch_size, device, seed=0)
            return {metric_name(task.task_type): metric_value(task.task_type, target, pred)}

        schedule = tuple(cfg.anytime_schedule)
        if cfg.anytime_every is not None:
            schedule = tuple(range(0, cfg.max_steps + 1, int(cfg.anytime_every)))
        anytime_cb = AnytimeEvalCallback(evaluate, base_schedule=schedule, meter=meter)
        callbacks.append(anytime_cb)

    audit = cfg.extra.get("audit_seen")
    if audit is not None:
        callbacks.append(_SeedAudit(task.entity_table, audit))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_cfg = None
    if cfg.plateau_patience is not None:
        higher = task.task_type.value != "regression"
        scheduler_cfg = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max" if higher else "min", factor=0.5, patience=int(cfg.plateau_patience))}
    lightning_model = LightningEntityTaskWrapper(
        model=model, optimizer=optimizer, task=task, modes=("val",), lr_scheduler_config=scheduler_cfg
    )
    if anchor is not None and len(anchor):
        lightning_model.cl_penalty = anchor.penalty
    logger = False
    if cfg.mlflow_uri:
        from lightning.pytorch.loggers import MLFlowLogger

        logger = MLFlowLogger(
            experiment_name=cfg.mlflow_experiment or f"cl2_{cfg.variant}",
            run_name=f"{cfg.chain_key}__s{cfg.seed}__e{ep.index:03d}",
            tracking_uri=cfg.mlflow_uri,
        )
        logger.log_hyperparams({"chain_key": cfg.chain_key, "seed": cfg.seed, "episode": ep.index,
                                "update_time": str(ep.update_time), "protocol_hash": ctx.protocol.hash(),
                                "mode": cfg.mode, "variant": cfg.variant})
    check_every = None if budget else _check_interval(cfg, n_train)
    val_kwargs = (dict(limit_val_batches=0) if budget
                  else dict(val_check_interval=check_every, check_val_every_n_epoch=None))
    trainer = L.Trainer(
        max_steps=cfg.max_steps,
        **val_kwargs,
        accelerator=device.type,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
    )
    setup_s = time.perf_counter() - t_setup
    trainer.fit(lightning_model, train_loader.loader, val_loader)

    # Continue (and score) from the primary patience's best weights, or from the
    # final weights under a fixed budget.
    if not budget:
        model.load_state_dict(torch.load(ep_dir / "best_model.pt", map_location="cpu"))
    model.to(device)
    torch.save(model.state_dict(), ep_dir / "model.pt")
    return _score_and_record(ctx, cfg, mode, ep, ep_dir, model, deploy, seed, loaded, sources=sources,
                             meter=meter, convergence=convergence, anytime_cb=anytime_cb,
                             selection=selection, selection_source=selection_source, check_every=check_every,
                             steps_per_epoch=steps_per_epoch, anytime_rows=len(anytime_table.df), setup_s=setup_s,
                             optimizer=optimizer, anchor=anchor, buffer=buffer, loss_fn=lightning_model.loss_fn)


def _score_and_record(ctx, cfg, mode, ep, ep_dir, model, deploy, seed, loaded, *, sources, meter, convergence,
                      anytime_cb, selection, selection_source, check_every, steps_per_epoch, anytime_rows,
                      setup_s, optimizer, anchor, buffer, loss_fn) -> dict:
    task, device = ctx.task, ctx.device
    t_score = time.perf_counter()

    mask = needed_rows(ctx.rows, ep.index, cfg.stale_windows)
    score_rows = ctx.wrapper.full_table.df[ROW_COL].isin(ctx.rows[ROW_COL][mask])
    to_score = Table(
        df=ctx.wrapper.full_table.df[score_rows.to_numpy()].reset_index(drop=True),
        fkey_col_to_pkey_table=ctx.wrapper.full_table.fkey_col_to_pkey_table,
        pkey_col=ctx.wrapper.full_table.pkey_col,
        time_col=ctx.wrapper.full_table.time_col,
    )
    pred = predict_table(model, ctx.data, task, to_score, _neighbors(cfg), cfg.eval_batch_size, device)
    pd.DataFrame({ROW_COL: to_score.df[ROW_COL].to_numpy(), "pred": pred}).to_parquet(ep_dir / "preds.parquet", index=False)

    # Every other patience: its own next-window predictions (sensitivity of the stopping rule).
    deploy_scores = {}
    for p in (convergence.patiences if convergence is not None else ()):
        path = ep_dir / f"best_P{p}.pt"
        if p != convergence.primary and path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"))
            pp = predict_table(model, ctx.data, task, deploy, _neighbors(cfg), cfg.eval_batch_size, device)
            pd.DataFrame({ROW_COL: deploy.df[ROW_COL].to_numpy(), "pred": pp}).to_parquet(
                ep_dir / f"preds_P{p}.parquet", index=False
            )
            deploy_scores[str(p)] = metric_value(task.task_type, deploy.df[task.target_col], pp)
    model.load_state_dict(torch.load(ep_dir / "model.pt", map_location="cpu"))
    cl_state = None
    if anchor is not None or buffer is not None:
        new_rows = sources["new"][0]
        if anchor is not None and len(new_rows.df):
            et = task.entity_table
            fisher_loader = make_eval_loader(ctx.data, task, new_rows, _neighbors(cfg), cfg.batch_size)
            fisher = fisher_diagonal(
                model,
                (b.to(device) for b in fisher_loader),
                loss_fn,
                lambda m, b: m(b, et).view(-1)[: b[et].batch_size],
                lambda b: b[et].y.float()[: b[et].batch_size],
                max_batches=64,
            )
            anchor.consolidate(model, fisher)
        if buffer is not None:
            buffer.add(new_rows.df[ROW_COL].to_numpy())
        torch.save({"anchor": anchor.state_dict() if anchor is not None else None,
                    "buffer": buffer.state_dict() if buffer is not None else None}, ep_dir / "cl_state.pt")
        cl_state = {"anchor_params": len(anchor) if anchor is not None else None,
                    "anchor_updates": anchor.episodes if anchor is not None else None,
                    "buffer_rows": len(buffer) if buffer is not None else None,
                    "buffer_seen": buffer.seen if buffer is not None else None}
    deploy_pred = pd.Series(pred, index=to_score.df[ROW_COL].to_numpy())
    primary = str(convergence.primary) if convergence is not None else "final"
    deploy_scores[primary] = metric_value(
        task.task_type, deploy.df[task.target_col], deploy_pred.reindex(deploy.df[ROW_COL]).to_numpy()
    )
    meter.add("scoring", time.perf_counter() - t_score)
    meter.add("setup", setup_s)
    (ep_dir / "best_running.pt").unlink(missing_ok=True)
    (ep_dir / "best_model.pt").unlink(missing_ok=True)  # identical to model.pt
    if not cfg.keep_patience_models:
        for path in ep_dir.glob("best_P*.pt"):
            path.unlink()

    return {
        "episode": ep.index,
        "update_time": ep.update_time,
        "deploy_span": list(ep.deploy_span),
        "mode": mode.name,
        "seed": cfg.seed,
        "update_seed": seed,
        "warm_started": loaded,
        "trained": bool(sources),
        "stop_rule": cfg.stop_rule,
        "protocol_hash": ctx.protocol.hash(),
        "train_rows": {name: len(t.df) for name, (t, _) in sources.items()},
        "train_shares": {name: share for name, (_, share) in sources.items()},
        "selection_rows": None if selection is None else len(selection.df),
        "selection_entities": None if selection is None else int(selection.df[task.entity_col].nunique()),
        "selection_source": selection_source,
        "deploy_rows": len(deploy.df),
        "scored_rows": len(to_score.df),
        "check_every": check_every,
        "final_lr": None if optimizer is None else float(optimizer.param_groups[0]["lr"]),
        "cl_state": cl_state,
        "steps_per_epoch": steps_per_epoch,
        "metric": metric_name(task.task_type),
        "deploy_score": deploy_scores,
        "convergence": None if convergence is None else convergence.summary(),
        "cost": meter.summary(),
        "anytime": anytime_cb.records if anytime_cb is not None else [],
        "anytime_rows": anytime_rows,
    }


def completed_episodes(seed_dir: Path, protocol_hash: str) -> list[int]:
    """Episodes with a committed, protocol-matching ``episode.json``, as a prefix."""
    done = []
    e = 1
    while True:
        path = seed_dir / f"ep{e:03d}" / EPISODE_FILE
        if not path.exists():
            break
        record = json.loads(path.read_text())
        if record.get("protocol_hash") != protocol_hash:
            raise RuntimeError(f"{path} was written under protocol {record.get('protocol_hash')}, not {protocol_hash}")
        if not (path.parent / "model.pt").exists() or not (path.parent / "preds.parquet").exists():
            break
        if record.get("cl_state") and not (path.parent / "cl_state.pt").exists():
            break
        done.append(e)
        e += 1
    return done


def _slurm_deadline() -> Optional[float]:
    value = os.environ.get("SLURM_JOB_END_TIME")
    return float(value) if value else None


def run_chain(cfg: ChainConfig, ctx: Optional[ExperimentContext] = None, **build_kwargs) -> dict:
    """Run (or resume) a chain. Returns a status summary; never trains an episode twice."""
    mode = get_mode(cfg.mode)
    threads_before = torch.get_num_threads()
    if cfg.torch_threads is not None:
        torch.set_num_threads(int(cfg.torch_threads))
    try:
        return _run_chain(cfg, mode, ctx, build_kwargs)
    finally:
        torch.set_num_threads(threads_before)


# Config fields that define a chain's protocol and extent: if they match the
# recorded chain.json, a chain whose every update is committed is complete.
_IDENTITY_FIELDS = ("dataset", "task", "mode", "seed", "variant", "delta_i_days", "anchor_stride_days",
                    "anchor_phase_days", "holdout_frac", "min_initial_rows", "first_update", "max_updates")


def _already_complete(cfg: ChainConfig) -> Optional[dict]:
    """Recognise a finished chain without loading any data (a resumed job skips it in seconds)."""
    meta_path = cfg.seed_dir() / "chain.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    stored = meta.get("config", {})
    if any(stored.get(f) != getattr(cfg, f) for f in _IDENTITY_FIELDS):
        return None
    n = len(meta["update_times"]) if cfg.max_updates is None else min(cfg.max_updates, len(meta["update_times"]))
    try:
        done = completed_episodes(cfg.seed_dir(), meta["protocol_hash"])
    except RuntimeError:
        return None
    if len(done) < n:
        return None
    return {"chain_key": cfg.chain_key, "seed": cfg.seed, "status": "complete", "episodes_total": n,
            "episodes_done": len(done), "ran_now": []}


def _run_chain(cfg: ChainConfig, mode: ModeSpec, ctx: Optional[ExperimentContext], build_kwargs: dict) -> dict:
    if ctx is None:
        finished = _already_complete(cfg)
        if finished is not None:
            return finished
    ctx = ctx or build_context(cfg, **build_kwargs)
    seed_dir = cfg.seed_dir()
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cfg.chain_root() / "rows.parquet"
    if not rows_path.exists():
        write_rows_table(rows_path, ctx.rows)
    chain_meta = {"config": asdict(cfg), "protocol": ctx.protocol.as_dict(), "protocol_hash": ctx.protocol.hash(),
                  "update_times": [str(t) for t in ctx.plan.update_times], "horizon_end": str(ctx.plan.horizon_end)}
    meta_path = seed_dir / "chain.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["protocol_hash"] != chain_meta["protocol_hash"]:
            raise RuntimeError(f"{meta_path}: protocol changed ({old['protocol_hash']} -> {chain_meta['protocol_hash']})")
    _atomic_json(meta_path, chain_meta)

    episodes = ctx.plan.episodes
    if cfg.max_updates is not None:
        episodes = episodes[: cfg.max_updates]
    done = completed_episodes(seed_dir, ctx.protocol.hash())
    deadline = cfg.deadline or _slurm_deadline()
    durations = []
    prev_model = seed_dir / f"ep{done[-1]:03d}" / "model.pt" if done else None
    ran = []
    status = "complete"
    for ep in episodes:
        if ep.index in done:
            continue
        if deadline is not None and durations:
            needed = 1.5 * max(durations) + cfg.deadline_margin_s
            if time.time() + needed > deadline:
                status = "partial"
                break
        ep_dir = seed_dir / f"ep{ep.index:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        record = run_update(ctx, cfg, mode, ep, prev_model, ep_dir)
        record["wall_seconds"] = time.time() - start
        _atomic_json(ep_dir / EPISODE_FILE, record)
        durations.append(record["wall_seconds"])
        prev_model = ep_dir / "model.pt"
        ran.append(ep.index)
    return {
        "chain_key": cfg.chain_key,
        "seed": cfg.seed,
        "status": status,
        "episodes_total": len(episodes),
        "episodes_done": len(completed_episodes(seed_dir, ctx.protocol.hash())),
        "ran_now": ran,
    }


def parse_args(argv=None) -> ChainConfig:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    defaults = ChainConfig("", "", "", 0, "")
    for name, value in asdict(defaults).items():
        if name == "extra":
            continue
        flag = f"--{name}"
        if name in ("dataset", "task", "mode", "out_dir"):
            p.add_argument(flag, required=True)
        elif name == "seed":
            p.add_argument(flag, type=int, required=True)
        elif isinstance(value, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        elif isinstance(value, tuple):
            p.add_argument(flag, type=int, nargs="+", default=list(value))
        elif value is None and name in ("mlflow_uri", "mlflow_experiment", "first_update", "init_model"):
            p.add_argument(flag, type=lambda s: None if s.lower() == "none" else s, default=None)
        elif value is None:
            p.add_argument(flag, type=lambda s: None if s.lower() == "none" else float(s), default=None)
        else:
            p.add_argument(flag, type=type(value), default=value)
    args = vars(p.parse_args(argv))
    for name in ("patiences", "anytime_schedule"):
        args[name] = tuple(args[name])
    for name in ("check_every", "selection_max_rows", "stale_windows", "max_updates", "torch_threads",
                 "anytime_every", "plateau_patience"):
        if args.get(name) is not None:
            args[name] = int(args[name])
    return ChainConfig(**args)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    summary = run_chain(cfg)
    print(json.dumps(summary, default=str))
    return 0 if summary["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
