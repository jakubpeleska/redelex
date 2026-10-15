import json

import numpy as np
import pandas as pd
import pytest

from experiments.continuous_learning.chain_runner import (
    ChainConfig,
    build_context,
    completed_episodes,
    parse_args,
    run_chain,
    training_sources,
)
from experiments.continuous_learning.modes import ABLATION_ROSTER, MAIN_ROSTER, get_mode
from redelex.continual.anchors import HOLDOUT_COL, ROW_COL

from .conftest import STREAM_COL_TO_STYPE
from .helpers import FakeTextEmbedder, StreamChurnTask, StreamCountTask, StreamDataset

SMALL = dict(
    gnn_channels=8,
    num_neighbors=4,
    batch_size=64,
    max_steps=40,
    check_every=10,
    patiences=(2, 3),
    anytime_schedule=(0, 10, 25),
    min_initial_rows=128,
    holdout_frac=0.2,
    device="cpu",
    max_updates=3,
)


def _cfg(tmp_path, mode="cumulative", seed=0, task="churn", **kw):
    params = {**SMALL, **kw}
    return ChainConfig(
        dataset="stream", task=task, mode=mode, seed=seed, out_dir=str(tmp_path),
        extra={"no_anchor_cache": True, "no_graph_cache": True, **params.pop("extra", {})}, **params,
    )


_CTX = {}


def _ctx(cfg, task_cls=StreamChurnTask):
    key = (task_cls.__name__, cfg.holdout_frac, cfg.min_initial_rows, cfg.delta_i_days)
    if key not in _CTX:
        ds = StreamDataset()
        _CTX[key] = build_context(
            cfg, dataset=ds, task=task_cls(ds), text_embedder=FakeTextEmbedder(), col_to_stype=STREAM_COL_TO_STYPE
        )
    return _CTX[key]


def _episodes(cfg):
    out = []
    for e in completed_episodes(cfg.seed_dir(), json.loads((cfg.seed_dir() / "chain.json").read_text())["protocol_hash"]):
        out.append(json.loads((cfg.seed_dir() / f"ep{e:03d}" / "episode.json").read_text()))
    return out


def test_plan_has_several_updates_and_a_label_delay():
    cfg = _cfg("/unused")
    ctx = _ctx(cfg)
    eps = ctx.plan.episodes

    assert len(eps) >= 4
    for ep in eps:
        train = ctx.wrapper.trainable(ep.update_time).df
        assert train[ctx.task.time_col].max() <= ep.update_time - ctx.task.timedelta
        lo, hi = ep.deploy_span
        assert not train[ctx.task.time_col].between(lo, hi, inclusive="left").any()


@pytest.mark.parametrize("mode", [m.name for m in MAIN_ROSTER + ABLATION_ROSTER if m.ewc_lambda is None and m.buffer_size is None])
def test_every_mode_runs_three_updates(tmp_path, mode):
    cfg = _cfg(tmp_path, mode=mode)
    seen = []
    cfg.extra["audit_seen"] = seen
    summary = run_chain(cfg, ctx=_ctx(cfg))

    assert summary["status"] == "complete" and summary["episodes_done"] == 3
    eps = _episodes(cfg)
    spec = get_mode(mode)
    assert [e["warm_started"] for e in eps] == [False] + [spec.warm_start] * 2
    for e in eps:
        d = cfg.seed_dir() / f"ep{e['episode']:03d}"
        assert (d / "model.pt").exists() and (d / "best_P2.pt").exists() and (d / "best_P3.pt").exists()
        assert (d / "preds_P3.parquet").exists()
        assert e["selection_rows"] > 0 and e["cost"]["steps"] > 0
        assert e["anytime"][0]["step"] == 0
        assert set(e["deploy_score"]) == {"2", "3"}

    # No held-out entity is ever a training seed.
    ctx = _ctx(cfg)
    held = set(ctx.wrapper.full_table.df.loc[ctx.wrapper.full_table.df[HOLDOUT_COL], ctx.task.entity_col])
    assert seen and held and held.isdisjoint(seen)


def test_sources_match_each_regime():
    cfg = _cfg("/unused")
    ctx = _ctx(cfg)
    ep = ctx.plan.episodes[2]
    t = ctx.task.time_col

    def rows(mode):
        return {k: (v[0].df, v[1]) for k, v in training_sources(ctx, get_mode(mode), ep).items()}

    all_rows = rows("cumulative")["all"][0]
    new = rows("incremental")["new"][0]
    mix = rows("replay_p50")
    assert new[t].min() > ep.new_after and new[t].max() <= ep.labelled_until
    assert len(mix["new"][0]) + len(mix["old"][0]) == len(all_rows)
    assert mix["new"][1] == mix["old"][1] == 0.5
    assert rows("scratch_last")["new"][0].equals(new)
    k2 = rows("scratch_k2")["window"][0]
    assert len(new) < len(k2) < len(all_rows)


def test_resume_equals_an_uninterrupted_chain(tmp_path):
    full = _cfg(tmp_path / "full", mode="replay_p50")
    run_chain(full, ctx=_ctx(full))

    part = _cfg(tmp_path / "part", mode="replay_p50", max_updates=2)
    run_chain(part, ctx=_ctx(part))
    part.max_updates = 3
    again = run_chain(part, ctx=_ctx(part))

    assert again["ran_now"] == [3]
    a = pd.read_parquet(full.seed_dir() / "ep003" / "preds.parquet")
    b = pd.read_parquet(part.seed_dir() / "ep003" / "preds.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_preds_cover_next_window_and_past_holdout(tmp_path):
    cfg = _cfg(tmp_path, mode="incremental")
    ctx = _ctx(cfg)
    run_chain(cfg, ctx=ctx)
    rows = pd.read_parquet(cfg.chain_root() / "rows.parquet")
    preds = pd.read_parquet(cfg.seed_dir() / "ep002" / "preds.parquet").merge(rows, on=ROW_COL)

    assert set(preds.window[preds.window < 3].unique()) <= {1, 2}
    assert preds.loc[preds.window <= 2, "holdout"].all()  # past windows: held-out rows only
    deploy = rows[rows.window == 3]
    assert set(deploy[ROW_COL]) <= set(preds[ROW_COL])  # next window scored in full
    assert np.isfinite(preds.pred).all()


def test_regression_task_and_protocol_guard(tmp_path):
    cfg = _cfg(tmp_path, mode="scratch_all", task="count", max_updates=1)
    ctx = _ctx(cfg, StreamCountTask)
    run_chain(cfg, ctx=ctx)

    assert _episodes(cfg)[0]["metric"] == "mae"
    other = _cfg(tmp_path, mode="scratch_all", task="count", max_updates=1, holdout_frac=0.3)
    with pytest.raises(RuntimeError, match="protocol changed"):
        run_chain(other, ctx=_ctx(other, StreamCountTask))


def test_cli_round_trip():
    cfg = parse_args(["--dataset", "rel-f1", "--task", "driver-dnf", "--mode", "cumulative", "--seed", "3",
                      "--out_dir", "/tmp/x", "--patiences", "5", "10", "--check_every_epochs", "0.25",
                      "--mlflow_uri", "http://h:1", "--max_updates", "4"])

    assert cfg.patiences == (5, 10) and cfg.check_every_epochs == 0.25
    assert cfg.mlflow_uri == "http://h:1" and cfg.max_updates == 4 and cfg.seed == 3
    assert cfg.chain_key == "rel-f1__driver-dnf__cumulative__main"


@pytest.mark.parametrize("mode", ["ewc", "reservoir"])
def test_unwired_ablations_refuse_to_run(tmp_path, mode):
    """Until EWC and the reservoir are wired in, running them must not quietly
    train plain incremental fine-tuning under their names."""
    with pytest.raises(NotImplementedError, match=mode):
        run_chain(_cfg(tmp_path, mode=mode), ctx=_ctx(_cfg(tmp_path)))
