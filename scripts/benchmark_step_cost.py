r"""Measure what one training step and one scored row cost on a dataset.

The rerun's budget (notebook 08) multiplies step counts from the convergence
calibration by these per-dataset costs. For one (dataset, task) this records:

* time to build the context (database, anchor table, graph, plan);
* the plan's size: updates, windows, rows, rows per update;
* seconds per training step on the largest update (cumulative mixing over all
  labelled history), split into data loading and compute, after warm-up;
* scoring throughput in rows per second;
* peak GPU memory.

Usage::

    .venv/bin/python scripts/benchmark_step_cost.py --dataset rel-stack --task user-engagement --device cuda
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval_rows", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "methodology" / "out" / "bench"))
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from experiments.continuous_learning.chain_runner import (
        ChainConfig, _subsample, build_context, new_model, training_sources,
    )
    from experiments.continuous_learning.modes import get_mode
    from experiments.continuous_learning.score_chain import predict_table
    from experiments.continuous_learning.utils import get_table_input
    from redelex.loaders import MixtureLoader, Source
    from redelex.nn.train import LightningEntityTaskWrapper
    from redelex.transforms import AttachTargetTransform

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = ChainConfig(args.dataset, args.task, "cumulative", 0, "/tmp/unused", device=args.device,
                      holdout_frac=args.holdout_frac)
    t0 = time.perf_counter()
    ctx = build_context(cfg)
    load_s = time.perf_counter() - t0
    device = ctx.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    plan = ctx.plan
    per_update = []
    for ep in plan.episodes:
        n_all = len(ctx.wrapper.trainable(ep.update_time).df)
        n_new = len(ctx.wrapper.trainable(ep.update_time, after=ep.new_after).df)
        per_update.append({"update": ep.index, "T": str(ep.update_time.date()), "labelled": n_all, "new": n_new,
                           "deploy": int((ctx.rows["window"] == ep.index + 1).sum())})

    ep = plan.episodes[-1]
    sources = training_sources(ctx, get_mode("cumulative"), ep)
    srcs = {}
    for name, (table, share) in sources.items():
        inp = get_table_input(table, ctx.task)
        srcs[name] = Source(inp.nodes[1], inp.time, inp.target, share)
    total_steps = args.steps + args.warmup
    loader = MixtureLoader(ctx.data, ctx.task.entity_table, srcs, [32, 16], 128, total_steps * 128, seed=0,
                           transform_factory=AttachTargetTransform)
    model = new_model(ctx, cfg).to(device)
    wrapper = LightningEntityTaskWrapper(model, torch.optim.Adam(model.parameters(), lr=1e-3), ctx.task, modes=("val",))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    data_t, comp_t = [], []
    it = iter(loader)
    for i in range(total_steps):
        sync()
        a = time.perf_counter()
        batch = next(it).to(device)
        sync()
        b = time.perf_counter()
        pred, target = wrapper(batch)
        loss = wrapper.loss_fn(pred.float(), target)
        wrapper.optimizer.zero_grad()
        loss.backward()
        wrapper.optimizer.step()
        sync()
        c = time.perf_counter()
        if i >= args.warmup:
            data_t.append(b - a)
            comp_t.append(c - b)

    full = ctx.wrapper.full_table
    sample = _subsample(full, args.eval_rows, "bench/eval")
    sync()
    a = time.perf_counter()
    predict_table(model, ctx.data, ctx.task, sample, [32, 16], 512, device)
    sync()
    eval_s = time.perf_counter() - a

    result = {
        "dataset": args.dataset,
        "task": args.task,
        "task_type": ctx.task.task_type.value,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu",
        "threads": torch.get_num_threads(),
        "load_seconds": round(load_s, 1),
        "updates": plan.num_updates,
        "rows": len(ctx.rows),
        "entities": int(ctx.rows["entity"].nunique()),
        "delta_w_days": ctx.protocol.delta_w.days,
        "delta_i_days": ctx.protocol.delta_i.days if ctx.protocol.delta_i is not None else None,
        "per_update": per_update,
        "train_rows_last_update": int(sum(len(s.nodes) for s in srcs.values())),
        "s_per_step": float(np.median(np.add(data_t, comp_t))),
        "data_s_per_step": float(np.median(data_t)),
        "compute_s_per_step": float(np.median(comp_t)),
        "eval_rows_per_s": len(sample.df) / eval_s if eval_s > 0 else None,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "gpu" if device.type == "cuda" else "cpu"
    path = out / f"{args.dataset}__{args.task}__{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_update"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
