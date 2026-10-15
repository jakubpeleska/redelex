r"""Shrink-and-perturb (Ash & Adams, NeurIPS 2020) for warm-started updates.

Before training on new data, every trainable parameter becomes
``shrink * theta + perturb * theta_fresh``, where ``theta_fresh`` is a freshly
initialised copy of the same parameter. Shrinking restores some of the
plasticity a warm start loses; the perturbation re-injects the scale of a
fresh initialisation. Buffers (e.g. BatchNorm running statistics) are left
alone, and frozen parameters are not touched.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def shrink_perturb_(model: torch.nn.Module, fresh: torch.nn.Module, shrink: float, perturb: float) -> None:
    """In place: ``p <- shrink * p + perturb * p_fresh`` for every trainable parameter."""
    fresh_params = dict(fresh.named_parameters())
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        q = fresh_params.get(name)
        if q is None or q.shape != p.shape:
            raise ValueError(f"fresh model has no parameter matching {name!r}")
        p.mul_(shrink).add_(q.to(p.device, p.dtype), alpha=perturb)


__all__ = ["shrink_perturb_"]
