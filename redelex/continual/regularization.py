r"""Regularisation-based continual learning: EWC and friends.

These methods add a quadratic penalty that anchors parameters to the values they
held after previous episodes, weighted by how much each parameter mattered.
Nothing here trains -- :func:`fisher_diagonal` measures importance after an
episode finishes, and :class:`ParameterAnchor` turns that measurement into a loss
term for the next episode.

The anchor is state that must survive an episode boundary, exactly as the model
weights already do. :meth:`ParameterAnchor.state_dict` exists for that.

Elastic Weight Consolidation (Kirkpatrick et al., 2017) uses the diagonal of the
Fisher information as the importance weight. Online EWC (Schwarz et al., 2018)
keeps a single running anchor instead of one per episode, which is what
:meth:`ParameterAnchor.consolidate` implements -- the per-episode variant would
grow linearly in the number of episodes, and these chains run to 52.
"""

from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn

__all__ = ["fisher_diagonal", "ParameterAnchor"]


def _trainable(model: nn.Module) -> Iterable:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


@torch.enable_grad()
def fisher_diagonal(
    model: nn.Module,
    batches: Iterable,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    forward_fn: Callable[[nn.Module, object], torch.Tensor],
    target_fn: Callable[[object], torch.Tensor],
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    r"""Empirical diagonal Fisher information, one entry per trainable parameter.

    Estimated as the mean squared gradient of the loss over ``batches``. This is
    the "empirical" Fisher: it uses the observed targets rather than sampling
    from the model's predictive distribution. It is the standard choice in the
    EWC literature and costs one backward pass per batch.

    Args:
        model: Network to measure. Left in its original training mode.
        batches: Iterable of batches, typically the episode's train loader.
        loss_fn: Called as ``loss_fn(pred, target)``.
        forward_fn: Called as ``forward_fn(model, batch)`` -> predictions.
        target_fn: Called as ``target_fn(batch)`` -> targets.
        max_batches: Stop after this many batches. Fisher estimates converge
            quickly, and these loaders can be very long.

    Returns:
        Mapping of parameter name to a non-negative tensor of its shape. Empty
        if no batch produced a gradient.
    """
    fisher = {name: torch.zeros_like(p) for name, p in _trainable(model)}
    if not fisher:
        return fisher

    was_training = model.training
    model.eval()  # freeze dropout/batch-norm so importance is not noise
    counted = 0
    try:
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            pred = forward_fn(model, batch)
            target = target_fn(batch)
            loss = loss_fn(pred.float(), target)
            loss.backward()
            for name, param in _trainable(model):
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if not torch.isfinite(grad).all():
                    # A single non-finite gradient would poison this parameter's
                    # importance for the rest of the chain, and the penalty it
                    # feeds is never inspected. Skip the contribution loudly
                    # rather than anchoring to NaN.
                    raise RuntimeError(
                        f"non-finite gradient for {name!r} while estimating the "
                        f"Fisher; the anchor would be poisoned for every "
                        f"subsequent episode"
                    )
                fisher[name] += grad**2
            counted += 1
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    if counted == 0:
        return {name: torch.zeros_like(p) for name, p in _trainable(model)}
    return {name: value / counted for name, value in fisher.items()}


class ParameterAnchor:
    r"""Quadratic pull toward consolidated parameter values.

    The penalty is ``0.5 * lambda * sum_i F_i (theta_i - theta*_i) ** 2``, where
    ``theta*`` are the anchored values and ``F`` their importances.

    Args:
        lam: Penalty strength. ``0`` disables the term.
        gamma: Decay applied to the existing Fisher on each
            :meth:`consolidate`, as in online EWC. ``1.0`` accumulates without
            decay; smaller values forget older episodes' importances.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(2, 1)
        >>> anchor = ParameterAnchor(lam=1.0)
        >>> anchor.penalty(model)          # nothing anchored yet
        tensor(0.)
    """

    def __init__(self, lam: float = 1.0, gamma: float = 1.0):
        if lam < 0:
            raise ValueError(f"`lam` must be non-negative, got {lam}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"`gamma` must be in [0, 1], got {gamma}")
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.means: Dict[str, torch.Tensor] = {}
        self.episodes = 0

    def __len__(self) -> int:
        return len(self.fisher)

    def consolidate(self, model: nn.Module, fisher: Dict[str, torch.Tensor]) -> None:
        """Fold one episode's importances and current weights into the anchor."""
        for name, param in _trainable(model):
            if name not in fisher:
                continue
            contribution = fisher[name].detach().clone()
            if name in self.fisher:
                # An anchor restored from disk holds CPU tensors while a freshly
                # computed Fisher follows the model, so the two sides of this sum
                # routinely disagree about device. Align on the contribution rather
                # than assuming either side.
                previous = self.fisher[name].to(contribution.device)
                self.fisher[name] = self.gamma * previous + contribution
            else:
                self.fisher[name] = contribution
            # Anchor to where the model ended this episode.
            self.means[name] = param.detach().clone()
        self.episodes += 1

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Penalty term to add to the task loss."""
        device = next(model.parameters()).device
        if self.lam == 0 or not self.fisher:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        for name, param in _trainable(model):
            if name not in self.fisher:
                continue
            fisher = self.fisher[name].to(device)
            mean = self.means[name].to(device)
            if fisher.shape != param.shape:
                continue  # architecture changed; that parameter is not anchored
            total = total + (fisher * (param - mean) ** 2).sum()
        return 0.5 * self.lam * total

    def state_dict(self) -> dict:
        return {
            "lam": self.lam,
            "gamma": self.gamma,
            "episodes": self.episodes,
            "fisher": {k: v.cpu() for k, v in self.fisher.items()},
            "means": {k: v.cpu() for k, v in self.means.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.lam = float(state["lam"])
        self.gamma = float(state["gamma"])
        self.episodes = int(state["episodes"])
        self.fisher = {k: v.clone() for k, v in state["fisher"].items()}
        self.means = {k: v.clone() for k, v in state["means"].items()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ParameterAnchor":
        anchor = cls(lam=float(state["lam"]), gamma=float(state["gamma"]))
        anchor.load_state_dict(state)
        return anchor
