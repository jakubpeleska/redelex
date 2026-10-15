r"""Update regimes for the rerun, as data: which rows each update trains on, and how.

Every regime trains on rows labelled by the update time ``T_e`` and never on
held-out entities. They differ in four ways:

* ``warm_start``: continue from this chain's own previous model, or start
  from a fresh random one. A cold regime never loads weights, whatever it is
  handed.
* ``history``: ``"all"`` labelled rows, the ``"last"`` episode only (the
  rows newly labelled since the previous update), or the last
  ``window_episodes`` episodes (``"last_k"``).
* ``new_share``: with ``history="all"``, the share of samples drawn from the
  last episode (the rest from older rows). ``None`` samples every training row
  uniformly. Jakub's ``p`` uses the same scale: p = 1 is new-only
  (``incremental``), p = 0.5 is ``replay_p50``, and his "p = 0" means uniform
  over all data (``cumulative``), not "old rows only".
* ablation-only extras: shrink-and-perturb, EWC, a bounded reservoir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModeSpec:
    name: str
    warm_start: bool
    history: str  # "all" | "last" | "last_k"
    window_episodes: Optional[int] = None
    new_share: Optional[float] = None
    shrink: Optional[float] = None
    perturb: Optional[float] = None
    ewc_lambda: Optional[float] = None
    buffer_size: Optional[int] = None
    ablation: bool = False

    def __post_init__(self):
        if self.history not in ("all", "last", "last_k"):
            raise ValueError(f"{self.name}: unknown history {self.history!r}")
        if (self.history == "last_k") != (self.window_episodes is not None):
            raise ValueError(f"{self.name}: window_episodes goes with history='last_k' only")
        if self.new_share is not None:
            if self.history != "all":
                raise ValueError(f"{self.name}: new_share needs history='all'")
            if not 0.0 < self.new_share < 1.0:
                raise ValueError(f"{self.name}: new_share must be in (0, 1)")
        if (self.shrink is None) != (self.perturb is None):
            raise ValueError(f"{self.name}: shrink and perturb go together")
        if (self.shrink is not None or self.ewc_lambda is not None) and not self.warm_start:
            raise ValueError(f"{self.name}: shrink-and-perturb and EWC need a warm start")


MAIN_ROSTER = (
    ModeSpec("scratch_all", warm_start=False, history="all"),
    ModeSpec("scratch_last", warm_start=False, history="last"),
    ModeSpec("cumulative", warm_start=True, history="all"),
    ModeSpec("replay_p50", warm_start=True, history="all", new_share=0.5),
    ModeSpec("incremental", warm_start=True, history="last"),
)

ABLATION_ROSTER = (
    ModeSpec("replay_p25", warm_start=True, history="all", new_share=0.25, ablation=True),
    ModeSpec("replay_p75", warm_start=True, history="all", new_share=0.75, ablation=True),
    ModeSpec("scratch_k2", warm_start=False, history="last_k", window_episodes=2, ablation=True),
    ModeSpec("scratch_k4", warm_start=False, history="last_k", window_episodes=4, ablation=True),
    ModeSpec("scratch_k8", warm_start=False, history="last_k", window_episodes=8, ablation=True),
    # Ash & Adams (2020) shrink 0.4 / perturb 0.1 on a warm start over all data.
    ModeSpec("shrink_perturb", warm_start=True, history="all", shrink=0.4, perturb=0.1, ablation=True),
    ModeSpec("ewc", warm_start=True, history="last", ewc_lambda=100.0, ablation=True),
    ModeSpec("reservoir", warm_start=True, history="last", new_share=None, buffer_size=10_000, ablation=True),
)

MODES = {m.name: m for m in MAIN_ROSTER + ABLATION_ROSTER}


def get_mode(name: str) -> ModeSpec:
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(f"unknown mode {name!r}; known: {sorted(MODES)}") from None


__all__ = ["ABLATION_ROSTER", "MAIN_ROSTER", "MODES", "ModeSpec", "get_mode"]
