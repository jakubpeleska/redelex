r"""Episode protocol for keeping a model current on a growing relational database.

Vocabulary, used the same way everywhere in the continual-learning code:

* **Anchor** ``t``: the timestamp of one task row. Its label aggregates events in
  ``(t, t + Δw]``, so it is only known at ``t + Δw``.
* **Update times** ``T_1 < T_2 < ... < T_E``, spaced by ``ΔI``. At ``T_e`` the
  model is updated with every row whose label is complete: ``t <= T_e - Δw``
  (the *label delay*).
* **Windows** ``1 .. E+1``. Window ``j`` holds the anchors in
  ``[T_{j-1}, T_j)``, with ``T_0 = -inf`` (the initial history) and
  ``T_{E+1} = horizon_end``. The model updated at ``T_e`` is deployed on window
  ``e + 1``: that is the next-window score.
* **Newly labelled rows** of update ``e``: anchors with
  ``T_{e-1} - Δw < t <= T_e - Δw``. Every anchor is "new" exactly once. On an
  anchor grid of stride ``Δw`` whose boundaries sit on the grid, this is exactly
  window ``e``, i.e. "the whole last episode".

The performance matrix uses one convention, pinned by the tests:
``R[e, j]`` is the model updated at ``T_e`` scored on window ``j``, both
1-based. The next-window score is ``R[e, e + 1]``.

The held-out entities are a fixed fraction of the task's entities chosen by a
salted CRC32 of the entity id, so the choice is identical for every task on the
same entity table, every configuration, seed and episode, and every process.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

HOLDOUT_BUCKETS = 10_000


def labelled_cutoff(update_time: pd.Timestamp, delta_w: pd.Timedelta) -> pd.Timestamp:
    """Latest anchor whose label is complete at ``update_time``."""
    return update_time - delta_w


def trainable_mask(
    anchor_times: pd.Series | np.ndarray | Sequence,
    update_time: pd.Timestamp,
    delta_w: pd.Timedelta,
    after: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """Rows labelled by ``update_time``: ``t <= update_time - delta_w``.

    With ``after``, only rows strictly later than it (used for "newly labelled
    since the previous update").
    """
    times = pd.to_datetime(pd.Series(anchor_times)).to_numpy()
    mask = times <= np.datetime64(labelled_cutoff(update_time, delta_w))
    if after is not None:
        mask &= times > np.datetime64(after)
    return mask


def grid_update_times(
    reference: pd.Timestamp,
    delta_i: Optional[pd.Timedelta],
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Update times on a calendar grid through ``reference``.

    Returns every ``reference + k * delta_i`` (integer ``k``) with
    ``first < T <= last``, ascending. ``delta_i=None`` means "never update
    again": the single update time ``reference`` (the pure-staleness baseline).
    """
    if delta_i is None:
        return [pd.Timestamp(reference)]
    if delta_i <= pd.Timedelta(0):
        raise ValueError(f"delta_i must be positive, got {delta_i}")
    k_lo = math.floor((first - reference) / delta_i) + 1
    k_hi = math.floor((last - reference) / delta_i)
    return [pd.Timestamp(reference + k * delta_i) for k in range(k_lo, k_hi + 1)]


@dataclass(frozen=True)
class Episode:
    """One update. ``index`` is 1-based; all spans are closed/open as stated."""

    index: int
    update_time: pd.Timestamp
    prev_update_time: Optional[pd.Timestamp]
    next_update_time: pd.Timestamp
    delta_w: pd.Timedelta

    @property
    def labelled_until(self) -> pd.Timestamp:
        """Anchors ``t <= labelled_until`` are trainable at this update."""
        return labelled_cutoff(self.update_time, self.delta_w)

    @property
    def new_after(self) -> Optional[pd.Timestamp]:
        """Newly labelled rows have ``new_after < t <= labelled_until``.

        ``None`` at the first update: everything labelled so far is new.
        """
        if self.prev_update_time is None:
            return None
        return labelled_cutoff(self.prev_update_time, self.delta_w)

    @property
    def deploy_window(self) -> int:
        """Window index (1-based) this update's model is deployed on."""
        return self.index + 1

    @property
    def deploy_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Anchors ``deploy_span[0] <= t < deploy_span[1]`` are the next window."""
        return self.update_time, self.next_update_time


@dataclass(frozen=True)
class EpisodePlan:
    """Update times plus the window grid they induce."""

    update_times: tuple[pd.Timestamp, ...]
    horizon_end: pd.Timestamp
    delta_w: pd.Timedelta

    def __post_init__(self):
        times = list(self.update_times)
        if not times:
            raise ValueError("an episode plan needs at least one update time")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("update times must be strictly increasing")
        if self.horizon_end <= times[-1]:
            raise ValueError("horizon_end must lie after the last update time")

    @property
    def num_updates(self) -> int:
        return len(self.update_times)

    @property
    def num_windows(self) -> int:
        return len(self.update_times) + 1

    @property
    def episodes(self) -> list[Episode]:
        times = list(self.update_times)
        nexts = times[1:] + [self.horizon_end]
        prevs: list[Optional[pd.Timestamp]] = [None] + times[:-1]
        return [
            Episode(i + 1, t, p, n, self.delta_w)
            for i, (t, p, n) in enumerate(zip(times, prevs, nexts))
        ]

    def window_bounds(self, window: int) -> tuple[Optional[pd.Timestamp], pd.Timestamp]:
        """``[lo, hi)`` of a 1-based window; ``lo`` is ``None`` for window 1."""
        if not 1 <= window <= self.num_windows:
            raise IndexError(f"window {window} outside 1..{self.num_windows}")
        edges: list[Optional[pd.Timestamp]] = [None, *self.update_times, self.horizon_end]
        return edges[window - 1], edges[window]

    def assign_windows(self, anchor_times) -> np.ndarray:
        """1-based window index of every anchor; 0 for anchors at or past the horizon."""
        times = pd.to_datetime(pd.Series(anchor_times)).to_numpy()
        edges = np.array(list(self.update_times), dtype="datetime64[ns]")
        window = np.searchsorted(edges, times, side="right") + 1
        window[times >= np.datetime64(self.horizon_end)] = 0
        return window


def entity_holdout_mask(
    entity_ids: Iterable, frac: float, salt: str
) -> np.ndarray:
    """``True`` for held-out entities: ``crc32(f"{salt}/{id}") % 10000 < frac * 10000``.

    The salt should name the entity table (e.g. ``"rel-f1/drivers/v1"``) so that
    every task on that table holds out the same entities.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"holdout fraction must be in [0, 1), got {frac}")
    threshold = int(round(frac * HOLDOUT_BUCKETS))
    ids = list(entity_ids)
    buckets = np.fromiter(
        (zlib.crc32(f"{salt}/{i}".encode()) % HOLDOUT_BUCKETS for i in ids),
        dtype=np.int64,
        count=len(ids),
    )
    return buckets < threshold


@dataclass(frozen=True)
class Protocol:
    """Everything that defines how a chain is updated and scored. Hashable."""

    dataset: str
    task: str
    delta_w: pd.Timedelta
    delta_i: Optional[pd.Timedelta]
    anchor_stride: pd.Timedelta
    anchor_phase: pd.Timedelta = pd.Timedelta(0)
    holdout_frac: float = 0.1
    holdout_salt: str = ""
    extra: dict = field(default_factory=dict, hash=False, compare=False)

    def as_dict(self) -> dict:
        out = {}
        for key in (
            "dataset",
            "task",
            "delta_w",
            "delta_i",
            "anchor_stride",
            "anchor_phase",
            "holdout_frac",
            "holdout_salt",
        ):
            value = getattr(self, key)
            out[key] = None if value is None else str(value)
        out["extra"] = {k: str(v) for k, v in sorted(self.extra.items())}
        return out

    def hash(self) -> str:
        """Short stable digest; a resumed chain must match it exactly."""
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


__all__ = [
    "Episode",
    "EpisodePlan",
    "Protocol",
    "entity_holdout_mask",
    "grid_update_times",
    "labelled_cutoff",
    "trainable_mask",
]
