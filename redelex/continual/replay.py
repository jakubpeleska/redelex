r"""A bounded replay buffer of task rows.

The reservoir ablation keeps at most ``capacity`` past training rows and replays
them beside the new rows. It is the memory-restricted counterpart of
``replay_p50``, which may draw from the whole history. The buffer holds task-row
ids (``__row__``), so whatever it replays can be checked against the held-out
entities like any other training table.

Selection is Vitter's Algorithm R: after ``n`` rows have been offered, each one is
in the buffer with probability ``min(1, capacity / n)``, whenever it arrived.
"""

from __future__ import annotations

import numpy as np


class ReservoirBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.seen = 0
        self.row_ids = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.row_ids.size)

    def add(self, row_ids) -> None:
        """Offer rows in order; keep a uniform sample of everything offered so far."""
        ids = np.asarray(row_ids, dtype=np.int64).ravel()
        room = max(0, self.capacity - len(self))
        head, tail = ids[:room], ids[room:]
        if head.size:
            self.row_ids = np.concatenate([self.row_ids, head])
            self.seen += int(head.size)
        for rid in tail:
            self.seen += 1
            j = int(self._rng.integers(0, self.seen))
            if j < self.capacity:
                self.row_ids[j] = rid

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "row_ids": self.row_ids.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ReservoirBuffer":
        buf = cls(capacity=int(state["capacity"]), seed=int(state["seed"]))
        buf.seen = int(state["seen"])
        buf.row_ids = np.asarray(state["row_ids"], dtype=np.int64).copy()
        buf._rng.bit_generator.state = state["rng_state"]
        return buf


__all__ = ["ReservoirBuffer"]
