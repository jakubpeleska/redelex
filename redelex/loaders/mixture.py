r"""One temporal neighbour loader over several row sources, mixed at exact shares.

:class:`~redelex.loaders.ComposedLoader` mixes whole *batches* from separate
loaders. Its batch counts follow the loader lengths ("rnd_uni") or are rounded
from the requested weights ("weighted"); either way a small source with a
partial last batch contributes fewer *samples* than its share. Here every
sample is drawn independently: row ``r`` of source ``s`` is drawn with
probability ``share_s / n_s``, so the expected sample share of ``s`` is exactly
``share_s``. One mechanism covers every update regime: new-only (one source),
uniform over all history (one source), and any mix of new and older rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


@dataclass
class Source:
    """Rows of one source: entity node ids, anchor unix times and targets."""

    nodes: torch.Tensor
    time: torch.Tensor
    target: torch.Tensor
    share: float

    def __len__(self) -> int:
        return int(self.nodes.numel())


def mixture_weights(sizes: Mapping[str, int], shares: Mapping[str, float]) -> np.ndarray:
    """Per-row sampling weight ``share_s / n_s``, rows ordered source by source.

    Shares are normalised over the non-empty sources, so an empty source's share
    is given to the others rather than silently shrinking the epoch.
    """
    live = [s for s in sizes if sizes[s] > 0 and shares[s] > 0]
    if not live:
        raise ValueError("a mixture needs at least one non-empty source with a positive share")
    total = sum(shares[s] for s in live)
    parts = [
        np.full(sizes[s], (shares[s] / total) / sizes[s] if s in live else 0.0)
        for s in sizes
    ]
    return np.concatenate(parts)


class MixtureLoader:
    """A :class:`NeighborLoader` fed by a :class:`WeightedRandomSampler`.

    ``source_of_row[i]`` names the source of input row ``i``; with
    ``batch[entity_table].input_id`` it attributes every sampled seed to its
    source, which is how the cost meter counts new versus older samples.
    """

    def __init__(
        self,
        data: HeteroData,
        entity_table: str,
        sources: Mapping[str, Source],
        num_neighbors: list[int],
        batch_size: int,
        num_samples: int,
        seed: int,
        transform_factory,
        temporal_strategy: str = "uniform",
        num_workers: int = 0,
    ):
        names = list(sources)
        self.names = names
        self.sizes = {s: len(sources[s]) for s in names}
        self.shares = {s: float(sources[s].share) for s in names}
        weights = torch.as_tensor(mixture_weights(self.sizes, self.shares), dtype=torch.double)

        self.source_of_row = np.concatenate(
            [np.full(self.sizes[s], i, dtype=np.int16) for i, s in enumerate(names)]
        )
        nodes = torch.cat([sources[s].nodes for s in names])
        time = torch.cat([sources[s].time for s in names])
        target = torch.cat([sources[s].target for s in names])
        self.num_rows = int(nodes.numel())

        self.generator = torch.Generator().manual_seed(int(seed))
        self.sampler = WeightedRandomSampler(
            weights, num_samples=int(num_samples), replacement=True, generator=self.generator
        )
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=(entity_table, nodes),
            input_time=time,
            transform=transform_factory(entity_table, target),
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            sampler=self.sampler,
            num_workers=num_workers,
        )
        self.entity_table = entity_table

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def source_counts(self, batch) -> dict[str, int]:
        """How many seeds of ``batch`` came from each source."""
        ids = batch[self.entity_table].input_id.cpu().numpy()
        counts = np.bincount(self.source_of_row[ids], minlength=len(self.names))
        return {s: int(c) for s, c in zip(self.names, counts)}


def epoch_steps(num_rows: int, batch_size: int, fraction: float = 1.0, minimum: int = 1) -> int:
    """Optimizer steps in ``fraction`` of one pass over ``num_rows`` rows."""
    return max(minimum, int(np.ceil(fraction * num_rows / batch_size)))


__all__ = ["MixtureLoader", "Source", "epoch_steps", "mixture_weights"]
