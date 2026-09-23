from typing import Literal, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader


class ComposedLoader:
    r"""Iterates over several node loaders as a single loader.

    Modes:

    * ``minimum``: every loader contributes ``min(len(loader))`` batches, in a
      round-robin order shuffled within each round.
    * ``full``: every loader is exhausted; the order is shuffled.
    * ``rnd_uni``: like ``full``, kept for backwards compatibility.

    An epoch is ``__len__`` batches long, which is ``sum`` of the loader lengths
    for ``full`` and ``min(len) * n_loaders`` otherwise.
    """

    def __init__(
        self,
        loaders: dict[str, NodeLoader],
        mode: Literal["minimum", "full", "rnd_uni", "weighted"] = "minimum",
        weights: Optional[dict[str, float]] = None,
    ):
        self.loaders = loaders
        self.names = list(self.loaders.keys())
        self.loaders_len = [len(self.loaders[tn]) for tn in self.names]
        self.mode = mode
        self.weights = weights

        if self.mode in ["minimum", "rnd_uni"]:
            self.total_len = min(self.loaders_len) * len(self.loaders)
        elif self.mode == "full":
            self.total_len = sum(self.loaders_len)
        elif self.mode == "weighted":
            if weights is None:
                raise ValueError("mode='weighted' requires `weights`")
            missing = set(self.names) - set(weights)
            if missing:
                raise ValueError(f"`weights` is missing entries for {sorted(missing)}")
            total_w = sum(weights[n] for n in self.names)
            if total_w <= 0:
                raise ValueError("`weights` must sum to a positive value")
            self._fracs = [weights[n] / total_w for n in self.names]
            # Longest epoch that respects the ratio without exhausting a loader.
            self.total_len = min(
                int(n / f)
                for n, f in zip(self.loaders_len, self._fracs, strict=True)
                if f > 0
            )
            self._counts = [int(round(f * self.total_len)) for f in self._fracs]
            self._counts = [
                min(c, n) for c, n in zip(self._counts, self.loaders_len, strict=True)
            ]
            self.total_len = sum(self._counts)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __iter__(self):
        self.idx = 0
        if self.mode == "minimum":
            N = min(self.loaders_len)
            L = len(self.loaders)
            self.rnd_loader_idx = (
                torch.stack([torch.randperm(L) for _ in range(N)]).flatten().long().tolist()
            )
        elif self.mode == "weighted":
            rnd_cat = torch.repeat_interleave(
                torch.arange(len(self.loaders_len)), repeats=torch.tensor(self._counts)
            )
            rnd_idx = torch.randperm(rnd_cat.shape[0])
            self.rnd_loader_idx = rnd_cat[rnd_idx].long().tolist()
        elif self.mode in ["full", "rnd_uni"]:
            loader_indices = torch.arange(len(self.loaders_len))
            rnd_cat = torch.repeat_interleave(
                loader_indices, repeats=torch.tensor(self.loaders_len)
            )
            rnd_idx = torch.randperm(rnd_cat.shape[0])
            self.rnd_loader_idx = rnd_cat[rnd_idx].long().tolist()
        self.loader_iter = [iter(self.loaders[tn]) for tn in self.names]
        return self

    def __next__(self) -> HeteroData:
        if self.idx >= len(self):
            raise StopIteration
        _loader_idx = self.rnd_loader_idx[self.idx]
        self.idx += 1
        return next(self.loader_iter[_loader_idx])

    def __len__(self):
        return self.total_len
