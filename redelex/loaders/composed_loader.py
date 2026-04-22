from typing import Literal, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader


class ComposedLoader:
    def __init__(
        self,
        loaders: dict[str, NodeLoader],
        mode: Literal["minimum", "full", "rnd_uni", "rnd_weighted"] = "minimum",
        weights: Optional[list[float]] = None,
    ):
        self.loaders = loaders
        self.task_names = list(self.loaders.keys())
        self.loaders_len = [len(self.loaders[tn]) for tn in self.task_names]
        self.mode = mode

        if self.mode in ["minimum", "rnd_uni", "rnd_weighted"]:
            self.total_len = min(self.loaders_len) * len(self.loaders)
        elif self.mode == "full":
            self.total_len = sum(self.loaders_len)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.mode == "rnd_weighted":
            assert weights is not None, "Weights must be provided for rnd_weighted mode"
            assert len(weights) == len(self.loaders), (
                "Weights length must match number of loaders"
            )
            self.weights = weights

    def __iter__(self):
        self.idx = 0
        if self.mode == "minimum":
            N = min(self.loaders_len)
            L = len(self.loaders)
            self.rnd_loader_idx = (
                torch.stack([torch.randperm(L) for _ in range(N)]).flatten().long().tolist()
            )
        elif self.mode in ["full", "rnd_uni"]:
            loader_indices = torch.arange(len(self.loaders_len))
            rnd_cat = torch.repeat_interleave(
                loader_indices, repeats=torch.tensor(self.loaders_len)
            )
            rnd_idx = torch.randperm(rnd_cat.shape[0])
            self.rnd_loader_idx = rnd_cat[rnd_idx].long().tolist()
        elif self.mode == "rnd_weighted":
            rnd_cat = torch.multinomial(
                self.weights, num_samples=self.total_len, replacement=True
            )
            self.rnd_loader_idx = rnd_cat.long().tolist()
        self.loader_iter = [iter(self.loaders[tn]) for tn in self.task_names]
        return self

    def __next__(self) -> HeteroData:
        if self.idx >= len(self):
            raise StopIteration
        _loader_idx = self.rnd_loader_idx[self.idx]
        self.idx += 1
        return next(self.loader_iter[_loader_idx])

    def __len__(self):
        return self.total_len
