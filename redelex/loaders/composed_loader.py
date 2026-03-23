from typing import Literal

import torch

from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader


class ComposedLoader:
    def __init__(
        self, loaders: dict[str, NodeLoader], mode: Literal["minimum", "full"] = "minimum"
    ):
        self.loaders = loaders
        self.task_names = list(self.loaders.keys())
        self.loaders_len = [len(self.loaders[tn]) for tn in self.task_names]
        self.mode = mode

        if self.mode == "minimum":
            self.total_len = min(self.loaders_len) * len(self.loaders)
        elif self.mode == "full":
            self.total_len = sum(self.loaders_len)
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
        elif self.mode == "full":
            loader_indices = torch.arange(len(self.loaders_len))
            rnd_cat = torch.repeat_interleave(
                loader_indices, repeats=torch.tensor(self.loaders_len)
            )
            rnd_idx = torch.randperm(rnd_cat.shape[0])
            self.rnd_loader_idx = rnd_cat[rnd_idx].long().tolist()
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
