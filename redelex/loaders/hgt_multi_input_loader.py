from typing import Callable, Optional, Tuple, List, Union

import torch

from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.typing import NodeType


class HGTMultiInputLoader:
    def __init__(
        self,
        data: HeteroData,
        num_samples: List[int],
        input_nodes: Union[List[NodeType], List[Tuple[NodeType, torch.Tensor]]],
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.data = data
        self.current_index = 0
        self.input_nodes = (
            [node[0] for node in input_nodes]
            if isinstance(input_nodes[0], tuple)
            else input_nodes
        )
        self.loaders = {
            input_node: HGTLoader(
                data,
                num_samples=num_samples,
                input_nodes=input_node,
                transform=transform,
                **kwargs,
            )
            for input_node in input_nodes
        }
        self.loaders_len = [len(loader) for loader in self.loaders.values()]
        self.total_len = sum(self.loaders_len)

    def __iter__(self):
        self.idx = 0
        _rnd_cat = torch.repeat_interleave(torch.tensor(self.loaders_len))
        _rnd_idx = torch.randperm(_rnd_cat.shape[0])
        self.rnd_loader_idx = _rnd_cat[_rnd_idx].long().tolist()
        self.loader_iter = [iter(self.loaders[e]) for e in self.input_nodes]
        return self

    def __next__(self) -> Tuple[NodeType, HeteroData]:
        if self.idx >= len(self):
            raise StopIteration
        _loader_idx = self.rnd_loader_idx[self.idx]
        self.idx += 1
        return next(self.loader_iter[_loader_idx])

    def __len__(self):
        return self.total_len
