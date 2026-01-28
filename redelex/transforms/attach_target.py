import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class AttachTargetTransform(BaseTransform):
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: torch.Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch
