from typing import Any, Union

from torch_geometric.typing import NodeType
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class AttachDictTransform(BaseTransform):
    def __init__(
        self,
        attach_data: Union[
            tuple[str, dict[NodeType, Any]], list[tuple[str, dict[NodeType, Any]]]
        ],
    ):
        super().__init__()
        self.attach_data = attach_data
        if type(attach_data) is not list:
            self.attach_data = [attach_data]
        for t in self.attach_data:
            assert isinstance(t, tuple) and len(t) == 2
            assert type(t[0]) is str and type(t[1]) is dict

    def __call__(self, batch: HeteroData) -> HeteroData:
        for nt in batch.node_types:
            for name, data_dict in self.attach_data:
                batch[nt][name] = data_dict[nt]

        return batch
