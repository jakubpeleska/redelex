from typing import Any, Union

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeType


class AttachDictTransform(BaseTransform):
    def __init__(
        self,
        attach_data: Union[
            tuple[str, dict[NodeType, Any]], list[tuple[str, dict[NodeType, Any]]]
        ],
    ):
        super().__init__()
        self.attach_data = attach_data
        if not isinstance(attach_data, list):
            self.attach_data = [attach_data]
        for t in self.attach_data:
            assert isinstance(t, tuple) and len(t) == 2
            assert isinstance(t[0], str) and isinstance(t[1], dict)

    def forward(self, batch: HeteroData) -> HeteroData:
        for nt in batch.node_types:
            for name, data_dict in self.attach_data:
                batch[nt][name] = data_dict[nt]

        return batch
