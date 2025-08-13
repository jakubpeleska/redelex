from typing import Any, Dict, Literal

import torch
from torch import Tensor

from torch_frame.data.stats import StatType
from torch_frame.nn import ResNet

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

from redelex.nn.encoders import LinearRowEncoder


class SAGEModel(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        tabular_model: Literal["resnet", "linear"],
        out_channels: int,
        aggr: str,
        norm: str,
    ):
        super().__init__()

        def get_tabular_model(row_encoder: str):
            if row_encoder == "resnet":
                return ResNet, {
                    "channels": 128,
                    "num_layers": 4,
                }
            elif row_encoder == "linear":
                return LinearRowEncoder, {"channels": 128}
            else:
                raise ValueError(f"Unknown row_encoder: {row_encoder}")

        encoder_cls, encoder_kwargs = get_tabular_model(tabular_model)

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=encoder_cls,
            torch_frame_model_kwargs=encoder_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)

        if hasattr(batch[entity_table], "seed_time"):
            seed_time = batch[entity_table].seed_time
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        if hasattr(batch[entity_table], "seed_time"):
            return self.head(x_dict[entity_table][: seed_time.size(0)])

        return self.head(x_dict[entity_table])
