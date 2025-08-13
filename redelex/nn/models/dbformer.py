from typing import Any, Dict, Literal

import torch
from torch import Tensor

from torch_frame.data.stats import StatType

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder

from redelex.nn.encoders import PerFeatureRowEncoder
from redelex.nn.models.rgnn import DBFormer


class DBFormerModel(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        entity_table: NodeType,
        num_layers: int,
        channels: int,
        row_encoder: Literal["linear"],
        out_channels: int,
        aggr: str,
        norm: str,
    ):
        super().__init__()

        self.entity_table = entity_table

        def get_encoder(row_encoder: str):
            if row_encoder == "linear":
                return PerFeatureRowEncoder, {
                    "channels": 128,
                    "feature_transform": "linear",
                }
            else:
                raise ValueError(f"Unknown row_encoder: {row_encoder}")

        encoder_cls, encoder_kwargs = get_encoder(row_encoder)

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
        self.gnn = DBFormer(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
            col_stats_dict=col_stats_dict,
        )

        entity_cols = len(col_stats_dict[entity_table].keys())
        self.head = MLP(
            entity_cols * channels,
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

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)

        if hasattr(batch[self.entity_table], "seed_time"):
            seed_time = batch[self.entity_table].seed_time
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time[:, None, :]

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        entity_x = x_dict[self.entity_table]
        if hasattr(batch[self.entity_table], "seed_time"):
            entity_x = entity_x[: seed_time.size(0)]

        entity_x = entity_x.view(entity_x.size(0), -1)
        return self.head(entity_x)
