from typing import Optional

import torch
import torch_frame
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, MLP
from torch_geometric.typing import NodeType, EdgeType

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE


class HeterogeneousSAGE(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict[str, dict[str, dict]],
        gnn_channels: int,
        gnn_layers: int = 2,
        gnn_aggr: str = "sum",
        out_channels: int = 1,
        norm: str = "batch_norm",
    ):
        super().__init__()

        # `na_strategy` is not optional here. With the library default, a numerical
        # column holding ANY missing cell gets a NaN gradient (the forward pass
        # nan_to_num's its output, but the backward computes NaN * 0 = NaN), Adam
        # turns that into a NaN weight on the first step, and the column is dead
        # for the rest of the run. In a chain of updates the set of dead columns
        # grows at whichever episode first samples a missing cell, which looks
        # exactly like a model getting worse over time. MEAN maps a missing cell
        # to the column mean. This differs from the RelBench default.
        self.encoder = HeteroEncoder(
            channels=gnn_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            default_stype_encoder_cls_kwargs={
                torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
                torch_frame.numerical: (
                    torch_frame.nn.LinearEncoder,
                    {"na_strategy": torch_frame.NAStrategy.MEAN},
                ),
                torch_frame.multicategorical: (
                    torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                    {},
                ),
                torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
                torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
            },
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=data.node_types, channels=gnn_channels
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=gnn_channels,
            aggr=gnn_aggr,
            num_layers=gnn_layers,
        )
        self.head = MLP(
            in_channels=gnn_channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    @property
    def row_encoder(self):
        # An alias, not a second submodule: assigning both names registered the
        # encoder twice, so every checkpoint stored it twice.
        return self.encoder

    def reset_parameters(self):
        self.row_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> torch.Tensor:
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
