from typing import Any

import torch
import torch_geometric.nn.aggr as geo_aggr
from torch_geometric.data import HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.nn import SAGEConv, MLP
from torch_geometric.typing import NodeType, EdgeType

from torch_frame import stype, TensorFrame
from torch_frame.data import StatType
from torch_frame.nn.models import ResNet

from relbench.modeling.nn import HeteroTemporalEncoder, HeteroGraphSAGE, HeteroEncoder

from redelex.data import TextEmbedder, TensorStatType
from redelex.nn.encoders import UniversalRowEncoder, RelativeTemporalEncoder


class HeteroSAGEModel(torch.nn.Module):
    def __init__(
        self,
        col_channels: int,
        gnn_channels: int,
        out_channels: int,
        col_names_dict: dict[NodeType, dict[stype, list[str]]],
        col_stats_dict: dict[NodeType, dict[str, dict[StatType, Any]]],
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        tabular_encoder_model: str = "resnet",
        tabular_encoder_layers: int = 2,
        gnn_layers: int = 2,
        gnn_aggr: str = "mean",
        head_norm: str = "batch_norm",
    ):
        super().__init__()

        if tabular_encoder_model == "resnet":
            torch_frame_model_cls = ResNet
        else:
            raise ValueError(f"Unknown tabular encoder model: {tabular_encoder_model}")

        self.tabular_encoder = HeteroEncoder(
            channels=gnn_channels,
            node_to_col_names_dict=col_names_dict,
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=torch_frame_model_cls,
            torch_frame_model_kwargs={
                "channels": col_channels,
                "num_layers": tabular_encoder_layers,
            },
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=node_types, channels=gnn_channels
        )
        if gnn_aggr == "attn":
            gnn_aggr = geo_aggr.SetTransformerAggregation(
                channels=gnn_channels, concat=False
            )
        self.gnn = HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=gnn_channels,
            aggr=gnn_aggr,
            num_layers=gnn_layers,
        )
        self.head = MLP(
            in_channels=gnn_channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tabular_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        x_dict: dict[str, torch.Tensor] = self.tabular_encoder(batch.tf_dict)

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


class UniversalSAGEModel(torch.nn.Module):
    def __init__(
        self,
        col_channels: int,
        gnn_channels: int,
        out_channels: int,
        text_embedder: TextEmbedder,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        tabular_encoder_heads: int = 4,
        tabular_encoder_layers: int = 2,
        tabular_encoder_dropout: float = 0.1,
        gnn_layers: int = 2,
        gnn_aggr: str = "mean",
        head_norm: str = "batch_norm",
    ):
        super().__init__()

        self.text_embedder = text_embedder

        self.tabular_encoder = UniversalRowEncoder(
            out_channels=gnn_channels,
            text_embedder=text_embedder,
            data_channels=col_channels // 2,
            type_channels=col_channels // 32,
            name_channels=col_channels * 3 // 8,
            stats_channels=col_channels * 3 // 32,
            encoder_heads=tabular_encoder_heads,
            encoder_layers=tabular_encoder_layers,
            encoder_dropout=tabular_encoder_dropout,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=node_types, channels=gnn_channels
        )
        if gnn_aggr == "attn":
            gnn_aggr = geo_aggr.SetTransformerAggregation(
                channels=gnn_channels, concat=False
            )
        self.gnn = HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=gnn_channels,
            aggr=gnn_aggr,
            num_layers=gnn_layers,
        )
        self.head = MLP(
            in_channels=gnn_channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tabular_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        tensor_stats_dict: dict[str, dict[stype, dict[TensorStatType, torch.Tensor]]] = (
            batch.tensor_stats_dict
        )
        name_embeddings_dict: dict[str, dict[str, torch.Tensor]] = (
            batch.name_embeddings_dict
        )

        tf_dict: dict[str, TensorFrame] = batch.tf_dict

        x_dict: dict[str, torch.Tensor] = {}
        for tname, tf in tf_dict.items():
            x_dict[tname] = self.tabular_encoder(
                tname,
                tf,
                tensor_stats_dict[tname],
                name_embeddings_dict[tname],
            )

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


class UniversalHomogeneousSAGEModel(torch.nn.Module):
    def __init__(
        self,
        col_channels: int,
        gnn_channels: int,
        out_channels: int,
        text_embedder: TextEmbedder,
        tabular_encoder_heads: int = 4,
        tabular_encoder_layers: int = 2,
        tabular_encoder_dropout: float = 0.1,
        gnn_layers: int = 2,
        gnn_aggr: str = "mean",
        head_norm: str = "batch_norm",
    ):
        super().__init__()

        self.text_embedder = text_embedder

        self.tabular_encoder = UniversalRowEncoder(
            out_channels=gnn_channels,
            text_embedder=text_embedder,
            data_channels=col_channels // 2,
            type_channels=col_channels // 32,
            name_channels=col_channels * 3 // 8,
            stats_channels=col_channels * 3 // 32,
            encoder_heads=tabular_encoder_heads,
            encoder_layers=tabular_encoder_layers,
            encoder_dropout=tabular_encoder_dropout,
        )

        self.temporal_encoder = RelativeTemporalEncoder(channels=gnn_channels)

        if gnn_aggr == "attn":
            gnn_aggr = geo_aggr.SetTransformerAggregation(
                channels=gnn_channels, concat=False
            )
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(gnn_layers):
            self.convs.append(
                SAGEConv((gnn_channels, gnn_channels), gnn_channels, aggr=gnn_aggr)
            )
            self.norms.append(torch.nn.LayerNorm(gnn_channels))

        self.head = MLP(
            in_channels=gnn_channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tabular_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        tensor_stats_dict: dict[str, dict[stype, dict[TensorStatType, torch.Tensor]]] = (
            batch.tensor_stats_dict
        )
        name_embeddings_dict: dict[str, dict[str, torch.Tensor]] = (
            batch.name_embeddings_dict
        )

        tf_dict: dict[str, TensorFrame] = batch.tf_dict

        x_dict: dict[str, torch.Tensor] = {}
        for tname, tf in tf_dict.items():
            x_dict[tname] = self.tabular_encoder(
                tname,
                tf,
                tensor_stats_dict[tname],
                name_embeddings_dict[tname],
            )

        if hasattr(batch[entity_table], "seed_time"):
            seed_time = batch[entity_table].seed_time
            for node_type, time in batch.time_dict.items():
                rel_time = self.temporal_encoder(seed_time, time, batch[node_type].batch)
                x_dict[node_type] = x_dict[node_type] + rel_time

        edge_index, node_slices, edge_slices = to_homogeneous_edge_index(batch)
        x_list = []
        for nt in node_slices.keys():
            x_list.append(x_dict[nt])

        x = torch.cat(x_list, dim=0)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)

        if hasattr(batch[entity_table], "seed_time"):
            return self.head(
                x[node_slices[entity_table][0] : node_slices[entity_table][1]][
                    : seed_time.size(0)
                ]
            )

        return self.head(x[node_slices[entity_table][0] : node_slices[entity_table][1]])
