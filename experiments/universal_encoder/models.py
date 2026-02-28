from typing import Optional

import torch
import torch_geometric.nn.aggr as geo_aggr
from torch_geometric.data import HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.nn import SAGEConv, MLP
from torch_geometric.typing import NodeType, EdgeType

from torch_frame import stype, TensorFrame

from relbench.modeling.nn import HeteroTemporalEncoder, HeteroGraphSAGE

from redelex.data import TensorStatType
from redelex.nn.encoders import UniversalRowEncoder, RelativeTemporalEncoder


class TableEncoder(torch.nn.Module):
    def __init__(
        self,
        col_channels: int,
        out_channels: int,
        embedding_dim: int,
        tabular_encoder_heads: int = 4,
        tabular_encoder_layers: int = 2,
        tabular_encoder_dropout: float = 0.1,
        use_stype_emb: bool = True,
        use_name_emb: bool = True,
        use_stats_emb: bool = True,
    ):
        super().__init__()
        self.tabular_encoder = UniversalRowEncoder(
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            data_channels=col_channels // 2,
            type_channels=col_channels // 32,
            name_channels=col_channels * 3 // 8,
            stats_channels=col_channels * 3 // 32,
            encoder_heads=tabular_encoder_heads,
            encoder_layers=tabular_encoder_layers,
            encoder_dropout=tabular_encoder_dropout,
            use_stype_emb=use_stype_emb,
            use_name_emb=use_name_emb,
            use_stats_emb=use_stats_emb,
        )

    def reset_parameters(self):
        self.tabular_encoder.reset_parameters()

    def forward(self, batch: HeteroData) -> dict[str, torch.Tensor]:
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
        return x_dict


class HomogeneousGNN(torch.nn.Module):
    def __init__(
        self,
        gnn_channels: int,
        gnn_layers: int = 2,
        gnn_aggr: str = "mean",
    ):
        super().__init__()
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

        self.reset_parameters()

    def reset_parameters(self):
        self.temporal_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(
        self, x_dict: dict[str, torch.Tensor], batch: HeteroData, entity_table: NodeType
    ) -> tuple[torch.Tensor, dict, dict]:
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

        return x, node_slices, edge_slices


class HeterogeneousGNN(torch.nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        gnn_channels: int,
        gnn_layers: int = 2,
        gnn_aggr: str = "mean",
    ):
        super().__init__()
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

        self.reset_parameters()

    def reset_parameters(self):
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()

    def forward(
        self, x_dict: dict[str, torch.Tensor], batch: HeteroData, entity_table: NodeType
    ) -> dict[str, torch.Tensor]:
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

        return x_dict


class HomogeneousTaskHead(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_norm: str = "batch_norm",
    ):
        super().__init__()
        self.head = MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.head.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        node_slices: dict,
        entity_table: str,
        seed_time_size: Optional[int] = None,
    ) -> torch.Tensor:
        entity_x = x[node_slices[entity_table][0] : node_slices[entity_table][1]]

        if seed_time_size is not None:
            entity_x = entity_x[:seed_time_size]

        return self.head(entity_x)


class HeterogeneousTaskHead(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_norm: str = "batch_norm",
    ):
        super().__init__()
        self.head = MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=head_norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.head.reset_parameters()

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        entity_table: str,
        seed_time_size: Optional[int] = None,
    ) -> torch.Tensor:
        entity_x = x_dict[entity_table]

        if seed_time_size is not None:
            entity_x = entity_x[:seed_time_size]

        return self.head(entity_x)


class TaskGNNAndHead(torch.nn.Module):
    def __init__(
        self, gnn: torch.nn.Module, head: torch.nn.Module, gnn_type: str = "homogeneous"
    ):
        super().__init__()
        self.gnn = gnn
        self.head = head
        self.gnn_type = gnn_type

    def forward(
        self,
        x_dict: dict,
        batch: HeteroData,
        entity_table: str,
        seed_time_size: Optional[int] = None,
    ):
        if self.gnn_type == "homogeneous":
            x, node_slices, edge_slices = self.gnn(x_dict, batch, entity_table)
            return self.head(x, node_slices, entity_table, seed_time_size)
        else:
            x_dict = self.gnn(x_dict, batch, entity_table)
            return self.head(x_dict, entity_table, seed_time_size)
