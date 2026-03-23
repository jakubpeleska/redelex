from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from torch_geometric.nn import HeteroConv, HeteroDictLinear
from torch_geometric.typing import NodeType, EdgeType

from torch_frame.data.stats import StatType

from redelex.nn.layers import CrossAttentionConv, SelfAttention


class DBFormer(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0,
        with_norm: bool = True,
        with_residuals: bool = True,
        with_output_transform: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.with_norm = with_norm
        self.with_residuals = with_residuals

        self.attn = torch.nn.ModuleList()
        for _ in range(num_layers):
            attn_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                attn_dict[node_type] = SelfAttention(
                    channels, num_heads=num_heads, dropout=dropout
                )
            self.attn.append(attn_dict)

        self.attn_norm = None
        if with_norm:
            self.attn_norm = torch.nn.ModuleList()
            for _ in range(num_layers):
                norm_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    num_cols = len(col_stats_dict[node_type].keys())
                    norm_dict[node_type] = torch.nn.LayerNorm([num_cols, channels])
                self.attn_norm.append(norm_dict)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: CrossAttentionConv(
                        channels, num_heads=num_heads, dropout=dropout, aggr=aggr
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.conv_norm = None
        if with_norm:
            self.conv_norm = torch.nn.ModuleList()
            for _ in range(num_layers):
                norm_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    num_cols = len(col_stats_dict[node_type].keys())
                    norm_dict[node_type] = torch.nn.LayerNorm([num_cols, channels])
                self.conv_norm.append(norm_dict)

        self.with_output_transform = with_output_transform
        if with_output_transform:
            self._preout_channels = {
                node_type: channels * len(col_stats_dict[node_type].keys())
                for node_type in node_types
            }
            self.output_transform = HeteroDictLinear(
                in_channels=self._preout_channels,
                out_channels=channels,
                types=node_types,
                bias=True,
            )

    def reset_parameters(self):
        for attn_dict in self.attn:
            for attn in attn_dict.values():
                attn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        if self.with_norm:
            for norm_dict in self.attn_norm + self.conv_norm:
                for norm in norm_dict.values():
                    norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for i in range(self.num_layers):
            x_dict_next = {}
            # Apply self-attention
            for key in x_dict.keys():
                x_dict_next[key] = self.attn[i][key](x_dict[key])
                if self.with_norm:
                    x = x_dict_next[key]
                    if self.with_residuals:
                        # Optionally apply residuals
                        x += x_dict[key]
                    # Apply normalization
                    x_dict_next[key] = self.attn_norm[i][key](x)
            # Update x_dict
            x_dict = x_dict_next
            # Apply cross-attention
            x_dict_next: Dict[str, Tensor] = self.convs[i](x_dict, edge_index_dict)
            if self.with_norm:
                for key in x_dict.keys():
                    x = x_dict_next[key]
                    if self.with_residuals:
                        # Optionally apply residuals
                        x += x_dict[key]
                    # Apply normalization
                    x_dict_next[key] = self.conv_norm[i][key](x)
            # Update x_dict
            x_dict = x_dict_next

        if self.with_output_transform:
            x_dict = {
                node_type: x.view(x.size(0), self._preout_channels[node_type])
                for node_type, x in x_dict.items()
            }
            x_dict = self.output_transform(x_dict)

        return x_dict
