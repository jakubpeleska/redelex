# This file includes code originally from relgt (MIT licensed):
# https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/model.py

# Original copyright:
# MIT License

# Copyright (c) 2025 Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey, Jure Leskovec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any

import torch
import torch.nn as nn

from torch_geometric.nn import MLP

from torch_frame.data.stats import StatType

from redelex.nn.layers import RelGTLayer
from redelex.nn.encoders import (
    NeighborNodeTypeEncoder,
    NeighborHopEncoder,
    NeighborTimeEncoder,
    NeighborTfsEncoder,
    GNNPostionalEncoder,
)


class RelGT(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: dict[str, int],
        col_names_dict: dict[str, dict[str, list[str]]],
        col_stats_dict: dict[str, dict[str, dict[StatType, Any]]],
        local_num_layers: int,
        channels: int,
        out_channels: int,
        global_dim: int,
        heads: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        conv_type: str = "full",
        ablate: str = "none",
        gnn_pe_dim: int = 0,
        num_centroids: int = 4096,
        sample_node_len: int = 100,
        args: Any = None,
    ):
        super(RelGT, self).__init__()

        self.max_neighbor_hop = max_neighbor_hop
        self.node_type_map = node_type_map
        # num_node_types = len(node_type_map) + 1  # extra element for mask token
        # num_hop_types = self.max_neighbor_hop + 1  # extra element for mask token

        self.type_encoder = NeighborNodeTypeEncoder(
            embedding_dim=channels, node_type_map=self.node_type_map
        )
        self.hop_encoder = NeighborHopEncoder(
            embedding_dim=channels, max_neighbor_hop=self.max_neighbor_hop
        )
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder = NeighborTfsEncoder(
            channels=channels,
            node_type_map=self.node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
        self.pe_encoder = GNNPostionalEncoder(embedding_dim=channels, pe_dim=gnn_pe_dim)

        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_time = nn.LayerNorm(channels)
        self.layer_norm_tfs = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)

        hidden_channels = channels

        ablate_key_dict = {"type": 0, "hop": 1, "time": 2, "tfs": 3, "gnn": 4}
        self.ablate_idx = ablate_key_dict.get(ablate, None)
        channel_mult = 5 if self.ablate_idx is None else 4

        self.in_mixture = nn.Sequential(
            nn.Linear(channel_mult * channels, 2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
        )

        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        _overall_num_layers = 1
        for _ in range(_overall_num_layers):
            self.convs.append(
                RelGTLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    local_num_layers=local_num_layers,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                    sample_node_len=sample_node_len,
                )
            )
            h_times = 2 if conv_type == "full" else 1

            self.ffs.append(
                nn.Sequential(
                    nn.BatchNorm1d(hidden_channels * h_times),  # BN in
                    nn.Linear(h_times * hidden_channels, hidden_channels * 2),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Dropout(ff_dropout),
                    nn.BatchNorm1d(hidden_channels),  # BN out
                )
            )

        # supervised task head
        self.head = MLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            num_layers=2,
        )

    def reset_parameters(self):
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.tfs_encoder.reset_parameters()
        self.pe_encoder.reset_parameters()

        for layer in self.in_mixture:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            if hasattr(ff, "reset_parameters"):
                ff.reset_parameters()

        self.head.reset_parameters()

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
    ):
        neighbor_tfs = self.layer_norm_tfs(
            self.tfs_encoder(grouped_tf_dict, neighbor_types)
        )
        neighbor_types = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
        neighbor_hops = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
        neighbor_times = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
        neighbor_subgraph_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))

        cat_list = [
            neighbor_types,
            neighbor_hops,
            neighbor_times,
            neighbor_tfs,
            neighbor_subgraph_pe,
        ]
        if self.ablate_idx is not None:
            cat_list.pop(self.ablate_idx)
        x_set = torch.cat(cat_list, dim=-1)
        x_set = self.in_mixture(x_set)

        x = x_set[:, 0, :]  # select seed token representation
        for i, conv in enumerate(self.convs):
            x_set = conv(x_set, x, node_indices)
            x_set = self.ffs[i](x_set)
        x_set = self.head(x_set)

        return x_set

    def global_forward(self, x, pos_enc, node_indices):
        raise NotImplementedError
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv.global_forward(x, pos_enc, node_indices)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x
