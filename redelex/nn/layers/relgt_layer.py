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

import torch

from .relgt_global import GlobalRelGTModule
from .relgt_local import LocalRelGTModule


class RelGTLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        local_num_layers,
        global_dim,
        num_nodes,
        heads=1,
        concat=True,
        ff_dropout=0.0,
        attn_dropout=0.0,
        edge_dim=None,
        conv_type="local",
        num_centroids=None,
        sample_node_len=100,
        **kwargs,
    ):
        super(RelGTLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len

        if self.conv_type != "global":
            self.local_module = LocalRelGTModule(
                seq_len=self.sample_node_len,
                input_dim=in_channels,
                n_layers=local_num_layers,
                num_heads=heads,
                hidden_dim=out_channels,
                dropout_rate=ff_dropout,
                attention_dropout_rate=attn_dropout,
            )
            self.layer_norm_local = torch.nn.LayerNorm(out_channels)

        if self.conv_type != "local":
            self.global_module = GlobalRelGTModule(
                in_channels=in_channels,
                out_channels=out_channels,
                global_dim=global_dim,
                num_nodes=num_nodes,
                heads=heads,
                attn_dropout=attn_dropout,
                num_centroids=num_centroids,
            )
            self.layer_norm_global = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # Reinitialize global attention layers
        if self.conv_type != "local":
            self.global_module.reset_parameters()
            self.layer_norm_global.reset_parameters()

        # Reinitialize LocalModule
        if self.conv_type != "global":
            self.local_module.reset_parameters()
            self.layer_norm_local.reset_parameters()

    def forward(self, x_set, x, node_indices):
        if self.conv_type == "local":
            out = self.local_module(x_set, pretrain_token=False)
            out = self.layer_norm_local(out)

        elif self.conv_type == "global":
            out = self.global_module(x, node_indices)
            out = self.layer_norm_global(out)

        elif self.conv_type == "full":
            out_local = self.local_module(x_set, pretrain_token=False)
            out_global = self.global_module(x, node_indices)
            out_local = self.layer_norm_local(out_local)
            out_global = self.layer_norm_global(out_global)
            out = torch.cat([out_local, out_global], dim=1)

        else:
            raise NotImplementedError

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"local_num_layers={self.local_num_layers})"
        )
