# This file includes code originally from relgt (MIT licensed):
# https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/encoders.py

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

import torch_frame
from torch_frame.nn.models import (
    ResNet,
)  # Ensure torch_frame is installed and imported correctly


class NeighborTfsEncoder(nn.Module):
    """
    Encoder for neighbor TorchFrame objects.

    Processes a batch of lists of TorchFrame objects using a two-stage encoding style,
    similar to HeteroEncoder, for a single node type context.
    """

    def __init__(
        self,
        channels: int,
        node_type_map,  # Mapping from node type to index (if needed externally)
        col_names_dict,
        col_stats_dict,
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        """
        Args:
            channels (int): Output channels for the encoder.
            node_type_map: Mapping from node type to index.
            col_names_dict (dict): Dictionary mapping column types to list of column names.
            col_stats_dict (dict): Dictionary of statistics for columns.
            torch_frame_model_cls: Class for the TorchFrame model (default: ResNet).
            torch_frame_model_kwargs (dict): Keyword arguments for the model class.
            default_stype_encoder_cls_kwargs (dict): Dictionary mapping stype to a tuple of
                                                      (encoder class, kwargs) for that stype.
        """
        super(NeighborTfsEncoder, self).__init__()

        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for nt, idx in node_type_map.items()}
        self.encoders = nn.ModuleDict()
        self.channels = channels

        # Initialize encoders for each node type using provided dictionaries
        for node_type, stype_dict in col_names_dict.items():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in stype_dict.keys()
                if stype in default_stype_encoder_cls_kwargs
            }
            self.encoders[node_type] = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=col_stats_dict[node_type],
                col_names_dict=stype_dict,
                stype_encoder_dict=stype_encoder_dict,
            )

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(self, batch_dict, neighbor_types):
        """
        Args:
            batch_dict (dict): A dictionary containing:
              - grouped_tfs[t_int]: A single concatenated TorchFrame of all neighbors
                                    for node type 't_int' in the batch.
              - grouped_indices[t_int]: The list of flat indices corresponding to
                                       each row in grouped_tfs[t_int].
              - flat_batch_idx (List[int]): The batch index 'i' for each flattened neighbor.
              - flat_nbr_idx (List[int]): The neighbor index 'j' for each flattened neighbor.
            neighbor_types (Tensor): A [B, K] tensor specifying the node type indices
                                     for each neighbor in the original (batch, neighbor) shape.

        This method performs a single-pass encoding for each node type by:
          1) Encoding the concatenated TorchFrame (big_tf) for that type in one shot.
          2) Scattering the resulting embeddings back to the flattened positions.
          3) Reassembling the final [B, K, channels] tensor using 'flat_batch_idx' and 'flat_nbr_idx'.

        Returns:
            Tensor: A [B, K, channels] tensor of encoded neighbor features, preserving
                    the original ordering of neighbors per sample.
        """
        grouped_tfs = batch_dict["grouped_tfs"]
        grouped_indices = batch_dict["grouped_indices"]
        flat_batch_idx = batch_dict["flat_batch_idx"]
        flat_nbr_idx = batch_dict["flat_nbr_idx"]

        B, K = neighbor_types.shape
        N = len(flat_batch_idx)  # total flattened neighbors
        device = neighbor_types.device

        # Pre-allocate an [N, channels] buffer
        # (Even if N==0, this works fine: shape is [0, channels].)
        encoded_flat_tensor = torch.zeros((N, self.channels), device=device)

        # 1) Encode in one shot per node type
        for t_int, big_tf in grouped_tfs.items():
            node_type_str = self.inv_node_type_map[t_int]
            encoder = self.encoders[node_type_str]

            big_tf = big_tf.to(device=device)

            for stype, tensor in big_tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    big_tf.feat_dict[stype] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            # assert torch.isfinite(big_tf.feat_dict[torch_frame.numerical]).all(), f"NaN/Inf in the raw big_tf for {node_type_str}?"

            out_t = encoder(
                big_tf
            )  # shape: [num_rows, channels] or [num_rows, 1, channels]
            if out_t.dim() == 3 and out_t.shape[1] == 1:
                out_t = out_t.squeeze(1)  # => [num_rows, channels]

            # Insert each row into encoded_flat_tensor
            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            encoded_flat_tensor[idx_tensor] = out_t

        # 2) Scatter [N, channels] -> [B, K, channels]
        output = torch.zeros((B, K, self.channels), device=device)

        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)
        output[indices_i, indices_j] = encoded_flat_tensor

        return output
