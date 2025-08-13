from typing import Any, Dict, Literal

import torch
from torch import Tensor

import torch_frame
from torch_frame.data.stats import StatType
from torch_frame.nn import ResNet

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP, PositionalEncoding

from redelex.nn.encoders import LinearRowEncoder


class TabularModel(torch.nn.Module):
    def __init__(
        self,
        tf: torch_frame.TensorFrame,
        col_stats: Dict[str, Dict[StatType, Any]],
        tabular_model: Literal["resnet", "linear"],
        channels: int,
        out_channels: int,
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

        tabular_cls, tabular_kwargs = get_tabular_model(tabular_model)

        self.tabular = tabular_cls(
            **tabular_kwargs,
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=tf.col_names_dict,
            stype_encoder_dict={
                torch_frame.categorical: torch_frame.nn.EmbeddingEncoder(),
                torch_frame.numerical: torch_frame.nn.LinearEncoder(),
                torch_frame.multicategorical: torch_frame.nn.MultiCategoricalEmbeddingEncoder(),
                torch_frame.embedding: torch_frame.nn.LinearEmbeddingEncoder(),
                torch_frame.timestamp: torch_frame.nn.TimestampEncoder(),
            },
        )

        self.temporal_pos = PositionalEncoding(channels)
        self.temporal_lin = torch.nn.Linear(channels, channels)

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tabular.reset_parameters()
        self.temporal_pos.reset_parameters()
        self.temporal_lin.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData) -> Tensor:
        x = self.tabular(batch.tf)

        if hasattr(batch, "seed_time"):
            if hasattr(batch, "time"):
                rel_time = batch.seed_time[batch.batch] - batch.time
                rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

                _x = self.temporal_pos(rel_time)
                _x = self.temporal_lin(_x)
                x = x + _x

            return self.head(x[: batch.seed_time.size(0)])

        return self.head(x)
