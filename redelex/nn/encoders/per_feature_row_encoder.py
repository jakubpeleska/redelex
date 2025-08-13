from typing import Any, Optional

import math

import torch
from torch.nn import init

import torch_frame
from torch_frame.data import StatType
from torch_frame.nn.encoder import (
    StypeWiseFeatureEncoder,
    StypeEncoder,
    EmbeddingEncoder,
    LinearEncoder,
)


class PerFeatureRowEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: Optional[dict[torch_frame.stype, StypeEncoder]] = None,
    ):
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                torch_frame.stype.categorical: EmbeddingEncoder(),
                torch_frame.stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        num_cols = sum([len(col_names) for col_names in col_names_dict.values()])

        self.weight = torch.nn.Parameter(torch.empty(num_cols, channels, out_channels))
        self.bias = torch.nn.Parameter(torch.empty(num_cols, out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()

        # Taken from torch.nn.Linear
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, tf: torch_frame.TensorFrame) -> torch.Tensor:
        encoded = self.encoder(tf)
        x: torch.Tensor = encoded[0]

        x = torch.einsum("bcd, cdo -> bco", x, self.weight) + self.bias

        return x
