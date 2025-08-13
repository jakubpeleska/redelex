from typing import Any, Optional
import torch

import torch_frame
from torch_frame.data import StatType
from torch_frame.nn.encoder import (
    StypeWiseFeatureEncoder,
    StypeEncoder,
    EmbeddingEncoder,
    LinearEncoder,
)


class LinearRowEncoder(torch.nn.Module):
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

        self.linear = torch.nn.Linear(channels * num_cols, out_channels)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, tf: torch_frame.TensorFrame) -> torch.Tensor:
        encoded = self.encoder(tf)
        x: torch.Tensor = encoded[0]

        assert len(x.shape) == 3

        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.linear(x)

        return x
