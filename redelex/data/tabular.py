from typing import Optional, Any

import pandas as pd

import torch

from torch_frame import stype
from torch_frame.data import Dataset, TensorFrame, StatType
from torch_frame.config import TextEmbedderConfig

from .text_embedder import TextEmbedder


def make_tensor_frame(
    df: pd.DataFrame,
    col_to_stype: dict[str, stype],
    text_embedder: Optional[TextEmbedder] = None,
    cache_path: Optional[str] = None,
) -> tuple[TensorFrame, dict[str, dict[StatType, Any]]]:
    dataset = Dataset(
        df=df,
        col_to_stype=col_to_stype,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=text_embedder,
            batch_size=256,
        ),
    ).materialize(device=torch.device("cpu"), path=cache_path)

    return dataset.tensor_frame, dataset.col_stats
