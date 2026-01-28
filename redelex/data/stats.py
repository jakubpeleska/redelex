from typing import Any, Optional

from enum import Enum

import torch

import torch_frame
from torch_frame import stype
from torch_frame.data import StatType

from .text_embedder import TextEmbedder


class TensorStatType(Enum):
    # Numerical stats
    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"

    # Categorical and MultiCategorical stats
    CARDINALITY = "cardinality"
    VALUE_EMBEDDINGS = "value_embeddings"

    # Timestamp stats
    EARLIEST_DATE = "earliest"
    LATEST_DATE = "latest"
    MIN_YEAR = "min_year"
    MAX_YEAR = "max_year"


def make_tensor_stats_dict(
    col_stats_dict: dict[str, dict[StatType, Any]],
    col_names_dict: dict[stype, list[str]],
    text_embedder: TextEmbedder,
    device: Optional[torch.device] = None,
) -> dict[stype, dict[TensorStatType, torch.Tensor]]:
    stype_stats: dict[stype, dict[TensorStatType, torch.Tensor]] = {}

    for st, cols in col_names_dict.items():
        if st == torch_frame.stype.numerical:
            stats_list = [
                TensorStatType.MEAN,
                TensorStatType.STD,
                TensorStatType.MIN,
                TensorStatType.MAX,
                TensorStatType.MEDIAN,
                TensorStatType.Q1,
                TensorStatType.Q3,
            ]
        elif (
            st == torch_frame.stype.categorical or st == torch_frame.stype.multicategorical
        ):
            stats_list = [TensorStatType.CARDINALITY, TensorStatType.VALUE_EMBEDDINGS]
        elif st == torch_frame.stype.timestamp:
            stats_list = [
                TensorStatType.EARLIEST_DATE,
                TensorStatType.LATEST_DATE,
                TensorStatType.MIN_YEAR,
                TensorStatType.MAX_YEAR,
            ]
        else:
            continue

        stype_stats[st] = {}

        for stat in stats_list:
            if stat == TensorStatType.MEAN:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.MEAN] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.STD:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.STD] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.MIN:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.QUANTILES][0] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.MAX:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.QUANTILES][4] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.MEDIAN:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.QUANTILES][2] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.Q1:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.QUANTILES][1] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.Q3:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.QUANTILES][3] for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.CARDINALITY:
                stat_type = StatType.COUNT
                if st == torch_frame.stype.multicategorical:
                    stat_type = StatType.MULTI_COUNT
                stype_stats[st][stat] = torch.tensor(
                    [len(col_stats_dict[col][stat_type][0]) for col in cols],
                    device=device,
                )
            elif stat == TensorStatType.VALUE_EMBEDDINGS:
                stat_type = StatType.COUNT
                if st == torch_frame.stype.multicategorical:
                    stat_type = StatType.MULTI_COUNT
                embeddings_list = [
                    text_embedder([str(cat) for cat in col_stats_dict[col][stat_type][0]])
                    for col in cols
                ]
                stype_stats[st][stat] = (
                    torch.nn.utils.rnn.pad_sequence(  # TODO: change to nested tensor
                        embeddings_list, batch_first=True, padding_value=0.0
                    )
                )
            elif stat == TensorStatType.EARLIEST_DATE:
                stype_stats[st][stat] = torch.stack(
                    [col_stats_dict[col][StatType.OLDEST_TIME] for col in cols],
                ).to(device)
            elif stat == TensorStatType.LATEST_DATE:
                stype_stats[st][stat] = torch.stack(
                    [col_stats_dict[col][StatType.NEWEST_TIME] for col in cols]
                ).to(device)
            elif stat == TensorStatType.MIN_YEAR:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.YEAR_RANGE][0] for col in cols]
                ).to(device)
            elif stat == TensorStatType.MAX_YEAR:
                stype_stats[st][stat] = torch.tensor(
                    [col_stats_dict[col][StatType.YEAR_RANGE][1] for col in cols]
                ).to(device)
    return stype_stats
