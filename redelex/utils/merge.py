from typing import Literal, Optional

from collections import defaultdict
import copy

import torch

from torch_frame import stype, TensorFrame
from torch_frame.data import MultiEmbeddingTensor
from torch_frame.typing import TensorData


def merge_tf(
    left_tf: TensorFrame,
    right_tf: TensorFrame,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    left_prefix: Optional[str] = None,
    right_prefix: Optional[str] = None,
    how: Literal["left"] = "left",
) -> TensorFrame:
    """Merge two tensor frames."""

    left_col_names_dict = defaultdict(list)
    left_col_names_dict.update(
        {
            st: [f"{left_prefix}{c}" for c in left_tf.col_names_dict[st]]
            for st in left_tf.stypes
        }
        if left_prefix is not None
        else copy.deepcopy(left_tf.col_names_dict)
    )

    right_col_names_dict = defaultdict(list)
    right_col_names_dict.update(
        {
            st: [f"{right_prefix}{c}" for c in right_tf.col_names_dict[st]]
            for st in right_tf.stypes
        }
        if right_prefix is not None
        else copy.deepcopy(right_tf.col_names_dict)
    )

    left_cols = []
    for cols in left_col_names_dict.values():
        left_cols.extend(cols)

    right_cols = []
    for cols in right_col_names_dict.values():
        right_cols.extend(cols)

    if len(set(left_cols).intersection(set(right_cols))) > 0:
        raise ValueError(f"Column names overlap. Left: {left_cols} Right: {right_cols}")

    col_names_dict = {
        st: left_col_names_dict[st] + right_col_names_dict[st]
        for st in left_tf.stypes + right_tf.stypes
    }

    if how != "left":
        raise NotImplementedError("Only left join is supported.")

    num_rows = left_tf.num_rows
    feat_dict = copy.deepcopy(left_tf.feat_dict)
    for st in right_tf.stypes:
        new_data = _init_stype_features(
            st,
            len(right_col_names_dict[st]),
            num_rows,
            sparse_data=right_tf.feat_dict[st][right_idx],
            idx=left_idx,
        )

        st_data = feat_dict.get(st, None)
        feat_dict[st] = (
            new_data if st_data is None else _concat_features(st, [st_data, new_data])
        )
    return TensorFrame(
        feat_dict=feat_dict, col_names_dict=col_names_dict, y=left_tf.y, num_rows=num_rows
    )


def _init_stype_features(
    s: stype, num_cols: int, num_rows: int, sparse_data: TensorData, idx: torch.Tensor
) -> TensorData:
    data: TensorData = None
    if s == stype.numerical:
        data = torch.full((num_rows, num_cols), torch.nan)
        data[idx] = sparse_data
    elif s == stype.categorical:
        data = torch.full((num_rows, num_cols), -1)
        data[idx] = sparse_data
    elif s == stype.timestamp:
        data = torch.full((num_rows, num_cols, 7), -1)
        data[idx] = sparse_data

    elif s.use_multi_embedding_tensor:
        values = torch.full((num_rows, sparse_data.values.shape[1]), -1.0)
        values[idx] = sparse_data.values
        data = MultiEmbeddingTensor(
            num_rows=num_rows,
            num_cols=num_cols,
            values=values,
            offset=sparse_data.offset,
        )
    else:
        raise ValueError(f"Unknown stype: {s}")

    return data


def _concat_features(s: stype, data_list: list[TensorData]) -> TensorData:
    if s == stype.numerical or s == stype.categorical or s == stype.timestamp:
        return torch.cat(data_list, dim=1)
    elif s.use_multi_embedding_tensor:
        return MultiEmbeddingTensor.cat(data_list, dim=1)
    else:
        raise ValueError(f"Unknown stype: {s}")


__all__ = ["merge_tf"]
