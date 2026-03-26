import copy
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
from torch_frame import TensorFrame, stype
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor, StatType
from torch_frame.typing import TensorData
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeType


class ResampleCorruptor(BaseTransform):
    def __init__(
        self,
        data: HeteroData,
        corrupt_prob: float = 0.5,
        distribution: Literal["empirical", "uniform"] = "uniform",
    ):
        self.corrupt_prob = corrupt_prob
        self.distribution = distribution
        self.corruptors = {
            tname: TFCorruptor(tf, p=corrupt_prob, distribution=distribution)
            for tname, tf in data.collect("tf").items()
        }

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Corrupt the data by resampling features with a given probability.
        Args:
            data (HeteroData): The input heterogeneous data.
        Returns:
            HeteroData: The corrupted data with additional 'cor_tf' and 'cor_mask' attributes.
        """
        return self.corrupt_data(data)

    def corrupt_data(
        self, data: HeteroData
    ) -> Tuple[HeteroData, Dict[NodeType, TensorFrame]]:
        for node_type, tf in data.collect("tf").items():
            cor_tf, mask = self.corruptors[node_type](tf)
            data[node_type]["cor_tf"] = cor_tf
            data[node_type]["cor_col_mask"] = mask

        return data


class TFCorruptor:
    """
    A class to resample and corrupt features of a TensorFrame.
    """

    def __init__(
        self,
        tf: TensorFrame,
        p: float = 0.5,
        distribution: Literal["empirical", "uniform"] = "empirical",
    ):
        """
        Initialize the TFCorruptor with a TensorFrame and corruption probability.
        Args:
            tf (TensorFrame): The TensorFrame to corrupt.
            p (float): The probability of corruption.
            distribution (Literal["empirical", "uniform"]): The type of distribution to use for resampling.
        """
        self.p = p
        self.col_samplers: Dict[str, Callable] = {}

        # Create samplers for categorical features
        for col in tf._col_to_stype_idx:
            x = self.get_tf_col(tf, col)
            if x.ndim == 1:
                x = x[~x.isnan()]
            self.col_samplers[col] = self.get_categorical_sampler(
                x,
                empirical=(distribution == "empirical"),
            )

    def __call__(self, tf: TensorFrame) -> TensorFrame:
        return self.corrupt_tf(tf, self.col_samplers, p=self.p)

    @classmethod
    def get_tf_col(cls, tf: TensorFrame, col: str) -> torch.Tensor:
        x: TensorData = tf.get_col_feat(col)
        if isinstance(x, torch.Tensor):
            x = x.squeeze(1)
        if isinstance(x, MultiEmbeddingTensor):
            x = x.values.squeeze(1)
        if isinstance(x, MultiNestedTensor):
            x = x.values.squeeze(1)
        return x

    @classmethod
    def get_categorical_sampler(
        cls, x: torch.Tensor, empirical=True, max_values: int = 10000
    ) -> Callable[[torch.Size], torch.Tensor]:
        """
        Get the empirical marginal distribution of a categorical feature.
        Args:
            x (torch.Tensor): The categorical feature tensor.
            empirical (bool): If True, use empirical distribution; otherwise, use a uniform distribution.
        Returns:
            torch.distributions.Categorical: The empirical marginal distribution.
        """
        u_values: torch.Tensor
        u_values, counts = x.unique(sorted=True, return_counts=True, dim=0)

        top_idx = torch.argsort(counts, descending=True)
        u_values = u_values[top_idx[:max_values]]
        counts = counts[top_idx[:max_values]]

        if empirical:
            # Get the empirical marginal distribution
            marginal_prob = counts.float() / counts.sum()
        else:
            # Use a uniform distribution
            marginal_prob = torch.ones(len(u_values), dtype=torch.float) / len(u_values)
        distribution = torch.distributions.Categorical(marginal_prob)

        def sample(size: torch.Size) -> torch.Tensor:
            return u_values[distribution.sample(size)]

        return sample

    @classmethod
    def corrupt_tf(
        cls,
        tf: TensorFrame,
        col_samplers: Dict[str, Callable],
        p: float = 0.5,
    ) -> Tuple[TensorFrame, Dict[str, torch.Tensor]]:
        """
        Corrupts the features of a TensorFrame with probability p using empirical marginal distribution.
        Args:
            tf (TensorFrame): The TensorFrame to corrupt.
            p (float): The probability of corruption.
        Returns:
            Tuple[TensorFrame, Dict[str, torch.Tensor]]: The corrupted TensorFrame and a mask indicating which features were corrupted.
        """

        _tf = copy.deepcopy(tf)

        mask: dict[str, torch.Tensor] = {}
        for col, sampler in col_samplers.items():
            x = cls.get_tf_col(_tf, col)

            if x.ndim < 1:
                continue

            col_mask = torch.rand(x.shape[0], device=x.device) < p
            mask[col] = col_mask

            n_samples = col_mask.sum()
            if n_samples == 0:
                continue

            samples = sampler((n_samples,))

            if n_samples == 0:
                continue
            # if n_samples == 1:
            #     print(f"Corrupting {col} with {n_samples} samples")
            #     print(col_mask, x, samples)

            x[col_mask, ...] = samples
            s, idx = _tf._col_to_stype_idx[col]
            if isinstance(_tf.feat_dict[s][:, idx], MultiEmbeddingTensor):
                # If the feature is a MultiEmbeddingTensor, we need to handle it differently
                _tf.feat_dict[s][:, idx].values = x
            else:
                _tf.feat_dict[s][:, idx] = x

        return _tf, mask


def rescale_tf(tf: TensorFrame, stats: Optional[Dict[str, Dict[StatType, Any]]] = None):
    """
    Rescale the numerical features of a TensorFrame to the range [0, 1].

    Args:
        tf (TensorFrame): The TensorFrame to rescale.
    Returns:
        TensorFrame: The rescaled TensorFrame.
    """
    if stype.numerical not in tf.stypes or len(tf) == 0:
        return tf

    _tf = copy.deepcopy(tf)

    for col in _tf.col_names_dict[stype.numerical]:
        x: torch.Tensor = _tf.get_col_feat(col).squeeze()
        if stats is not None and col in stats:
            # Use the provided statistics to rescale
            x_min = stats[col][StatType.QUANTILES][0]
            x_max = stats[col][StatType.QUANTILES][-1]
        else:
            # Compute the min and max values of the feature
            x_min, x_max = x.nanquantile(q=torch.tensor([0.0, 1.0])).tolist()
        # Re-scale the numerical feature to [0, 1]
        x = (x - x_min) / (x_max - x_min)
        _tf.feat_dict[stype.numerical][:, _tf._col_to_stype_idx[col][1]] = x

    return _tf
