from typing import Optional, Union

from enum import Enum

import torch
import torch.nn.functional as F

import torch_frame
from torch_frame import stype
from torch_frame.data import MultiNestedTensor, MultiEmbeddingTensor
from torch_frame.data.mapper import TimestampTensorMapper
from torch_frame.nn.encoding import CyclicEncoding, PositionalEncoding

from redelex.data import TextEmbedder, TensorStatType


class TokenType(Enum):
    Numerical = 0
    Categorical = 1
    MultiCategorical = 2
    Timestamp = 3
    Embedding = 4


class TokenTypeEncoder(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(TokenType), channels)
        self.reset_parameters()

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_types)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()


class UniversalStypeEncoder(torch.nn.Module):
    def __init__(self, stats_channels: int):
        super().__init__()
        self.stats_channels = stats_channels

    def forward(
        self,
        feat: Union[torch.Tensor, MultiNestedTensor, MultiEmbeddingTensor],
        stats: Optional[dict[TensorStatType, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stats is None:
            stats_emb = self.empty_stats(feat.size(1), device=feat.device)
        else:
            stats_emb = self.encode_stats(stats)
        feat = self.fill_na(feat, stats)
        return self.encode_features(feat, stats), stats_emb

    def fill_na(
        self, feat: torch.Tensor, stats: Optional[dict[TensorStatType, torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def empty_stats(
        self, num_cols: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        # stats: dict[TensorStatType, Tensor[num_cols]]
        return torch.zeros((num_cols, self.stats_channels), device=device)

    def encode_features(
        self, feat: torch.Tensor, stats: Optional[dict[TensorStatType, torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def encode_stats(self, stats: dict[TensorStatType, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class UniversalLinearEncoder(UniversalStypeEncoder):
    stats_list = [
        TensorStatType.MEAN,
        TensorStatType.STD,
        TensorStatType.MIN,
        TensorStatType.MAX,
        TensorStatType.MEDIAN,
        TensorStatType.Q1,
        TensorStatType.Q3,
    ]

    def __init__(self, data_channels: int, stats_channels: int):
        self.data_channels = data_channels
        self.stats_channels = stats_channels

        super().__init__(stats_channels=stats_channels)

        self.feat_weight = torch.nn.Parameter(torch.empty(self.data_channels))
        self.feat_bias = torch.nn.Parameter(torch.empty(self.data_channels))

        self.stats_linear = torch.nn.Linear(
            len(self.stats_list), self.stats_channels, bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.feat_weight, std=0.01)
        torch.nn.init.zeros_(self.feat_bias)
        self.stats_linear.reset_parameters()

    def fill_na(
        self, feat: torch.Tensor, stats: Optional[dict[TensorStatType, torch.Tensor]] = None
    ) -> torch.Tensor:
        # feat: [batch_size, num_cols]
        na_mask = torch.isnan(feat)
        if stats is None or stats.get(TensorStatType.MEAN) is None:
            fill_value = feat.nanmean(dim=0)
        else:
            fill_value = stats[TensorStatType.MEAN]
        # fill_value: [num_cols]
        feat = torch.where(na_mask, fill_value, feat)
        return feat

    def encode_features(
        self,
        feat: torch.Tensor,
        stats: Optional[dict[TensorStatType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # feat: [batch_size, num_cols]
        # mean, std: [num_cols]
        feat = (feat - stats[TensorStatType.MEAN]) / stats[TensorStatType.STD]

        x_lin = torch.einsum(
            "ij,k->ijk", feat, self.feat_weight
        )  # [batch_size, num_cols] + [channels] -> [batch_size, num_cols, channels]
        x = x_lin + self.feat_bias
        return x

    def encode_stats(self, stats: dict[TensorStatType, torch.Tensor]) -> torch.Tensor:
        # stats: dict[TensorStatType, Tensor[num_cols]]
        stats_tensor = torch.stack(
            [stats[stat] for stat in self.stats_list], dim=-1
        )  # [num_cols, num_stats]
        return self.stats_linear(stats_tensor)  # [num_cols, stats_channels]


class UniversalCategoricalEncoder(UniversalStypeEncoder):
    stats_list = [TensorStatType.CARDINALITY]

    def __init__(
        self,
        data_channels: int,
        stats_channels: int,
        text_embedder: TextEmbedder,
        num_categories: int = 1000,
    ):
        self.data_channels = data_channels
        self.stats_channels = stats_channels
        self.text_embedder = text_embedder
        self.num_categories = num_categories

        super().__init__(stats_channels=stats_channels)

        self.feat_embedding = torch.nn.Embedding(
            self.num_categories + 1, self.data_channels
        )  # +1 for NA

        self.stats_linear = torch.nn.Linear(
            len(self.stats_list), self.stats_channels, bias=True
        )

        self.text_transform = torch.nn.Linear(
            self.text_embedder.embedding_dim, self.data_channels, bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feat_embedding.reset_parameters()
        self.stats_linear.reset_parameters()
        self.text_transform.reset_parameters()

    def fill_na(
        self,
        feat: Union[torch.Tensor, MultiNestedTensor],
        stats: Optional[dict[TensorStatType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # feat: [batch_size, num_cols]
        if isinstance(feat, MultiNestedTensor):
            na_mask = torch.isnan(feat.values)
            feat.values = torch.where(na_mask, -1, feat.values)
        else:
            na_mask = torch.isnan(feat)
            feat = torch.where(na_mask, -1, feat)
        return feat

    def encode_features(
        self,
        feat: Union[torch.Tensor, MultiNestedTensor],
        stats: Optional[dict[TensorStatType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if stats is None or stats.get(TensorStatType.VALUE_EMBEDDINGS) is None:
            value_emb = None
        else:
            value_emb = stats[
                TensorStatType.VALUE_EMBEDDINGS
            ]  # [num_cols, max_cardinality, embed_dim]
            value_emb = torch.cat(
                [
                    torch.zeros(
                        value_emb.size(0), 1, value_emb.size(2), device=feat.device
                    ),
                    value_emb,
                ],
                dim=1,
            )  # [num_cols, max_cardinality + 1, embed_dim]

        if isinstance(feat, MultiNestedTensor):
            indices = feat.values + 1  # +1 for NA, feat.values: [total_elems]

            x: torch.Tensor = self.feat_embedding(indices)  # [total_elems, data_channels]

            x = F.embedding_bag(
                input=torch.arange(x.size(0)),
                weight=x,
                offsets=feat.offset[:-1],
                mode="sum",
            )

            x = x.view(feat.num_rows, feat.num_cols, self.data_channels)

            if value_emb is None:
                return x

            flat_value_emb = value_emb.view(
                -1, self.text_embedder.embedding_dim
            )  # [num_cols * max_cardinality, embed_dim]

            bag_shifts = torch.arange(feat.num_cols) * value_emb.size(1)  # [num_cols]

            lengths = feat.offset.diff()  # [num_rows * num_cols]
            col_lengths = lengths.view(feat.num_rows, feat.num_cols).sum(
                dim=0
            )  # [num_cols]

            indices_shifts = torch.repeat_interleave(
                bag_shifts, col_lengths
            )  # [total_elems]

            adjusted_indices = indices + indices_shifts

            x_val_emb = F.embedding_bag(
                adjusted_indices,
                flat_value_emb,
                offsets=feat.offset[:-1],
                mode="sum",
            )  # [batch_size * num_cols, embed_dim]

            x_val_emb = x_val_emb.view(
                feat.num_rows, feat.num_cols, self.text_embedder.embedding_dim
            )
        else:
            # feat: [batch_size, num_cols]
            indices = feat + 1  # +1 for NA

            x = self.feat_embedding(indices)  # [batch_size, num_cols, data_channels]

            if value_emb is None:
                return x

            col_indices = torch.arange(value_emb.size(0)).unsqueeze(0)  # [1, num_cols]

            x_val_emb = value_emb[col_indices, indices]  # [batch_size, num_cols, embed_dim]

        x_val_emb = self.text_transform(x_val_emb)  # [batch_size, num_cols, data_channels]

        return x + x_val_emb

    def encode_stats(self, stats: dict[TensorStatType, torch.Tensor]) -> torch.Tensor:
        # stats: dict[TensorStatType, Tensor[num_cols]]
        stats_tensor = torch.stack(
            [stats[stat] for stat in self.stats_list], dim=-1
        ).float()  # [num_cols, num_stats]
        return self.stats_linear(stats_tensor)  # [num_cols, stats_channels]


# TODO: taken from torch_frame
class UniversalTimestampEncoder(UniversalStypeEncoder):
    stats_list = [TensorStatType.EARLIEST_DATE, TensorStatType.LATEST_DATE]

    def __init__(self, data_channels: int, stats_channels: int, encoding_channels: int = 8):
        self.data_channels = data_channels
        self.stats_channels = stats_channels
        self.encoding_channels = encoding_channels

        super().__init__(stats_channels=stats_channels)

        # Ensure that the first element is year.
        assert TimestampTensorMapper.TIME_TO_INDEX["YEAR"] == 0

        # Init normalization constant
        max_values = TimestampTensorMapper.CYCLIC_VALUES_NORMALIZATION_CONSTANT
        self.register_buffer("max_values", max_values)

        # Init positional/cyclic encoding
        self.positional_encoding = PositionalEncoding(self.encoding_channels)
        self.cyclic_encoding = CyclicEncoding(self.encoding_channels)

        # Init linear function
        self.feat_weight = torch.nn.Parameter(
            torch.empty(
                len(TimestampTensorMapper.TIME_TO_INDEX),
                self.encoding_channels,
                self.data_channels,
            )
        )
        self.feat_bias = torch.nn.Parameter(torch.empty(self.data_channels))

        self.stats_transform = torch.nn.Linear(
            len(self.stats_list) * len(TimestampTensorMapper.TIME_TO_INDEX),
            self.stats_channels,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.feat_weight, std=0.01)
        torch.nn.init.zeros_(self.feat_bias)

    def fill_na(
        self, feat: torch.Tensor, stats: Optional[dict[TensorStatType, torch.Tensor]] = None
    ) -> torch.Tensor:
        na_mask = torch.isnan(feat)
        # Fill NA with earliest date
        feat = torch.where(
            na_mask,
            stats[TensorStatType.EARLIEST_DATE],
            feat,
        )
        return feat

    def encode_features(
        self, feat: torch.Tensor, stats: dict[TensorStatType, torch.Tensor]
    ) -> torch.Tensor:
        feat = feat.to(torch.float32)
        # [batch_size, num_cols, 1] - [1, num_cols, 1]
        feat_year = feat[..., :1] - stats[TensorStatType.MIN_YEAR].view(1, -1, 1)
        # [batch_size, num_cols, num_rest] / [1, 1, num_rest]
        feat_rest = feat[..., 1:] / self.max_values.view(1, 1, -1)
        # [batch_size, num_cols, num_time_feats, out_size]
        x = torch.cat(
            [self.positional_encoding(feat_year), self.cyclic_encoding(feat_rest)], dim=2
        )
        # [batch_size, num_cols, num_time_feats, out_size] *
        # [num_time_feats, out_size, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x_lin = torch.einsum("ijkl,klm->ijm", x, self.feat_weight)
        # [batch_size, num_cols, out_channels] + [num_cols, out_channels]
        x = x_lin + self.feat_bias
        return x

    def encode_stats(self, stats: dict[TensorStatType, torch.Tensor]) -> torch.Tensor:
        # stats: dict[TensorStatType, Tensor[num_cols]]
        stats_tensor = torch.concat(
            [stats[stat].to(torch.float32) for stat in self.stats_list], dim=-1
        )  # [num_cols, num_stats, ]
        return self.stats_transform(stats_tensor)  # [num_cols, stats_channels]


class UniversalEmbeddingEncoder(UniversalStypeEncoder):
    stats_list = []

    def __init__(
        self, text_embedder: TextEmbedder, data_channels: int, stats_channels: int
    ):
        self.text_embedder = text_embedder
        self.data_channels = data_channels
        self.stats_channels = stats_channels

        super().__init__(stats_channels=stats_channels)

        self.text_transform = torch.nn.Linear(
            self.text_embedder.embedding_dim, self.data_channels, bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.text_transform.reset_parameters()

    def fill_na(
        self, feat: MultiEmbeddingTensor, stats: dict[TensorStatType, torch.Tensor]
    ) -> torch.Tensor:
        return feat

    def encode_features(
        self, feat: MultiEmbeddingTensor, stats: dict[TensorStatType, torch.Tensor]
    ) -> torch.Tensor:
        # feat: MultiEmbeddingTensor [num_rows, num_cols, embed_dim]
        feat_vals = feat.values.view(feat.num_rows, feat.num_cols, -1)
        x: torch.Tensor = self.text_transform(feat_vals)  # [total_elems, data_channels]
        x = x.view(feat.num_rows, feat.num_cols, self.data_channels)
        return x


class RelativeTemporalEncoder(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.encoder = PositionalEncoding(channels)
        self.transform = torch.nn.Linear(channels, channels)

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(
        self, seed_time: torch.Tensor, time: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        rel_time = seed_time[batch] - time
        rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

        x = self.encoder(rel_time)
        x = self.transform(x)

        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ff_channels: int = 512,
        dropout: float = 0.1,
        activation: Union[str, torch.nn.Module] = "relu",
    ):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(channels, num_heads, batch_first=True)
        if activation == "relu":
            self.activation_cls = torch.nn.ReLU
        elif activation == "gelu":
            self.activation_cls = torch.nn.GELU

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(channels, ff_channels),
            self.activation_cls(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_channels, channels),
        )

        self.norm1 = torch.nn.LayerNorm(channels)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src: [Batch, Num_Features, d_model]
        next_x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout1(next_x)
        x = self.norm1(x)
        next_x = self.ffn(x)
        x = x + self.dropout2(next_x)
        x = self.norm2(x)
        return x


class TransformerAggregator(torch.nn.Module):
    def __init__(
        self,
        encoder_channels: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_channels: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.transformer = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(encoder_channels, num_heads, ffn_channels, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attention_scorer = torch.nn.Sequential(
            torch.nn.Linear(encoder_channels, encoder_channels // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(encoder_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [Batch, Num_Features, d_model]

        for layer in self.transformer:
            x = layer(x, mask)

        # Prepare mask for broadcasting if it exists
        if mask is not None:
            # Fill padding with -inf for Max and 0 for Sum
            x_mask_max = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            x_mask_sum = x.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            x_mask_max = x
            x_mask_sum = x
        # Max Pooling
        out_max = torch.max(x_mask_max, dim=1)[0]

        # Sum Pooling
        out_sum = torch.sum(x_mask_sum, dim=1)

        # Attentive Pooling
        attn_scores: torch.Tensor = self.attention_scorer(x)
        # Handle Masking for Softmax
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        # This acts as the "Gate" to suppress noise
        attn_weights = F.softmax(attn_scores, dim=1)
        # Weighted Sum
        out_attn = torch.sum(attn_weights * x, dim=1)

        return {"sum": out_sum, "max": out_max, "attn": out_attn, "attn_w": attn_weights}


class UniversalRowEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        text_embedder: TextEmbedder,
        type_channels: int = 16,
        name_channels: int = 256,
        data_channels: int = 192,
        stats_channels: int = 48,
        encoder_heads: int = 4,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
    ):
        self.type_channels = type_channels
        self.name_channels = name_channels
        self.data_channels = data_channels
        self.stats_channels = stats_channels
        self.encoder_channels = (
            self.type_channels
            + self.name_channels
            + self.data_channels
            + self.stats_channels
        )

        self.out_channels = out_channels

        self.text_embedder = text_embedder

        self.attention_heads = encoder_heads
        self.attention_layers = encoder_layers
        self.dropout = encoder_dropout

        super().__init__()

        self.name_transform = torch.nn.Linear(
            text_embedder.embedding_dim, self.name_channels, bias=True
        )

        self.token_type_encoder = TokenTypeEncoder(channels=self.type_channels)

        self.cat_encoder = UniversalCategoricalEncoder(
            data_channels=self.data_channels,
            stats_channels=self.stats_channels,
            text_embedder=self.text_embedder,
            num_categories=1000,
        )

        self.num_encoder = UniversalLinearEncoder(
            data_channels=self.data_channels, stats_channels=self.stats_channels
        )

        self.timestamp_encoder = UniversalTimestampEncoder(
            data_channels=self.data_channels, stats_channels=self.stats_channels
        )

        self.text_encoder = UniversalEmbeddingEncoder(
            text_embedder=self.text_embedder,
            data_channels=self.data_channels,
            stats_channels=self.stats_channels,
        )

        self.stype_encoders: dict[stype, UniversalStypeEncoder] = {
            stype.numerical: self.num_encoder,
            stype.categorical: self.cat_encoder,
            stype.timestamp: self.timestamp_encoder,
            stype.embedding: self.text_encoder,
            stype.multicategorical: self.cat_encoder,
        }

        self.transformer = TransformerAggregator(
            encoder_channels=self.encoder_channels,
            num_heads=self.attention_heads,
            num_layers=self.attention_layers,
            ffn_channels=self.encoder_channels // 2,
            dropout=self.dropout,
        )

        self.out_transform = torch.nn.Linear(
            self.encoder_channels * 3, self.out_channels, bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.token_type_encoder.reset_parameters()
        self.name_transform.reset_parameters()
        for encoder in self.stype_encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tname: str,
        tf: torch_frame.TensorFrame,
        stype_stats: dict[stype, dict[TensorStatType, torch.Tensor]],
        name_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tname_emb = name_embeddings[tname]  # [text_channels]

        if tf.num_cols == 0 or tf.num_rows == 0:
            # Return zero tensor if there are no columns or rows
            return torch.zeros((tf.num_rows, self.out_channels), device=tf.device)

        cols_emb = []
        for st, cols in tf.col_names_dict.items():
            st_emb = self.token_type_encoder(
                torch.tensor([STYPE_TOKEN_TYPE_MAP[st].value], device=tf.device)
            )  # [type_channels]
            st_emb = st_emb.expand(len(cols), -1)  # [num_cols, type_channels]

            col_names_emb = torch.stack(
                [name_embeddings[col] + tname_emb for col in cols], dim=0
            )  # [num_cols, embed_dim]
            col_names_emb = self.name_transform(col_names_emb)  # [num_cols, text_channels]

            encoder = self.stype_encoders[st]
            data_emb, stats_emb = encoder(tf.feat_dict[st], stype_stats.get(st, None))
            # feat_emb: [batch_size, num_cols, data_channels]
            # stats_emb: [num_cols, stats_channels]
            meta_emb = torch.concat(
                [st_emb, col_names_emb, stats_emb], dim=-1
            )  # [num_cols, meta_channels]
            meta_emb = meta_emb.unsqueeze(0).expand(
                data_emb.size(0), -1, -1
            )  # [batch_size, num_cols, meta_channels]

            col_emb = torch.concat(
                [meta_emb, data_emb], dim=-1
            )  # [batch_size, num_cols, total_channels]
            cols_emb.append(col_emb)
        cols_emb = torch.concat(cols_emb, dim=1)
        emb_dict = self.transformer(cols_emb)  # dict with 'sum', 'max', 'attn', 'attn_w'
        x = torch.concat(
            [emb_dict["sum"], emb_dict["max"], emb_dict["attn"]], dim=-1
        )  # [batch_size, num_channels * 3]

        x = self.out_transform(x)  # [batch_size, out_channels]
        return x


STYPE_TOKEN_TYPE_MAP = {
    stype.numerical: TokenType.Numerical,
    stype.categorical: TokenType.Categorical,
    stype.multicategorical: TokenType.MultiCategorical,
    stype.text_embedded: TokenType.Embedding,
    stype.image_embedded: TokenType.Embedding,
    stype.embedding: TokenType.Embedding,
    stype.timestamp: TokenType.Timestamp,
}
