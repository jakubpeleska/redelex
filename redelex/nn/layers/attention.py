from typing import List, Optional, Union

import torch

from torch_geometric.nn import MessagePassing, Aggregation


class CrossAttentionConv(MessagePassing):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        per_column_embedding: bool = True,
    ):
        super().__init__(aggr=aggr, node_dim=-3 if per_column_embedding else -2)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def reset_parameters(self):
        return self.attn._reset_parameters()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, query=x, key=x, value=x)

    def message(self, query_i, key_j, value_j):
        x, _ = self.attn(query_i, key_j, value_j)
        return x


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def reset_parameters(self):
        return self.attn._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out
