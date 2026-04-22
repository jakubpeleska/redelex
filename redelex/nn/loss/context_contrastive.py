import math
from typing import Dict, List

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import conv
from torch_geometric.typing import EdgeType, NodeType


class ContextContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        temperature: float = 0.1,
        max_negatives: int = 255,
    ):
        super().__init__()
        self.channels = channels

        self.linear_dict = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(channels, channels, bias=True)
                for node_type in node_types
            }
        )

        self.mean_pooling = conv.HeteroConv(
            {
                edge_type: conv.SimpleConv(aggr="mean", combine_root=None)
                for edge_type in edge_types
            },
            aggr="mean",
        )

        self.temp = temperature
        self.max_negatives = max_negatives

    def forward(
        self, data: HeteroData, x_dict: Dict[NodeType, torch.Tensor]
    ) -> torch.Tensor:
        edge_index_dict = data.collect("edge_index")

        context_dict = {k: self.linear_dict[k](v) for k, v in x_dict.items()}
        context_dict = self.mean_pooling(context_dict, edge_index_dict)

        loss = 0.0
        count = 0
        for node_type, x in x_dict.items():
            context = context_dict[node_type]
            batch_size = x.size(0)
            if batch_size <= 1:
                continue

            sim_m: torch.Tensor = context @ x.T
            labels = torch.arange(batch_size, device=sim_m.device)
            num_negatives = context.size(0) - 1

            if num_negatives > self.max_negatives:
                sim_pos = sim_m.diag()
                n = sim_m.size(0)
                sim_neg = sim_m.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)

                rnd_idx = torch.stack(
                    [
                        torch.randperm(num_negatives - 1)[: self.max_negatives]
                        for _ in range(n)
                    ]
                )
                sim_neg = torch.gather(sim_neg, 1, rnd_idx)
                sim_m = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)

                labels = torch.zeros(batch_size, dtype=torch.long, device=sim_m.device)

                num_negatives = self.max_negatives

            norm_factor = -math.log(1 / (num_negatives + 1))
            loss += (
                torch.nn.functional.cross_entropy(
                    sim_m / self.temp, labels, reduction="sum"
                )
                / norm_factor
            )
            count += batch_size

        return loss / count if count > 0 else torch.tensor(0.0)
