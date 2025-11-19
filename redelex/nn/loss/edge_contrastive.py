from typing import List, Dict

import math

import torch

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter


class EdgeContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        edge_types: List[EdgeType],
        temperature: float = 0.1,
        max_negatives: int = 255,
    ):
        super().__init__()
        self.channels = channels

        self.weights_dict = torch.nn.ParameterDict(
            {
                f"{src_node}_{name}_{dst_node}": torch.nn.Parameter(
                    torch.nn.init.kaiming_uniform_(
                        torch.empty((channels, channels)), mode="fan_in", a=math.sqrt(5)
                    )
                )
                for (src_node, name, dst_node) in edge_types
                if not name.startswith("rev_")
            }
        )

        self.temp = temperature
        self.max_negatives = max_negatives

    def forward(
        self, data: HeteroData, x_dict: Dict[NodeType, torch.Tensor]
    ) -> torch.Tensor:
        edge_index_dict = data.collect("edge_index")

        loss = 0.0
        count = 0
        for edge_type, edge_index in edge_index_dict.items():
            src_node, name, dst_node = edge_type
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_x = x_dict[src_node]
            dst_x = x_dict[dst_node]

            if src_x.size(0) <= 1 or dst_x.size(0) <= 1:
                continue

            total_src = len(x_dict[src_node])
            total_dst = len(x_dict[dst_node])

            if name.startswith("rev_"):
                W = self.weights_dict[f"{dst_node}_{name.lstrip('rev_')}_{src_node}"].T
            else:
                W = self.weights_dict[f"{src_node}_{name}_{dst_node}"]

            sim_M = dst_x @ W @ src_x.T

            exp_sim_M = torch.exp(sim_M / self.temp)

            adj_M = torch.zeros((total_dst, total_src), dtype=torch.bool)
            adj_M[dst_idx, src_idx] = True
            pos_idx = adj_M.nonzero()
            neg_idx = (~adj_M).nonzero()

            pos_sim = exp_sim_M[pos_idx[:, 0], pos_idx[:, 1]]

            num_negatives = (~adj_M).sum(dim=1)[pos_idx[:, 0]]
            batch_size = pos_sim.size(0)

            if batch_size <= 1 or num_negatives.min() == 0:
                continue

            max_total_negatives = self.max_negatives * total_dst
            if neg_idx.size(0) > max_total_negatives:
                mask = torch.randperm(neg_idx.size(0), device=neg_idx.device)[
                    :max_total_negatives
                ]
                neg_idx = neg_idx[mask]
                num_negatives = torch.zeros(
                    total_dst, dtype=torch.long, device=neg_idx.device
                )
                idx, _num_negatives = torch.unique(
                    neg_idx[:, 0], return_counts=True, sorted=False
                )
                num_negatives[idx] = _num_negatives
                num_negatives = num_negatives[pos_idx[:, 0]]

            neg_sim = exp_sim_M[neg_idx[:, 0], neg_idx[:, 1]]
            neg_sim = scatter(neg_sim, neg_idx[:, 0], reduce="sum")[pos_idx[:, 0]]

            sum_sim = pos_sim + neg_sim

            norm_factor = -torch.log(1 / (num_negatives + 1))

            loss += (-torch.log(pos_sim / sum_sim) / norm_factor).sum()
            count += batch_size

        return loss / count if count > 0 else torch.tensor(0.0)
