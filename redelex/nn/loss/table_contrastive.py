import math
from typing import Dict, List

import torch
from torch_geometric.typing import NodeType


class TableContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        node_types: List[NodeType],
        temperature: float = 0.1,
        max_negatives: int = 255,
    ):
        super().__init__()

        self.channels = channels

        self.linear_dict = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(channels, channels, bias=False)
                for node_type in node_types
            }
        )

        self.temp = temperature
        self.max_negatives = max_negatives

    def forward(
        self, x_dict: Dict[NodeType, torch.Tensor], cor_dict: Dict[NodeType, torch.Tensor]
    ) -> torch.Tensor:
        x_dict = {k: self.linear_dict[k](v) for k, v in x_dict.items()}
        cor_dict = {k: self.linear_dict[k](v) for k, v in cor_dict.items()}
        loss = 0.0
        count = 0
        for tname in x_dict:
            x = x_dict[tname]
            cor = cor_dict[tname]
            batch_size = x.size(0)
            num_negatives = cor.size(0) - 1
            if batch_size <= 1:
                continue

            sim_m = cor @ x.T
            labels = torch.arange(batch_size, device=sim_m.device, dtype=torch.long)

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
