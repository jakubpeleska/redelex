import torch

from torch_geometric.nn import PositionalEncoding


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
