import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size: int = 2 ** 8
    ) -> None:
        super(QNet, self).__init__()
        self.core = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        return output
