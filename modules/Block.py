import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, drop_p: int = 0.) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


