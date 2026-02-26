# models/heads/forecasting.py
import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    def __init__(self, d_model: int, H: int, m: int, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.H = H
        self.m = m
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, H * m),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,L,d_model)
        z = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        y = self.net(z)
        return y.view(-1, self.H, self.m)
        