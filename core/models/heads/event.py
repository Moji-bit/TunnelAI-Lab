# models/heads/event.py
import torch
import torch.nn as nn

class EventHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        return self.net(z).squeeze(-1)  # logits (B,)