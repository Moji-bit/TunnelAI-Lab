# models/heads/event.py
import torch
import torch.nn as nn

class EventHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 2, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        return self.net(z)  # logits (B, n_classes)
