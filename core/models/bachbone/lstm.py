# models/backbone/lstm.py
import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.norm(h)  # (B,L,d_model)