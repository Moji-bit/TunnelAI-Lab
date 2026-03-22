import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        hidden_size = d_model // 2 if self.bidirectional else d_model
        if self.bidirectional and d_model % 2 != 0:
            raise ValueError("d_model must be even when bidirectional=True")

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.out_dim = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.norm(h)
