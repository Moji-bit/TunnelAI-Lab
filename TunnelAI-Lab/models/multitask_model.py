# models/multitask_model.py
import torch
import torch.nn as nn

from models.backbone.transformer import TransformerBackbone
from models.backbone.lstm import LSTMBackbone
from models.heads.forecasting import ForecastHead
from models.heads.event import EventHead
from models.heads.risk import RiskHead

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        d_in: int,
        d_model: int,
        H: int,
        m: int,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()

        if backbone.lower() == "transformer":
            self.backbone = TransformerBackbone(d_in, d_model, n_heads, n_layers, dropout)
        elif backbone.lower() == "lstm":
            self.backbone = LSTMBackbone(d_in, d_model, n_layers=2, dropout=dropout)
        else:
            raise ValueError("backbone must be 'transformer' or 'lstm'")

        self.forecast = ForecastHead(d_model, H, m, dropout, pooling)
        self.event = EventHead(d_model, dropout, pooling)
        self.risk = RiskHead(d_model, dropout, pooling)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        y_f = self.forecast(h)
        y_e = self.event(h)   # logits
        y_r = self.risk(h)    # score/logit
        return y_f, y_e, y_r