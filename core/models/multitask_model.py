# models/multitask_model.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from core.models.backbone.transformer import TransformerBackbone
from core.models.backbone.lstm import LSTMBackbone
from core.models.heads.forecasting import ForecastHead
from core.models.heads.event import EventHead
from core.models.heads.risk import RiskHead

class MultiTaskModel(nn.Module):
    """Configurable multitask model with optional forecast/risk heads.

    Forward output is a dict with keys:
      - backbone_features
      - event_logits
      - forecast (optional)
      - risk (optional)
    """

    def __init__(
        self,
        config: ModelConfig | dict | None = None,
        **legacy_kwargs,
    ):
        super().__init__()

        if config is None:
            config = ModelConfig(**legacy_kwargs)
        elif isinstance(config, dict):
            config = ModelConfig(**config)
        self.cfg = config

        if self.cfg.backbone.lower() == "transformer":
            self.backbone = TransformerBackbone(
                d_in=self.cfg.d_in,
                d_model=self.cfg.d_model,
                n_heads=self.cfg.n_heads,
                n_layers=self.cfg.n_layers,
                dropout=self.cfg.dropout,
                dim_feedforward=self.cfg.dim_feedforward,
            )
        elif self.cfg.backbone.lower() == "lstm":
            self.backbone = LSTMBackbone(
                d_in=self.cfg.d_in,
                d_model=self.cfg.d_model,
                n_layers=self.cfg.n_layers,
                dropout=self.cfg.dropout,
                bidirectional=self.cfg.bidirectional,
            )
        else:
            raise ValueError("backbone must be 'transformer' or 'lstm'")

        self.event = EventHead(
            d_model=self.cfg.d_model,
            n_classes=self.cfg.num_event_classes,
            dropout=self.cfg.dropout,
            pooling=self.cfg.pooling,
        )

        self.forecast = (
            ForecastHead(
                d_model=self.cfg.d_model,
                H=self.cfg.H,
                m=self.cfg.m,
                dropout=self.cfg.dropout,
                pooling=self.cfg.pooling,
            )
            if self.cfg.use_forecast_head
            else None
        )

        self.risk = (
            RiskHead(
                d_model=self.cfg.d_model,
                dropout=self.cfg.dropout,
                pooling=self.cfg.pooling,
            )
            if self.cfg.use_risk_head
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        out: dict[str, torch.Tensor] = {
            "backbone_features": h,
            "event_logits": self.event(h),
        }
        if self.forecast is not None:
            out["forecast"] = self.forecast(h)
        if self.risk is not None:
            out["risk"] = self.risk(h)
        return out
