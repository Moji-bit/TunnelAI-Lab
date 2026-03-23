from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowingConfig:
    enabled: bool = False
    window_size: int = 60
    horizon: int = 20
    stride: int = 10


def list_presets() -> list[str]:
    return [
        "normal traffic",
        "congestion",
        "accident",
        "fire",
        "sensor fault",
        "winter weather",
        "heavy rain",
        "mixed disturbance",
    ]
