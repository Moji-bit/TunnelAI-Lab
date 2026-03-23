from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


_REQUIRED_FILES = {
    "tunnel_config": "tunnel_config.csv",
    "scenario_metadata": "scenario_metadata.csv",
    "timeseries": "timeseries.csv",
    "ground_truth": "ground_truth.csv",
}


def _read_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if isinstance(source, bytes):
        return pd.read_csv(BytesIO(source))
    return pd.read_csv(source)


def load_tunnel_config(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    return _read_csv(source)


def load_scenario_metadata(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    return _read_csv(source)


def load_timeseries(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    return _read_csv(source)


def load_ground_truth(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    return _read_csv(source)


def load_from_directory(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
    loaded: dict[str, pd.DataFrame] = {}
    for key, filename in _REQUIRED_FILES.items():
        path = input_dir / filename
        if path.exists():
            loaded[key] = _read_csv(path)
    return loaded
