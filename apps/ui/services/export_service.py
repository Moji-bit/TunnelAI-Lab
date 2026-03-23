from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPORT_MAP = {
    "timeseries": "augmented_timeseries.csv",
    "ground_truth": "augmented_ground_truth.csv",
    "scenario_metadata": "augmented_scenario_metadata.csv",
    "tunnel_config": "augmented_tunnel_config.csv",
    "merged": "merged_training_dataset.csv",
}


def export_augmented_files(datasets: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for key, filename in EXPORT_MAP.items():
        if key in datasets and datasets[key] is not None:
            path = out_path / filename
            datasets[key].to_csv(path, index=False)
            written[key] = str(path)
    return written
