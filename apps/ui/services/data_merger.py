from __future__ import annotations

import pandas as pd


def build_merged_dataset(
    tunnel_config: pd.DataFrame,
    scenario_metadata: pd.DataFrame,
    timeseries: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    scenario_enriched = scenario_metadata.merge(tunnel_config, on="tunnel_id", how="left", suffixes=("", "_tunnel"))
    ts_gt = timeseries.merge(ground_truth, on=["scenario_id", "timestamp_s"], how="inner", suffixes=("", "_gt"))
    merged = ts_gt.merge(scenario_enriched, on="scenario_id", how="left", suffixes=("", "_meta"))

    merged = merged.sort_values(["scenario_id", "timestamp_s"]).reset_index(drop=True)
    return merged
