from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "tunnel_config": [
        "tunnel_id",
        "tunnel_name",
        "tunnel_type",
        "length_m",
        "width_m",
        "clearance_height_m",
        "gradient_pct",
        "curvature_radius_m",
        "cross_section_profile",
        "tube_count",
        "lanes_per_tube",
        "traffic_mode",
        "ventilation_type",
        "jet_fan_count",
        "camera_count",
        "fire_detector_count",
        "emergency_station_count",
        "lighting_zones_count",
        "segment_length_m",
        "segment_count",
    ],
    "scenario_metadata": [
        "scenario_id",
        "run_id",
        "tunnel_id",
        "random_seed",
        "simulation_start_ts",
        "simulation_duration_s",
        "time_step_s",
        "weather_type",
        "outside_temp_c",
        "outside_humidity_pct",
        "outside_pressure_hpa",
        "wind_direction",
        "wind_speed_mps",
        "daytime",
        "traffic_demand_level",
        "aadt",
        "heavy_vehicle_pct",
        "speed_limit_kmh",
        "entry_flow_veh_h",
        "event_type",
        "event_start_s",
        "event_duration_s",
        "event_location_m",
        "event_tube",
        "event_lane",
        "event_severity",
    ],
    "timeseries": [
        "scenario_id",
        "timestamp_s",
        "speed_mean_kmh",
        "flow_veh_h",
        "occupancy_pct",
        "vehicle_count",
        "queue_length_m",
        "stopped_vehicle_count",
        "heavy_vehicle_count",
        "co_ppm",
        "no2_ppm",
        "pm25_ug_m3",
        "temp_c",
        "humidity_pct",
        "visibility_m",
        "air_velocity_mps",
        "pressure_hpa",
        "jet_fan_active_count",
        "jet_fan_power_pct",
        "barrier_entry_state",
        "barrier_exit_state",
        "fire_alarm_state",
        "lighting_mode",
        "camera_alarm_count",
        "sos_calls_active",
        "emergency_mode_active",
        "sensor_fault_active",
        "fan_fault_active",
        "camera_fault_active",
    ],
    "ground_truth": [
        "scenario_id",
        "timestamp_s",
        "label_event_type",
        "label_event_active",
        "label_risk_level",
        "label_phase",
        "label_event_location_m",
    ],
}

PLAUSIBILITY_RANGES = {
    "occupancy_pct": (0, 100),
    "visibility_m": (0, None),
    "jet_fan_power_pct": (0, 100),
    "heavy_vehicle_pct": (0, 100),
    "outside_humidity_pct": (0, 100),
}


@dataclass
class ValidationIssue:
    level: str
    message: str
    file_key: str | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    def add(self, level: str, message: str, file_key: str | None = None) -> None:
        self.issues.append(ValidationIssue(level=level, message=message, file_key=file_key))

    def as_rows(self) -> list[dict[str, Any]]:
        return [{"level": x.level, "file": x.file_key or "cross_file", "message": x.message} for x in self.issues]


def validate_schema(file_key: str, df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    required = REQUIRED_COLUMNS[file_key]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        report.add("error", f"Missing required columns: {missing}", file_key=file_key)
        return report

    for col in required:
        na_count = int(df[col].isna().sum())
        if na_count > 0:
            report.add("warning", f"Column '{col}' contains {na_count} missing values.", file_key=file_key)

    if file_key in {"timeseries", "ground_truth", "scenario_metadata"} and "scenario_id" in df.columns:
        if df["scenario_id"].duplicated().any() and file_key == "scenario_metadata":
            report.add("error", "scenario_id must be unique in scenario_metadata.", file_key=file_key)

    for col, (min_v, max_v) in PLAUSIBILITY_RANGES.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if min_v is not None:
                n = int((values < min_v).sum())
                if n > 0:
                    report.add("warning", f"{n} values below minimum {min_v} in '{col}'.", file_key=file_key)
            if max_v is not None:
                n = int((values > max_v).sum())
                if n > 0:
                    report.add("warning", f"{n} values above maximum {max_v} in '{col}'.", file_key=file_key)

    return report


def validate_cross_file_consistency(
    tunnel_config: pd.DataFrame,
    scenario_metadata: pd.DataFrame,
    timeseries: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> ValidationReport:
    report = ValidationReport()

    scenario_meta_ids = set(scenario_metadata["scenario_id"].astype(str))
    ts_ids = set(timeseries["scenario_id"].astype(str))
    gt_ids = set(ground_truth["scenario_id"].astype(str))

    for name, ids in {"timeseries": ts_ids, "ground_truth": gt_ids}.items():
        unknown = sorted(ids - scenario_meta_ids)
        if unknown:
            report.add("error", f"{name} contains scenario_id not found in scenario_metadata (first 10): {unknown[:10]}")

    tunnel_ids = set(tunnel_config["tunnel_id"].astype(str))
    unknown_tunnel = sorted(set(scenario_metadata["tunnel_id"].astype(str)) - tunnel_ids)
    if unknown_tunnel:
        report.add("error", f"scenario_metadata has unknown tunnel_id values (first 10): {unknown_tunnel[:10]}")

    ts_key = timeseries[["scenario_id", "timestamp_s"]].copy()
    gt_key = ground_truth[["scenario_id", "timestamp_s"]].copy()
    ts_key["scenario_id"] = ts_key["scenario_id"].astype(str)
    gt_key["scenario_id"] = gt_key["scenario_id"].astype(str)

    ts_idx = set(map(tuple, ts_key.to_records(index=False)))
    gt_idx = set(map(tuple, gt_key.to_records(index=False)))

    miss_gt = list(ts_idx - gt_idx)
    miss_ts = list(gt_idx - ts_idx)
    if miss_gt:
        report.add("error", f"timeseries rows without matching ground_truth timestamps (first 10): {miss_gt[:10]}")
    if miss_ts:
        report.add("error", f"ground_truth rows without matching timeseries timestamps (first 10): {miss_ts[:10]}")

    return report
