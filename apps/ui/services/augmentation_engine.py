from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_SIGNAL_COLS = [
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
    "camera_alarm_count",
]

BINARY_COLS = [
    "barrier_entry_state",
    "barrier_exit_state",
    "fire_alarm_state",
    "sos_calls_active",
    "emergency_mode_active",
    "sensor_fault_active",
    "fan_fault_active",
    "camera_fault_active",
]

RANGES = {
    "occupancy_pct": (0, 100),
    "visibility_m": (0, None),
    "jet_fan_power_pct": (0, 100),
    "speed_mean_kmh": (0, 160),
    "humidity_pct": (0, 100),
}

EVENT_PRESETS: dict[str, dict[str, Any]] = {
    "normal traffic": {"event_type": "normal", "severity": "low"},
    "congestion": {"event_type": "congestion", "severity": "medium"},
    "accident": {"event_type": "accident", "severity": "high"},
    "fire": {"event_type": "fire", "severity": "high"},
    "sensor fault": {"event_type": "sensor_fault", "severity": "medium"},
    "winter weather": {"event_type": "weather", "weather_type": "snow", "severity": "medium"},
    "heavy rain": {"event_type": "weather", "weather_type": "rain", "severity": "medium"},
    "mixed disturbance": {"event_type": "accident", "weather_type": "fog", "severity": "high"},
}


@dataclass
class AugmentationConfig:
    target_scenarios: int = 5000
    augmentation_strength: float = 0.25
    noise_level: float = 0.03
    event_shift_range_s: int = 30
    missing_rate: float = 0.01
    outlier_rate: float = 0.003
    lag_max_steps: int = 2
    sampling_jitter_prob: float = 0.15
    allowed_weather: list[str] = field(default_factory=lambda: ["clear", "rain", "snow", "fog"])
    class_balance_targets: dict[str, float] = field(
        default_factory=lambda: {
            "normal": 0.40,
            "congestion": 0.18,
            "accident": 0.16,
            "fire": 0.12,
            "sensor_fault": 0.14,
        }
    )
    random_seed: int = 42


def _clip_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (min_v, max_v) in RANGES.items():
        if col in out.columns:
            series = pd.to_numeric(out[col], errors="coerce")
            if min_v is not None:
                series = series.clip(lower=min_v)
            if max_v is not None:
                series = series.clip(upper=max_v)
            out[col] = series

    for col in BINARY_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    return out


def _augment_signals(ts: pd.DataFrame, cfg: AugmentationConfig, rng: np.random.Generator) -> pd.DataFrame:
    out = ts.copy().sort_values("timestamp_s")

    for col in [c for c in NUMERIC_SIGNAL_COLS if c in out.columns]:
        s = pd.to_numeric(out[col], errors="coerce").astype(float)
        std = float(np.nanstd(s)) if len(s) else 0.0
        noise = rng.normal(0.0, cfg.noise_level * max(1.0, std), size=len(s))
        drift = np.linspace(0.0, rng.normal(0.0, cfg.augmentation_strength * max(0.5, std)), len(s))
        scale = float(rng.normal(1.0, cfg.augmentation_strength * 0.08))
        out[col] = (s * scale) + noise + drift

        if rng.random() < 0.4:
            out[col] = out[col].rolling(window=3, min_periods=1).mean()

        if rng.random() < cfg.sampling_jitter_prob:
            lag = int(rng.integers(0, cfg.lag_max_steps + 1))
            if lag > 0:
                out[col] = out[col].shift(lag).bfill()

        outlier_mask = rng.random(len(out)) < cfg.outlier_rate
        if outlier_mask.any():
            out.loc[outlier_mask, col] = out.loc[outlier_mask, col] * rng.uniform(1.1, 1.35)

        missing_mask = rng.random(len(out)) < cfg.missing_rate
        if missing_mask.any():
            out.loc[missing_mask, col] = np.nan
            out[col] = out[col].interpolate(limit_direction="both")

    return _clip_values(out)


def _event_mask(ts: pd.DataFrame, event_start_s: float, event_duration_s: float) -> pd.Series:
    start = float(max(0.0, event_start_s))
    end = start + float(max(0.0, event_duration_s))
    t = pd.to_numeric(ts["timestamp_s"], errors="coerce").fillna(0)
    return (t >= start) & (t < end)


def _apply_event_logic(ts: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    out = ts.copy()
    event_type = str(scenario.get("event_type", "normal")).lower()
    mask = _event_mask(out, scenario.get("event_start_s", 0), scenario.get("event_duration_s", 0))

    if not mask.any() or event_type in {"normal", "none"}:
        return out

    if event_type in {"accident", "collision"}:
        for col, scale in {"speed_mean_kmh": 0.6, "flow_veh_h": 0.8}.items():
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * scale
        for col, add in {"queue_length_m": 40, "stopped_vehicle_count": 2}.items():
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] + add

    if event_type in {"congestion", "stau"}:
        if "occupancy_pct" in out.columns:
            out.loc[mask, "occupancy_pct"] = out.loc[mask, "occupancy_pct"] + 18
        if "speed_mean_kmh" in out.columns:
            out.loc[mask, "speed_mean_kmh"] *= 0.72
        if "queue_length_m" in out.columns:
            out.loc[mask, "queue_length_m"] += 65

    if event_type in {"fire", "vehicle_fire"}:
        if "fire_alarm_state" in out.columns:
            out.loc[mask, "fire_alarm_state"] = 1
        if "visibility_m" in out.columns:
            out.loc[mask, "visibility_m"] *= 0.45
        if "co_ppm" in out.columns:
            out.loc[mask, "co_ppm"] = out.loc[mask, "co_ppm"] + 4.0
        if "jet_fan_power_pct" in out.columns:
            out.loc[mask, "jet_fan_power_pct"] = out.loc[mask, "jet_fan_power_pct"] + 20

    if event_type in {"sensor_fault", "fault"}:
        fault_col = [c for c in ["sensor_fault_active", "fan_fault_active", "camera_fault_active"] if c in out.columns]
        for col in fault_col:
            out.loc[mask, col] = 1

    return _clip_values(out)


def _build_ground_truth(ts: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    gt = pd.DataFrame({"scenario_id": ts["scenario_id"], "timestamp_s": ts["timestamp_s"]})
    event_type = str(scenario.get("event_type", "normal")).lower()
    mask = _event_mask(ts, scenario.get("event_start_s", 0), scenario.get("event_duration_s", 0))

    gt["label_event_type"] = "normal"
    gt.loc[mask, "label_event_type"] = event_type
    gt["label_event_active"] = mask.astype(int)

    risk = str(scenario.get("event_severity", "low")).lower()
    gt["label_risk_level"] = "low"
    gt.loc[mask, "label_risk_level"] = risk

    gt["label_phase"] = "normal"
    if mask.any():
        first_idx = int(np.where(mask.to_numpy())[0][0])
        gt.iloc[:first_idx, gt.columns.get_loc("label_phase")] = "pre_event"
        gt.loc[mask, "label_phase"] = "event"
        gt.iloc[first_idx + int(mask.sum()) :, gt.columns.get_loc("label_phase")] = "recovery"

    gt["label_event_location_m"] = 0.0
    gt.loc[mask, "label_event_location_m"] = float(scenario.get("event_location_m", 0.0))
    return gt


def scenario_quality_score(timeseries: pd.DataFrame, ground_truth: pd.DataFrame, scenario_meta: pd.Series) -> float:
    penalties = 0.0
    for col, (low, high) in RANGES.items():
        if col not in timeseries.columns:
            continue
        v = pd.to_numeric(timeseries[col], errors="coerce")
        if low is not None:
            penalties += float((v < low).mean())
        if high is not None:
            penalties += float((v > high).mean())

    active = pd.to_numeric(ground_truth.get("label_event_active", 0), errors="coerce").fillna(0)
    event_type = str(scenario_meta.get("event_type", "normal")).lower()
    if event_type == "normal" and active.mean() > 0:
        penalties += 0.3
    if event_type != "normal" and active.mean() == 0:
        penalties += 0.6

    score = max(0.0, min(1.0, 1.0 - penalties))
    return round(score, 4)


def augment_scenario(
    base_scenario: pd.Series,
    base_timeseries: pd.DataFrame,
    tunnel_row: pd.Series,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
    forced_event_type: str | None = None,
    scenario_id: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    scenario = base_scenario.copy()

    if scenario_id:
        scenario["scenario_id"] = scenario_id
    scenario["run_id"] = f"RUN_AUG_{rng.integers(10_000_000, 99_999_999)}"
    scenario["random_seed"] = int(rng.integers(1, 2_147_483_647))

    if forced_event_type:
        scenario["event_type"] = forced_event_type

    shift = int(rng.integers(-cfg.event_shift_range_s, cfg.event_shift_range_s + 1))
    scenario["event_start_s"] = max(0, float(scenario.get("event_start_s", 0)) + shift)
    duration_scale = float(rng.uniform(0.75, 1.25))
    scenario["event_duration_s"] = max(0, float(scenario.get("event_duration_s", 0)) * duration_scale)
    scenario["event_location_m"] = float(
        np.clip(
            float(scenario.get("event_location_m", 0)) + rng.normal(0, max(10, float(tunnel_row.get("length_m", 500)) * 0.05)),
            0,
            float(tunnel_row.get("length_m", 500)),
        )
    )

    if "event_severity" in scenario.index:
        sev = ["low", "medium", "high"]
        if rng.random() < 0.25:
            scenario["event_severity"] = str(rng.choice(sev))

    if "weather_type" in scenario.index and cfg.allowed_weather:
        scenario["weather_type"] = str(rng.choice(cfg.allowed_weather))

    for col, sigma in {
        "wind_speed_mps": 1.0,
        "outside_temp_c": 2.0,
        "entry_flow_veh_h": 120,
        "heavy_vehicle_pct": 1.8,
        "speed_limit_kmh": 5,
    }.items():
        if col in scenario.index:
            scenario[col] = float(scenario[col]) + float(rng.normal(0, sigma))

    ts = base_timeseries.copy()
    ts["scenario_id"] = str(scenario["scenario_id"])
    ts = _augment_signals(ts, cfg=cfg, rng=rng)
    ts = _apply_event_logic(ts, scenario=scenario)

    gt = _build_ground_truth(ts, scenario=scenario)
    quality = scenario_quality_score(ts, gt, scenario)
    scenario["quality_score"] = quality

    return scenario, ts, gt, tunnel_row.copy()


def _target_event_sequence(target_count: int, class_balance_targets: dict[str, float]) -> list[str]:
    if not class_balance_targets:
        return ["normal"] * target_count

    total = sum(class_balance_targets.values())
    normalized = {k: v / total for k, v in class_balance_targets.items()}
    counts = {k: int(round(target_count * p)) for k, p in normalized.items()}
    diff = target_count - sum(counts.values())
    if diff != 0:
        largest = max(normalized, key=normalized.get)
        counts[largest] += diff

    seq: list[str] = []
    for k, n in counts.items():
        seq.extend([k] * max(0, n))
    return seq[:target_count]


def generate_augmented_dataset(
    tunnel_config: pd.DataFrame,
    scenario_metadata: pd.DataFrame,
    timeseries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: AugmentationConfig,
) -> dict[str, pd.DataFrame]:
    del ground_truth  # ground_truth is regenerated from augmented metadata and timeseries.

    rng = np.random.default_rng(cfg.random_seed)

    scenarios_by_id = {str(x["scenario_id"]): x for _, x in scenario_metadata.iterrows()}
    ts_group = {sid: block.copy() for sid, block in timeseries.groupby("scenario_id")}
    tunnel_by_id = {str(x["tunnel_id"]): x for _, x in tunnel_config.iterrows()}

    base_ids = sorted(set(scenarios_by_id).intersection(ts_group))
    if not base_ids:
        raise ValueError("No common scenario_id between scenario_metadata and timeseries.")

    target_events = _target_event_sequence(cfg.target_scenarios, cfg.class_balance_targets)
    rng.shuffle(target_events)

    out_scenarios: list[pd.Series] = []
    out_ts: list[pd.DataFrame] = []
    out_gt: list[pd.DataFrame] = []
    out_tunnel: list[pd.Series] = []

    for i in range(cfg.target_scenarios):
        base_id = str(rng.choice(base_ids))
        base_scn = scenarios_by_id[base_id]
        tunnel = tunnel_by_id[str(base_scn["tunnel_id"])]
        event = target_events[i] if i < len(target_events) else None
        new_id = f"AUG_{i+1:06d}"

        scn, ts, gt, tun = augment_scenario(
            base_scenario=base_scn,
            base_timeseries=ts_group[base_id],
            tunnel_row=tunnel,
            cfg=cfg,
            rng=rng,
            forced_event_type=event,
            scenario_id=new_id,
        )
        out_scenarios.append(scn)
        out_ts.append(ts)
        out_gt.append(gt)
        out_tunnel.append(tun)

    scenario_df = pd.DataFrame(out_scenarios)
    timeseries_df = pd.concat(out_ts, ignore_index=True)
    ground_truth_df = pd.concat(out_gt, ignore_index=True)
    tunnel_df = pd.DataFrame(out_tunnel).drop_duplicates(subset=["tunnel_id"]).reset_index(drop=True)

    return {
        "scenario_metadata": scenario_df,
        "timeseries": timeseries_df,
        "ground_truth": ground_truth_df,
        "tunnel_config": tunnel_df,
    }
