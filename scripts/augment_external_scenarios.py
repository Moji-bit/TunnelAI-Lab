from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AugmentConfig:
    n_scenarios: int = 10000
    seed: int = 42
    continuous_noise: float = 0.04
    count_jitter: float = 0.08
    binary_flip_prob: float = 0.01
    event_probability: float = 0.35


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV ist leer: {path}")
    return df


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _augment_sensor_block(block: pd.DataFrame, rng: np.random.Generator, cfg: AugmentConfig) -> pd.DataFrame:
    out = block.copy()

    binary_cols = [c for c in out.columns if c.endswith("_active") or c.endswith("_state") or c.endswith("_fault_active")]
    binary_cols += [c for c in ["emergency_mode_active", "sos_calls_active"] if c in out.columns]
    binary_cols = sorted(set(binary_cols))

    count_like_cols = [
        c
        for c in out.columns
        if c.endswith("_count")
        or c.endswith("_veh_h")
        or c in {"vehicle_count", "queue_length_m", "camera_alarm_count", "jet_fan_active_count"}
    ]
    count_like_cols = [c for c in sorted(set(count_like_cols)) if c in out.columns and c not in binary_cols]

    exclude_num = {"timestamp_s", "scenario_id"}
    num_cols = _numeric_columns(out, exclude=exclude_num)
    continuous_cols = [c for c in num_cols if c not in count_like_cols and c not in binary_cols]

    for col in continuous_cols:
        s = out[col].astype(float)
        std = float(np.nanstd(s.values))
        scale = float(rng.normal(1.0, cfg.continuous_noise / 2.0))
        noise = rng.normal(0.0, max(1e-9, cfg.continuous_noise * max(1.0, std)), size=len(s))
        out[col] = s * scale + noise

    for col in count_like_cols:
        s = out[col].astype(float)
        mult = rng.normal(1.0, cfg.count_jitter, size=len(s))
        out[col] = np.clip(np.round(s * mult), a_min=0, a_max=None)

    for col in binary_cols:
        s = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        flips = rng.random(len(s)) < cfg.binary_flip_prob
        out[col] = np.where(flips, 1 - s, s)

    if "lighting_mode" in out.columns:
        if rng.random() < 0.08:
            out["lighting_mode"] = rng.choice(["normal", "dimmed", "emergency"], p=[0.8, 0.15, 0.05], size=len(out))

    return out


def _apply_event_to_labels(
    label_block: pd.DataFrame,
    event_type: str,
    event_start: int,
    event_end: int,
    event_location_m: float,
) -> pd.DataFrame:
    out = label_block.copy()

    if "label_event_active" not in out.columns:
        out["label_event_active"] = 0
    if "label_event_type" not in out.columns:
        out["label_event_type"] = "normal"
    if "label_risk_level" not in out.columns:
        out["label_risk_level"] = "low"
    if "label_phase" not in out.columns:
        out["label_phase"] = "normal"
    if "label_event_location_m" not in out.columns:
        out["label_event_location_m"] = 0

    out["label_event_active"] = 0
    out["label_event_type"] = "normal"
    out["label_risk_level"] = "low"
    out["label_phase"] = "normal"
    out["label_event_location_m"] = 0

    if event_type == "normal":
        return out

    ts = pd.to_numeric(out["timestamp_s"], errors="coerce").fillna(0).astype(int)
    mask = (ts >= event_start) & (ts < event_end)
    out.loc[mask, "label_event_active"] = 1
    out.loc[mask, "label_event_type"] = event_type
    out.loc[mask, "label_event_location_m"] = float(event_location_m)

    risk_map = {
        "collision": "high",
        "vehicle_fire": "high",
        "wrong_way_driver": "high",
        "stalled_vehicle": "medium",
        "congestion": "medium",
    }
    out.loc[mask, "label_risk_level"] = risk_map.get(event_type, "medium")
    out.loc[mask, "label_phase"] = "event"
    return out


def generate_augmented_dataset(
    labels_csv: str,
    sensors_csv: str,
    tunnel_csv: str,
    scenario_csv: str,
    out_dir: str,
    cfg: AugmentConfig,
) -> dict[str, object]:
    labels = _load_csv(labels_csv)
    sensors = _load_csv(sensors_csv)
    tunnels = _load_csv(tunnel_csv)
    scenarios = _load_csv(scenario_csv)

    required_ids = {"scenario_id"}
    for name, df in {
        "labels": labels,
        "sensors": sensors,
        "scenarios": scenarios,
    }.items():
        if not required_ids.issubset(df.columns):
            raise ValueError(f"{name}-CSV braucht Spalte scenario_id")

    base_ids = sorted(set(labels["scenario_id"]).intersection(sensors["scenario_id"]).intersection(scenarios["scenario_id"]))
    if not base_ids:
        raise ValueError("Keine gemeinsamen scenario_id in labels/sensors/scenarios gefunden")

    rng = np.random.default_rng(cfg.seed)

    scenario_rows: list[pd.DataFrame] = []
    sensor_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []

    observed_events = sorted({str(v).strip().lower() for v in scenarios.get("event_type", pd.Series(["normal"])) if pd.notna(v)})
    observed_events = [e for e in observed_events if e]
    non_normal_events = [e for e in observed_events if e not in {"normal", "none"}]
    if not non_normal_events:
        non_normal_events = ["collision", "stalled_vehicle", "congestion", "vehicle_fire"]

    tunnel_length = float(pd.to_numeric(tunnels.get("length_m", pd.Series([1500])), errors="coerce").fillna(1500).iloc[0])

    for i in range(cfg.n_scenarios):
        base_id = str(rng.choice(base_ids))
        new_id = f"AUG_{i + 1:05d}"

        base_scenario = scenarios.loc[scenarios["scenario_id"] == base_id].iloc[0].copy()
        base_sensor = sensors.loc[sensors["scenario_id"] == base_id].copy()
        base_label = labels.loc[labels["scenario_id"] == base_id].copy()

        aug_sensor = _augment_sensor_block(base_sensor, rng=rng, cfg=cfg)

        duration = int(pd.to_numeric(base_scenario.get("simulation_duration_s", len(aug_sensor) * 2), errors="coerce"))
        if duration <= 0:
            duration = int(max(10, len(aug_sensor) * 2))

        is_event = bool(rng.random() < cfg.event_probability)
        event_type = str(rng.choice(non_normal_events)) if is_event else "normal"

        if is_event:
            event_start = int(rng.integers(0, max(1, int(duration * 0.65))))
            event_duration = int(rng.integers(max(8, int(duration * 0.1)), max(12, int(duration * 0.35))))
            event_end = min(duration, event_start + event_duration)
            event_location = float(rng.uniform(0, max(1.0, tunnel_length)))
        else:
            event_start, event_duration, event_end, event_location = 0, 0, 0, 0.0

        aug_label = _apply_event_to_labels(
            label_block=base_label,
            event_type=event_type,
            event_start=event_start,
            event_end=event_end,
            event_location_m=event_location,
        )

        aug_sensor["scenario_id"] = new_id
        aug_label["scenario_id"] = new_id

        base_scenario["scenario_id"] = new_id
        base_scenario["run_id"] = f"RUN_AUG_{cfg.seed}_{i + 1:05d}"
        base_scenario["random_seed"] = int(rng.integers(1, 1_000_000_000))
        if "event_type" in base_scenario.index:
            base_scenario["event_type"] = event_type
        if "event_start_s" in base_scenario.index:
            base_scenario["event_start_s"] = event_start
        if "event_duration_s" in base_scenario.index:
            base_scenario["event_duration_s"] = event_duration
        if "event_location_m" in base_scenario.index:
            base_scenario["event_location_m"] = event_location
        if "event_severity" in base_scenario.index:
            base_scenario["event_severity"] = "high" if event_type in {"vehicle_fire", "collision", "wrong_way_driver"} else (
                "medium" if event_type != "normal" else "low"
            )

        scenario_rows.append(base_scenario.to_frame().T)
        sensor_rows.append(aug_sensor)
        label_rows.append(aug_label)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenario_out = out / "augmented_scenarios.csv"
    sensor_out = out / "augmented_sensors.csv"
    label_out = out / "augmented_labels.csv"
    tunnel_out = out / "augmented_tunnels.csv"

    scenarios_df = pd.concat(scenario_rows, ignore_index=True)
    sensors_df = pd.concat(sensor_rows, ignore_index=True)
    labels_df = pd.concat(label_rows, ignore_index=True)

    scenarios_df.to_csv(scenario_out, index=False)
    sensors_df.to_csv(sensor_out, index=False)
    labels_df.to_csv(label_out, index=False)
    tunnels.to_csv(tunnel_out, index=False)

    manifest = {
        "n_scenarios": cfg.n_scenarios,
        "seed": cfg.seed,
        "event_probability": cfg.event_probability,
        "files": {
            "scenarios": str(scenario_out),
            "sensors": str(sensor_out),
            "labels": str(label_out),
            "tunnels": str(tunnel_out),
        },
    }
    with (out / "augmentation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Erzeuge 10.000 augmentierte Tunnel-Szenarien aus 4 Rohdaten-Dateien.")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--sensors-csv", required=True)
    ap.add_argument("--tunnel-csv", required=True)
    ap.add_argument("--scenario-csv", required=True)
    ap.add_argument("--out-dir", default="data/augmented")
    ap.add_argument("--n-scenarios", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--continuous-noise", type=float, default=0.04)
    ap.add_argument("--count-jitter", type=float, default=0.08)
    ap.add_argument("--binary-flip-prob", type=float, default=0.01)
    ap.add_argument("--event-probability", type=float, default=0.35)
    args = ap.parse_args()

    cfg = AugmentConfig(
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        continuous_noise=args.continuous_noise,
        count_jitter=args.count_jitter,
        binary_flip_prob=args.binary_flip_prob,
        event_probability=args.event_probability,
    )

    manifest = generate_augmented_dataset(
        labels_csv=args.labels_csv,
        sensors_csv=args.sensors_csv,
        tunnel_csv=args.tunnel_csv,
        scenario_csv=args.scenario_csv,
        out_dir=args.out_dir,
        cfg=cfg,
    )

    print("✅ Augmentation abgeschlossen")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
