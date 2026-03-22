"""streaming/run_record.py

Command-line entry point to generate one scenario run and persist it as CSV.

In plain words:
1) load scenario configuration (JSON or defaults)
2) run simulator stream second-by-second
3) flatten snapshots into long-format rows
4) write rows into a CSV file for later analysis/training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Make repository root importable when this file is executed directly.
CORE_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
if __package__ is None or __package__ == "":
    sys.path.insert(0, BASE_DIR)

from core.streaming.opcua_mock_server import generate_stream
from core.streaming.recorder import write_long_csv
from core.sim.event_generator import Scenario


def _resolve_path(path: str | None) -> str | None:
    """Resolve relative paths against repo base directory.

    This allows users to pass short paths like `data/raw/out.csv` from anywhere.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_scenario(path: str | None) -> Scenario:
    """Load a scenario from JSON file or return default `Scenario()`.

    JSON keys are expected to match dataclass field names in `Scenario`.
    """
    if path is None:
        return Scenario()

    resolved_path = _resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Scenario(**d)


def record_to_csv(
    scenario: Scenario,
    out_csv: str,
    start_time_iso: str,
    max_seconds: int | None = None,
) -> str:
    """Run stream generation and write long-format CSV.

    Output schema per row:
    - timestamp (ISO8601)
    - tag_id
    - value
    - quality
    - scenario_id

    Args:
        scenario: scenario object defining demand/incidents/weather/etc.
        out_csv: output path (relative or absolute)
        start_time_iso: wall-clock start timestamp for first sample
        max_seconds: optional truncation for quick demo runs

    Returns:
        Resolved absolute/normalized output path.
    """
    t0 = datetime.fromisoformat(start_time_iso)

    # In-memory collector for all generated tag rows.
    rows: List[Tuple[str, str, float, str, str]] = []

    # `n` counts seconds/snapshots, not number of rows.
    n = 0
    for snap in generate_stream(scenario, t0):
        ts_iso = snap.timestamp.isoformat()

        # One snapshot contains many tags; convert each tag to one CSV row.
        for tag_id, value in snap.tags.items():
            rows.append((ts_iso, tag_id, float(value), snap.quality, snap.scenario_id))

        n += 1
        if max_seconds is not None and n >= max_seconds:
            break

    resolved_out_csv = _resolve_path(out_csv) or out_csv
    os.makedirs(os.path.dirname(resolved_out_csv) or ".", exist_ok=True)
    write_long_csv(resolved_out_csv, rows)
    return resolved_out_csv


def _risk_level(speed_kmh: float, occ_pct: float, co_ppm: float, visibility_m: float, event_type: str) -> str:
    congestion = max(0.0, min(1.0, (80.0 - speed_kmh) / 80.0)) + max(0.0, min(1.0, occ_pct / 100.0))
    air_risk = max(0.0, min(1.0, co_ppm / 150.0)) + max(0.0, min(1.0, (300.0 - visibility_m) / 300.0))
    event_bonus = 0.8 if event_type != "none" else 0.0
    score = (0.6 * congestion + 0.7 * air_risk + event_bonus) / 2.1
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def record_to_exact_csv(
    scenario: Scenario,
    out_csv: str,
    start_time_iso: str,
    max_seconds: int | None = None,
) -> str:
    """Write an exact/wide CSV with the requested parameter columns."""
    t0 = datetime.fromisoformat(start_time_iso)
    rows: List[Dict[str, object]] = []

    for n, snap in enumerate(generate_stream(scenario, t0)):
        tags = snap.tags
        incident_active = bool(tags.get("Z3.EVT.Incident.Active", 0.0) >= 0.5)
        weather_active = bool(tags.get("Z3.EVT.Weather.Active", 0.0) >= 0.5)

        event_type = scenario.incident_type if incident_active else "none"
        speed_kmh = float(tags.get("Z2.TRAF.AGG.S01.Speed_10s", tags.get("Z1.TRAF.DET.S01.Speed", 0.0)))
        flow_veh_h = float(tags.get("Z2.TRAF.AGG.S01.FlowIn_10s", tags.get("Z1.TRAF.DET.S01.FlowIn", 0.0))) * 60.0
        occupancy_pct = float(tags.get("Z1.TRAF.DET.S01.Occ", 0.0))
        co_ppm = float(tags.get("Z2.ENV.AGG.S01.CO_10s", tags.get("Z1.ENV.CO.S01.Value", 0.0)))
        temp_c = float(tags.get("Z1.ENV.TEMP.S01.Value", scenario.temperature_c))
        visibility_pct = float(tags.get("Z2.ENV.AGG.S01.VIS_10s", tags.get("Z1.ENV.VIS.S01.Value", 100.0)))
        visibility_m = max(20.0, visibility_pct / 100.0 * 400.0)
        no2_ppm = max(0.0, co_ppm * 0.32)
        pm25 = max(0.0, co_ppm * 0.18)
        humidity_pct = 45.0 + (scenario.weather_intensity_pct * 0.45 if weather_active else 0.0)
        fan_stage = float(tags.get("Z1.VENT.FAN.F01.Stage", scenario.fan_stage))
        air_velocity = max(0.1, float(scenario.air_velocity_ms) + 0.35 * fan_stage)
        air_pressure = 101325.0 * (1.0 - 2.25577e-5 * max(0.0, float(scenario.altitude_m))) ** 5.25588

        rows.append(
            {
                "tunnel_length_m": float(scenario.tunnel_length_m),
                "tunnel_width_m": float(scenario.tunnel_width_m),
                "clearance_height_m": float(scenario.clearance_height_m),
                "gradient_pct": float(scenario.gradient_pct),
                "curvature_radius_m": float(scenario.curvature_radius_m),
                "profile": scenario.profile,
                "tubes": int(scenario.tubes),
                "lanes_per_tube": int(scenario.lanes_per_tube),
                "direction_mode": scenario.direction_mode,
                "aadt": float(scenario.aadt),
                "heavy_vehicle_pct": float(scenario.heavy_vehicle_pct),
                "speed_limit_kmh": float(scenario.speed_limit_kmh),
                "traffic_volume_pct": float(scenario.traffic_volume_pct),
                "escape_route_spacing_m": float(scenario.escape_route_spacing_m),
                "emergency_call_spacing_m": float(scenario.emergency_call_spacing_m),
                "layby_spacing_m": float(scenario.layby_spacing_m),
                "vent_system": scenario.vent_system,
                "jet_fan_count": int(scenario.jet_fan_count),
                "air_velocity_ms": float(scenario.air_velocity_ms),
                "volume_flow_m3s": float(scenario.volume_flow_m3s),
                "altitude_m": float(scenario.altitude_m),
                "entry_luminance_cd": float(scenario.entry_luminance_cd),
                "interior_luminance_cd": float(scenario.interior_luminance_cd),
                "emergency_lighting": int(scenario.emergency_lighting),
                "weather": scenario.weather_type if weather_active else "clear",
                "temperature_c": float(scenario.temperature_c),
                "weather_intensity_pct": float(scenario.weather_intensity_pct if weather_active else 0.0),
                "wind_speed_ms": float(scenario.wind_speed_ms),
                "event_fire": int(event_type == "vehicle_fire"),
                "event_accident": int(event_type == "collision"),
                "event_ghostdriver": int(event_type == "wrong_way_driver"),
                "event_standstill": int(event_type == "stalled_vehicle"),
                "event_breakdown": int(event_type == "stalled_vehicle"),
                "event_environmental": int(weather_active),
                "timestamp": snap.timestamp.isoformat(),
                "speed_kmh": speed_kmh,
                "flow_veh_h": flow_veh_h,
                "occupancy_pct": occupancy_pct,
                "co_ppm": co_ppm,
                "no2_ppm": no2_ppm,
                "pm25": pm25,
                "temp_c": temp_c,
                "humidity_pct": max(0.0, min(100.0, humidity_pct)),
                "visibility_m": visibility_m,
                "air_velocity": air_velocity,
                "air_pressure": air_pressure,
                "event_type": event_type,
                "risk_level": _risk_level(speed_kmh, occupancy_pct, co_ppm, visibility_m, event_type),
            }
        )

        if max_seconds is not None and (n + 1) >= max_seconds:
            break

    resolved_out_csv = _resolve_path(out_csv) or out_csv
    os.makedirs(os.path.dirname(resolved_out_csv) or ".", exist_ok=True)
    with open(resolved_out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return resolved_out_csv


def main():
    """CLI wrapper so this module can be run directly from terminal."""
    p = argparse.ArgumentParser(
        description="TunnelAI-Lab: record mock OPC-UA stream to long-format CSV"
    )
    p.add_argument("--scenario", type=str, default=None, help="Path to scenario JSON (optional)")
    p.add_argument("--out", type=str, default="data/raw/stau_run_long.csv", help="Output CSV path")
    p.add_argument(
        "--exact-out",
        type=str,
        default=None,
        help="Optional output path for exact/wide CSV with fixed parameter columns",
    )
    p.add_argument(
        "--start",
        type=str,
        default="2026-01-01T08:00:00+01:00",
        help="Start time ISO8601 with timezone",
    )
    p.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Limit recording length (seconds) for quick tests",
    )

    args = p.parse_args()

    scenario = load_scenario(args.scenario)
    out_csv = record_to_csv(
        scenario=scenario,
        out_csv=args.out,
        start_time_iso=args.start,
        max_seconds=args.max_seconds,
    )
    exact_out_csv = None
    if args.exact_out:
        exact_out_csv = record_to_exact_csv(
            scenario=scenario,
            out_csv=args.exact_out,
            start_time_iso=args.start,
            max_seconds=args.max_seconds,
        )

    print("✅ Recorded stream to:", out_csv)
    if exact_out_csv:
        print("✅ Recorded exact CSV to:", exact_out_csv)
    print("Scenario:", scenario.scenario_id, "| duration_s =", scenario.duration_s)


if __name__ == "__main__":
    main()
