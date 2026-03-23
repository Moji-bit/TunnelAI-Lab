from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def _to_float(v: str | None, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _to_int(v: str | None, default: int = 0) -> int:
    return int(round(_to_float(v, float(default))))


def _event_type_from_row(row: dict[str, str]) -> str:
    event = (row.get("event_type") or "").strip().lower()
    if event in {"accident", "collision"}:
        return "collision"
    if event in {"ghostdriver", "wrong_way_driver", "wrong-way"}:
        return "wrong_way_driver"
    if event in {"standstill", "breakdown", "stalled_vehicle"}:
        return "stalled_vehicle"
    if event in {"fire", "vehicle_fire"}:
        return "vehicle_fire"
    if event in {"none", ""}:
        return "none"

    if _to_int(row.get("event_fire"), 0) == 1:
        return "vehicle_fire"
    if _to_int(row.get("event_ghostdriver"), 0) == 1:
        return "wrong_way_driver"
    if _to_int(row.get("event_standstill"), 0) == 1 or _to_int(row.get("event_breakdown"), 0) == 1:
        return "stalled_vehicle"
    if _to_int(row.get("event_accident"), 0) == 1:
        return "collision"
    return "none"


def _incident_active(row: dict[str, str]) -> bool:
    return _event_type_from_row(row) != "none"


def infer_scenario(rows: list[dict[str, str]], scenario_id: str) -> dict[str, object]:
    if not rows:
        raise ValueError("CSV enthält keine Datenzeilen.")

    first = rows[0]
    n = len(rows)

    speed_values = [_to_float(r.get("speed_kmh"), 0.0) for r in rows]
    flow_values = [_to_float(r.get("flow_veh_h"), 0.0) for r in rows]
    occ_values = [_to_float(r.get("occupancy_pct"), 0.0) for r in rows]

    active_idx = [i for i, r in enumerate(rows) if _incident_active(r)]
    incident_start_s = active_idx[0] if active_idx else 0
    incident_end_s = active_idx[-1] + 1 if active_idx else 0

    event_types = [_event_type_from_row(r) for r in rows if _incident_active(r)]
    incident_type = event_types[0] if event_types else "collision"

    weather_active_idx = [
        i
        for i, r in enumerate(rows)
        if (r.get("weather", "clear").strip().lower() != "clear") or (_to_int(r.get("event_environmental"), 0) == 1)
    ]
    weather_start_s = weather_active_idx[0] if weather_active_idx else 0
    weather_end_s = weather_active_idx[-1] + 1 if weather_active_idx else 0
    weather_type = (
        rows[weather_start_s].get("weather", "clear").strip().lower() if weather_active_idx else "clear"
    )

    speed_limit = _to_float(first.get("speed_limit_kmh"), 80.0)
    min_speed = min(speed_values) if speed_values else speed_limit
    severity = max(0.0, min(1.0, (speed_limit - min_speed) / max(1.0, speed_limit)))

    base_flow = median(flow_values[: max(3, min(30, len(flow_values) // 3))]) if flow_values else 1800.0
    peak_amp = max(100.0, abs(max(flow_values) - min(flow_values)) / 2.0) if flow_values else 600.0
    period_s = max(600, min(3600, n))

    weather_visibility_drop_pct = max(0.0, 100.0 - _to_float(first.get("visibility_m"), 800.0) / 8.0)
    weather_speed_drop_kmh = max(0.0, speed_limit - mean(speed_values[-10:])) if speed_values else 0.0

    scenario: dict[str, object] = {
        "scenario_id": scenario_id,
        "seed": 42,
        "duration_s": n,
        "q_in_base_veh_per_h": base_flow,
        "q_in_peak_amp_veh_per_h": peak_amp,
        "q_in_peak_period_s": period_s,
        "incident_start_s": incident_start_s,
        "incident_end_s": incident_end_s,
        "incident_capacity_drop": max(0.05, min(0.8, severity * 0.7)),
        "incident_type": incident_type,
        "incident_severity": max(0.1, min(1.0, severity)),
        "weather_type": weather_type,
        "weather_start_s": weather_start_s,
        "weather_end_s": weather_end_s,
        "weather_visibility_drop_pct": weather_visibility_drop_pct if weather_active_idx else 0.0,
        "weather_speed_drop_kmh": weather_speed_drop_kmh if weather_active_idx else 0.0,
        "vms_speed_limit_kmh": speed_limit,
        "fan_stage": max(0, _to_int(first.get("jet_fan_count"), 8) // 4),
        "tunnel_length_m": _to_float(first.get("tunnel_length_m"), 1200.0),
        "tunnel_width_m": _to_float(first.get("tunnel_width_m"), 10.0),
        "clearance_height_m": _to_float(first.get("clearance_height_m"), 4.8),
        "gradient_pct": _to_float(first.get("gradient_pct"), 1.0),
        "curvature_radius_m": _to_float(first.get("curvature_radius_m"), 800.0),
        "profile": first.get("profile", "road_tunnel"),
        "tubes": _to_int(first.get("tubes"), 2),
        "lanes_per_tube": _to_int(first.get("lanes_per_tube"), 2),
        "direction_mode": first.get("direction_mode", "unidirectional"),
        "aadt": _to_float(first.get("aadt"), 30000.0),
        "heavy_vehicle_pct": _to_float(first.get("heavy_vehicle_pct"), 12.0),
        "speed_limit_kmh": speed_limit,
        "traffic_volume_pct": _to_float(first.get("traffic_volume_pct"), 100.0),
        "escape_route_spacing_m": _to_float(first.get("escape_route_spacing_m"), 300.0),
        "emergency_call_spacing_m": _to_float(first.get("emergency_call_spacing_m"), 150.0),
        "layby_spacing_m": _to_float(first.get("layby_spacing_m"), 600.0),
        "vent_system": first.get("vent_system", "longitudinal"),
        "jet_fan_count": _to_int(first.get("jet_fan_count"), 10),
        "air_velocity_ms": _to_float(first.get("air_velocity_ms"), 2.5),
        "volume_flow_m3s": _to_float(first.get("volume_flow_m3s"), 300.0),
        "altitude_m": _to_float(first.get("altitude_m"), 400.0),
        "entry_luminance_cd": _to_float(first.get("entry_luminance_cd"), 200.0),
        "interior_luminance_cd": _to_float(first.get("interior_luminance_cd"), 8.0),
        "emergency_lighting": _to_int(first.get("emergency_lighting"), 1),
        "temperature_c": _to_float(first.get("temperature_c"), _to_float(first.get("temp_c"), 18.0)),
        "weather_intensity_pct": _to_float(first.get("weather_intensity_pct"), 0.0),
        "wind_speed_ms": _to_float(first.get("wind_speed_ms"), 0.0),
    }

    # Occupancy fallback nudges severity slightly for dense stop-and-go sequences.
    if occ_values:
        occ_med = median(occ_values)
        if occ_med > 60:
            scenario["incident_severity"] = min(1.0, float(scenario["incident_severity"]) + 0.08)

    return scenario


def main() -> None:
    ap = argparse.ArgumentParser(description="Importiere externes exact-CSV und erzeuge TunnelAI-Szenario JSON.")
    ap.add_argument("--csv", required=True, help="Pfad zur externen exact CSV Datei")
    ap.add_argument("--out", required=True, help="Ausgabe-Pfad für Scenario JSON")
    ap.add_argument("--scenario-id", default=None, help="Optionaler Scenario Name (default: Dateiname)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    scenario_id = args.scenario_id or csv_path.stem
    scenario = infer_scenario(rows, scenario_id=scenario_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    print(f"✅ Szenario geschrieben: {out_path}")
    print(f"scenario_id={scenario['scenario_id']}, duration_s={scenario['duration_s']}")


if __name__ == "__main__":
    main()
