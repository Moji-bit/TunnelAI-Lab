"""sim/event_generator.py

Human-friendly scenario helpers for the TunnelAI-Lab simulator.

This module answers a simple question:
"Given a scenario definition and a time `t` (in seconds), what is happening now?"

It provides deterministic helper functions used by the streaming/simulation layer:
- traffic inflow curve
- incident on/off flags and codes
- weather on/off flags and codes
- effective capacity reduction
- control set-points (e.g., VMS speed)

The goal is readability and reproducibility rather than perfect physical realism.
"""

from dataclasses import dataclass
import math
from typing import Tuple


# -----------------------------------------------------------------------------
# Public lookup tables
# -----------------------------------------------------------------------------
# These mappings are used to convert human-readable labels into numeric codes
# that are easy to log, store, and later use for model training/evaluation.
EVENT_TYPE_TO_CODE = {
    "none": 0,
    "collision": 1,
    "stalled_vehicle": 2,
    "wrong_way_driver": 3,
    "vehicle_fire": 4,
    "flooding": 5,
}

WEATHER_TO_CODE = {
    "clear": 0,
    "rain": 1,
    "fog": 2,
    "snow": 3,
}


@dataclass
class Scenario:
    """Single scenario configuration for a full simulation run.

    Think of this object as a "recipe" for one experiment.
    All downstream functions read values from this dataclass.

    Attributes are intentionally explicit and verbose so non-experts can map them
    to domain concepts quickly (traffic demand, incident timing, weather, controls).
    """

    # Identifier shown in logs/CSV so each run can be traced later.
    scenario_id: str = "A_stau_peak_incident_v1"

    # Random seed used by other modules (e.g., noise generation) for reproducibility.
    seed: int = 42

    # Total simulation duration in seconds.
    duration_s: int = 7200

    # -----------------------------
    # Traffic demand profile
    # -----------------------------
    # Baseline inflow in vehicles per hour.
    q_in_base_veh_per_h: float = 2200.0
    # Peak amplitude for sinusoidal demand fluctuations.
    q_in_peak_amp_veh_per_h: float = 1200.0
    # Period of the sinusoidal demand wave in seconds.
    q_in_peak_period_s: int = 1800

    # -----------------------------
    # Incident configuration
    # -----------------------------
    incident_start_s: int = 2400
    incident_end_s: int = 3600
    # Capacity drop intensity (before severity modulation).
    incident_capacity_drop: float = 0.35
    incident_type: str = "collision"
    # Normalized severity in [0..1], used by coupled models.
    incident_severity: float = 0.60

    # -----------------------------
    # Weather configuration
    # -----------------------------
    weather_type: str = "clear"
    weather_start_s: int = 0
    weather_end_s: int = 0
    weather_visibility_drop_pct: float = 0.0
    weather_speed_drop_kmh: float = 0.0

    # -----------------------------
    # Operator / SCADA controls
    # -----------------------------
    vms_speed_limit_kmh: float = 80.0
    fan_stage: int = 2


def inflow(sc: Scenario, t_s: int) -> float:
    """Return input traffic demand at time `t_s` (veh/h).

    We model demand as a sinusoidal baseline + peak component. This is a compact
    way to produce repeated high/low load phases without hardcoding many values.
    """
    q = sc.q_in_base_veh_per_h + sc.q_in_peak_amp_veh_per_h * math.sin(
        2 * math.pi * t_s / sc.q_in_peak_period_s
    )
    # Never allow negative demand.
    return max(0.0, q)


def incident_flag(sc: Scenario, t_s: int) -> int:
    """Return 1 while incident window is active, else 0."""
    return int(sc.incident_start_s <= t_s < sc.incident_end_s)


def incident_type_code(sc: Scenario, t_s: int) -> int:
    """Return numeric incident type code when incident is active, otherwise 0."""
    if not incident_flag(sc, t_s):
        return 0
    # Fallback to "collision" if scenario string is unknown.
    return EVENT_TYPE_TO_CODE.get(sc.incident_type, EVENT_TYPE_TO_CODE["collision"])


def weather_flag(sc: Scenario, t_s: int) -> int:
    """Return 1 while configured weather window is active, else 0."""
    return int(sc.weather_start_s <= t_s < sc.weather_end_s)


def weather_type_code(sc: Scenario, t_s: int) -> int:
    """Return numeric weather code when weather event is active, otherwise 0."""
    if not weather_flag(sc, t_s):
        return 0
    # Fallback to "clear" code if unknown weather label is provided.
    return WEATHER_TO_CODE.get(sc.weather_type, WEATHER_TO_CODE["clear"])


def capacity_factor(sc: Scenario, t_s: int) -> float:
    """Return multiplicative road capacity factor in [0.1..1.0].

    1.0 means no reduction.
    Lower values indicate effective capacity loss due to incident and/or weather.
    """
    factor = 1.0

    # Incident reduces capacity. Severity scales the drop.
    if incident_flag(sc, t_s):
        severity = min(1.0, max(0.0, float(sc.incident_severity)))
        factor *= max(0.1, 1.0 - sc.incident_capacity_drop * (0.5 + 0.5 * severity))

    # Weather causes an additional mild reduction.
    if weather_flag(sc, t_s):
        factor *= 0.92

    # Safety floor: never return 0 to avoid division-by-zero / dead dynamics.
    return max(0.1, factor)


def controls(sc: Scenario, t_s: int) -> Tuple[float, int]:
    """Return current (VMS speed limit, fan stage) control set-points.

    Right now this is simple and deterministic.
    Future extension point: time-varying control schedules or rule-based control.
    """
    vms = sc.vms_speed_limit_kmh
    if weather_flag(sc, t_s):
        # During weather events, reduce speed set-point for safety.
        vms = max(40.0, vms - sc.weather_speed_drop_kmh)
    return vms, sc.fan_stage
