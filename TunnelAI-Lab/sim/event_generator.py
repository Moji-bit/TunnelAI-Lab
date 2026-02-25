# sim/event_generator.py
from dataclasses import dataclass
import math
from typing import Tuple


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
    scenario_id: str = "A_stau_peak_incident_v1"
    seed: int = 42
    duration_s: int = 7200

    q_in_base_veh_per_h: float = 2200.0
    q_in_peak_amp_veh_per_h: float = 1200.0
    q_in_peak_period_s: int = 1800

    incident_start_s: int = 2400
    incident_end_s: int = 3600
    incident_capacity_drop: float = 0.35
    incident_type: str = "collision"
    incident_severity: float = 0.60  # 0..1

    weather_type: str = "clear"
    weather_start_s: int = 0
    weather_end_s: int = 0
    weather_visibility_drop_pct: float = 0.0
    weather_speed_drop_kmh: float = 0.0

    vms_speed_limit_kmh: float = 80.0
    fan_stage: int = 2


def inflow(sc: Scenario, t_s: int) -> float:
    q = sc.q_in_base_veh_per_h + sc.q_in_peak_amp_veh_per_h * math.sin(2 * math.pi * t_s / sc.q_in_peak_period_s)
    return max(0.0, q)


def incident_flag(sc: Scenario, t_s: int) -> int:
    return int(sc.incident_start_s <= t_s < sc.incident_end_s)


def incident_type_code(sc: Scenario, t_s: int) -> int:
    if not incident_flag(sc, t_s):
        return 0
    return EVENT_TYPE_TO_CODE.get(sc.incident_type, EVENT_TYPE_TO_CODE["collision"])


def weather_flag(sc: Scenario, t_s: int) -> int:
    return int(sc.weather_start_s <= t_s < sc.weather_end_s)


def weather_type_code(sc: Scenario, t_s: int) -> int:
    if not weather_flag(sc, t_s):
        return 0
    return WEATHER_TO_CODE.get(sc.weather_type, WEATHER_TO_CODE["clear"])


def capacity_factor(sc: Scenario, t_s: int) -> float:
    factor = 1.0
    if incident_flag(sc, t_s):
        severity = min(1.0, max(0.0, float(sc.incident_severity)))
        factor *= max(0.1, 1.0 - sc.incident_capacity_drop * (0.5 + 0.5 * severity))
    if weather_flag(sc, t_s):
        factor *= 0.92
    return max(0.1, factor)


def controls(sc: Scenario, t_s: int) -> Tuple[float, int]:
    # later: time varying schedules
    vms = sc.vms_speed_limit_kmh
    if weather_flag(sc, t_s):
        vms = max(40.0, vms - sc.weather_speed_drop_kmh)
    return vms, sc.fan_stage
