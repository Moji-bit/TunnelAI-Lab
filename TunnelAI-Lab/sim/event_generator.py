# sim/event_generator.py
from dataclasses import dataclass
import math
from typing import Tuple

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

    vms_speed_limit_kmh: float = 80.0
    fan_stage: int = 2

def inflow(sc: Scenario, t_s: int) -> float:
    q = sc.q_in_base_veh_per_h + sc.q_in_peak_amp_veh_per_h * math.sin(2 * math.pi * t_s / sc.q_in_peak_period_s)
    return max(0.0, q)

def incident_flag(sc: Scenario, t_s: int) -> int:
    return int(sc.incident_start_s <= t_s < sc.incident_end_s)

def capacity_factor(sc: Scenario, t_s: int) -> float:
    if incident_flag(sc, t_s):
        return max(0.1, 1.0 - sc.incident_capacity_drop)
    return 1.0

def controls(sc: Scenario, t_s: int) -> Tuple[float, int]:
    # later: time varying schedules
    return sc.vms_speed_limit_kmh, sc.fan_stage