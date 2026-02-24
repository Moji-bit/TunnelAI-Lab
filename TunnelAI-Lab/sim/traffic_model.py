# sim/traffic_model.py
from dataclasses import dataclass
import math

@dataclass
class TrafficParams:
    dt_s: float = 1.0
    segment_length_km: float = 1.0
    v_free_kmh: float = 100.0
    rho_max_veh_per_km: float = 180.0
    gamma: float = 2.0
    kappa_v: float = 0.25           # relaxation rate
    q_max_veh_per_h: float = 3600.0 # nominal outflow capacity

@dataclass
class TrafficState:
    rho_veh_per_km: float
    v_kmh: float

def fundamental_speed(p: TrafficParams, rho: float) -> float:
    x = max(0.0, min(1.0, rho / p.rho_max_veh_per_km))
    return p.v_free_kmh * (1.0 - (x ** p.gamma))

def step_traffic(
    p: TrafficParams,
    s: TrafficState,
    q_in_veh_per_h: float,
    vms_speed_limit_kmh: float,
    capacity_factor: float = 1.0,
) -> TrafficState:
    dt = p.dt_s
    dt_h = dt / 3600.0

    q_max = p.q_max_veh_per_h * max(0.1, capacity_factor)

    v_des = min(fundamental_speed(p, s.rho_veh_per_km), vms_speed_limit_kmh)
    v_next = s.v_kmh + p.kappa_v * (v_des - s.v_kmh) * dt

    q_out = min(s.rho_veh_per_km * max(0.0, s.v_kmh), q_max)

    rho_next = s.rho_veh_per_km + (dt_h / p.segment_length_km) * (q_in_veh_per_h - q_out)

    rho_next = max(0.0, min(p.rho_max_veh_per_km, rho_next))
    v_next = max(0.0, min(p.v_free_kmh, v_next))
    return TrafficState(rho_veh_per_km=rho_next, v_kmh=v_next)