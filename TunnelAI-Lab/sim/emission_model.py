# sim/emission_model.py
from dataclasses import dataclass


@dataclass
class EmissionParams:
    dt_s: float = 1.0
    air_volume_m3: float = 300_000.0

    # Ventilation
    Q0_m3_per_s: float = 60.0
    Q1_m3_per_s_per_stage: float = 40.0

    # e_CO(v) = a0 + a1/(v+delta) + a2*v
    e_a0: float = 0.6
    e_a1: float = 40.0
    e_a2: float = 0.005
    e_delta: float = 5.0

    # CO scaling to ppm/s (surrogate)
    co_scale: float = 0.002

    # Visibility proxy dynamics
    vis_beta0: float = 0.0008
    vis_beta1: float = 0.0012
    vis_beta2_incident: float = 0.02

    # first-order source inertia (delay)
    source_tau_s: float = 35.0


@dataclass
class EmissionState:
    co_ppm: float
    vis_proxy: float
    co_source_proxy: float = 0.0


def emission_per_vehicle(p: EmissionParams, v_kmh: float) -> float:
    v = max(0.0, v_kmh)
    return p.e_a0 + p.e_a1 / (v + p.e_delta) + p.e_a2 * v


def ventilation_flow(p: EmissionParams, fan_stage: int) -> float:
    return p.Q0_m3_per_s + p.Q1_m3_per_s_per_stage * max(0, fan_stage)


def step_emissions(
    p: EmissionParams,
    s: EmissionState,
    rho_veh_per_km: float,
    v_kmh: float,
    fan_stage: int,
    incident_flag: int = 0,
    incident_severity: float = 0.0,
    heavy_ratio_pct: float = 12.0,
) -> EmissionState:
    dt = p.dt_s
    V = max(1.0, p.air_volume_m3)

    heavy_factor = 1.0 + max(0.0, min(1.0, heavy_ratio_pct / 100.0)) * 1.4
    stopgo_factor = 1.0 + max(0.0, (40.0 - max(0.0, v_kmh)) / 40.0) * 1.2
    incident_factor = 1.0 + max(0.0, min(1.0, incident_severity)) * 1.5 * int(incident_flag)

    E_proxy_raw = max(0.0, rho_veh_per_km) * emission_per_vehicle(p, v_kmh) * heavy_factor * stopgo_factor * incident_factor

    # source delay / inertia
    alpha = min(1.0, dt / max(1.0, p.source_tau_s))
    source_next = s.co_source_proxy + alpha * (E_proxy_raw - s.co_source_proxy)

    Q = ventilation_flow(p, fan_stage)

    co_next = s.co_ppm + dt * (
        (p.co_scale * source_next / V) - (Q / V) * s.co_ppm
    )

    vis_next = s.vis_proxy + dt * (
        (p.vis_beta0 * source_next + p.vis_beta2_incident * int(incident_flag) * (1.0 + incident_severity))
        - (p.vis_beta1 * Q) * s.vis_proxy
    )

    co_next = max(0.0, co_next)
    vis_next = max(0.0, vis_next)
    return EmissionState(co_ppm=co_next, vis_proxy=vis_next, co_source_proxy=source_next)
