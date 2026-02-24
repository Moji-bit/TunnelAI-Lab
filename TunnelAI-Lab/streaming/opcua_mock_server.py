# streaming/opcua_mock_server.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterator, Optional

from sim.traffic_model import TrafficParams, TrafficState, step_traffic
from sim.emission_model import EmissionParams, EmissionState, step_emissions
from sim.event_generator import (
    Scenario,
    inflow,
    incident_flag,
    incident_type_code,
    weather_flag,
    weather_type_code,
    capacity_factor,
    controls,
)

import random

@dataclass(frozen=True)
class TagSnapshot:
    timestamp: datetime
    tags: Dict[str, float]
    quality: str
    scenario_id: str


# -------------------------
# Small helper functions
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def jam_index_from_speed(speed_kmh: float, freeflow_kmh: float = 100.0) -> float:
    """
    0 = free flow, 1 = severe congestion.
    Very simple proxy: JamIndex = 1 - v/v_free.
    """
    if freeflow_kmh <= 0:
        return 0.0
    return clamp(1.0 - (speed_kmh / freeflow_kmh), 0.0, 1.0)


def smoke_index(co_ppm: float, vis_pct: float, co_alarm_ppm: float = 200.0) -> float:
    """
    0..1 smoke proxy:
    - CO high increases index
    - Visibility low increases index
    """
    co_term = clamp(co_ppm / co_alarm_ppm, 0.0, 1.0)
    vis_term = clamp(1.0 - (vis_pct / 100.0), 0.0, 1.0)
    return clamp(0.6 * co_term + 0.4 * vis_term, 0.0, 1.0)


def density_to_occ_pct(rho_veh_per_km: float) -> float:
    """
    Crude occupancy proxy from density.
    You can calibrate this later.
    """
    return clamp(rho_veh_per_km / 2.5, 0.0, 100.0)


def vis_proxy_to_pct(vis_proxy_0_1: float) -> float:
    """
    Your emission model uses vis_proxy ~ 0..1 (higher = worse).
    Convert to "visibility %" 0..100 (higher = better).
    """
    return clamp((1.0 - vis_proxy_0_1) * 100.0, 0.0, 100.0)

# Noise-Generator: eine kleine Helper-Funktion
def add_noise(rng, x: float, sigma: float, lo: float | None = None, hi: float | None = None) -> float:
    y = float(x) + float(rng.gauss(0.0, sigma))
    if lo is not None:
        y = max(lo, y)
    if hi is not None:
        y = min(hi, y)
    return y

# -------------------------
# Main generator
# -------------------------
def generate_stream(
    sc: Scenario,
    t0: datetime,
    traffic_p: Optional[TrafficParams] = None,
    emis_p: Optional[EmissionParams] = None,
    segment: str = "S01",
) -> Iterator[TagSnapshot]:
    """
    Generates TagSnapshot stream compatible with tags/tags.yaml naming.

    Notes:
    - inflow(sc, t_s) returns veh/h in your event_generator.py.
    - tags.yaml defines Z1.TRAF.DET.*.Flow as veh/min.
      => we convert veh/h -> veh/min.

    Currently only emits one segment (default S01).
    Extend later to S01..S12 by looping segments and replicating logic.
    """
    traffic_p = traffic_p or TrafficParams()
    emis_p = emis_p or EmissionParams()

    # reproducible RNG (Seed)
    seed = int(getattr(sc, "seed", 42))
    rng = random.Random(seed)
    
    # Base states
    traffic = TrafficState(rho_veh_per_km=25.0, v_kmh=95.0)
    emis = EmissionState(co_ppm=10.0, vis_proxy=0.02)
    prev_inc = False

    for t_s in range(sc.duration_s):
        ts = t0 + timedelta(seconds=t_s)

        q_in_veh_per_h = float(inflow(sc, t_s))  # veh/h
        q_in_veh_per_min = q_in_veh_per_h / 60.0

        inc = bool(incident_flag(sc, t_s))
        inc_type = int(incident_type_code(sc, t_s))
        weather_active = bool(weather_flag(sc, t_s))
        weather_type = int(weather_type_code(sc, t_s))

        cap = float(capacity_factor(sc, t_s))
        vms_speed_kmh, fan_stage = controls(sc, t_s)

        traffic = step_traffic(traffic_p, traffic, q_in_veh_per_h, vms_speed_kmh, cap)
        emis = step_emissions(
            emis_p,
            emis,
            traffic.rho_veh_per_km,
            traffic.v_kmh,
            fan_stage,
            int(inc),
        )

        vis_pct = vis_proxy_to_pct(float(emis.vis_proxy))
        if weather_active:
            vis_pct = clamp(vis_pct - float(sc.weather_visibility_drop_pct), 0.0, 100.0)
        occ_pct = density_to_occ_pct(float(traffic.rho_veh_per_km))

        tags: Dict[str, float] = {}
        tags["Z0.SIM.Seed"] = float(seed)

        # -------------------------
        # Zone 1 — traffic raw detectors
        # -------------------------
        tags[f"Z1.TRAF.DET.{segment}.Flow"]  = add_noise(rng, q_in_veh_per_min, sigma=0.8, lo=0.0)
        tags[f"Z1.TRAF.DET.{segment}.Speed"] = add_noise(rng, float(traffic.v_kmh), sigma=1.5, lo=0.0, hi=140.0)
        tags[f"Z1.TRAF.DET.{segment}.Occ"]   = add_noise(rng, occ_pct, sigma=1.0, lo=0.0, hi=100.0)

        # -------------------------
        # Zone 1 — environment sensors
        # -------------------------
        tags[f"Z1.ENV.CO.{segment}.Value"]   = add_noise(rng, float(emis.co_ppm), sigma=0.8, lo=0.0)
        tags[f"Z1.ENV.VIS.{segment}.Value"]  = add_noise(rng, vis_pct, sigma=0.8, lo=0.0, hi=100.0)
        tags[f"Z1.ENV.TEMP.{segment}.Value"] = add_noise(rng, 18.0, sigma=0.2, lo=-20.0, hi=80.0)

        # -------------------------
        # Zone 1 — actuators (one fan + one VMS)
        # -------------------------
        tags["Z1.VENT.FAN.F01.Stage"] = float(fan_stage)
        tags["Z1.VENT.FAN.F01.State"] = float(1.0 if fan_stage > 0 else 0.0)
        tags["Z1.VENT.FAN.F01.Direction"] = 0.0  # placeholder enum
        tags["Z1.VENT.FAN.F01.Fault"] = 0.0      # placeholder

        tags["Z1.VMS.SIGN.V01.SpeedSet"] = float(vms_speed_kmh)
        tags["Z1.VMS.SIGN.V01.MessageCode"] = 0.0  # placeholder enum
        tags["Z1.VMS.SIGN.V01.Fault"] = 0.0        # placeholder

        # -------------------------
        # Zone 2 — aggregates (10s style, but emitted each second for demo)
        # -------------------------
        tags[f"Z2.TRAF.AGG.{segment}.Flow_10s"] = q_in_veh_per_min
        tags[f"Z2.TRAF.AGG.{segment}.Speed_10s"] = float(traffic.v_kmh)
        tags[f"Z2.TRAF.AGG.{segment}.Density_10s"] = float(traffic.rho_veh_per_km)
        tags[f"Z2.TRAF.AGG.{segment}.JamIndex"] = jam_index_from_speed(float(traffic.v_kmh), freeflow_kmh=100.0)

        tags[f"Z2.ENV.AGG.{segment}.CO_10s"] = float(emis.co_ppm)
        tags[f"Z2.ENV.AGG.{segment}.VIS_10s"] = vis_pct
        tags[f"Z2.ENV.AGG.{segment}.CO_Rate"] = 0.0   # placeholder
        tags[f"Z2.ENV.AGG.{segment}.VIS_Rate"] = 0.0  # placeholder
        tags[f"Z2.ENV.AGG.{segment}.SmokeIndex"] = smoke_index(float(emis.co_ppm), vis_pct)

        # -------------------------
        # Zone 3 — minimal event labels (extend later)
        # -------------------------
        onset = inc and (not prev_inc)
        offset = (not inc) and prev_inc

        tags["Z3.EVT.Incident.Active"] = float(1.0 if inc else 0.0)
        tags["Z3.EVT.Incident.Onset"] = float(1.0 if onset else 0.0)
        tags["Z3.EVT.Incident.Offset"] = float(1.0 if offset else 0.0)
        tags["Z3.EVT.Incident.Type"] = float(inc_type)
        tags["Z3.EVT.Incident.Severity"] = float(sc.incident_severity if inc else 0.0)
        tags["Z3.EVT.Incident.LocationSegment"] = float(int(segment[1:])) if segment.startswith("S") else 1.0

        tags["Z3.EVT.Weather.Active"] = float(1.0 if weather_active else 0.0)
        tags["Z3.EVT.Weather.Type"] = float(weather_type)

        # -------------------------
        # Output snapshot
        # -------------------------
        yield TagSnapshot(timestamp=ts, tags=tags, quality="GOOD", scenario_id=sc.scenario_id)
        prev_inc = inc