"""sim/traffic_model.py

Minimal traffic dynamics model for TunnelAI-Lab.

Design goals:
- easy to read
- stable for long simulation runs
- captures essential nonlinear effects (free-flow vs congestion)

This is not a full microscopic traffic simulation. It is a compact, macroscopic
surrogate suitable for synthetic data generation and ML experimentation.
"""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class TrafficParams:
    """Model parameters controlling traffic behavior."""

    # Simulation time step in seconds.
    dt_s: float = 1.0

    # Effective segment length represented by this model cell.
    segment_length_km: float = 1.0

    # Free-flow speed limit (upper bound for uncongested traffic).
    v_free_kmh: float = 100.0

    # Max physically plausible density for this simplified model.
    rho_max_veh_per_km: float = 180.0

    # Shape exponent for fundamental diagram speed curve.
    gamma: float = 2.0

    # Speed relaxation rate toward desired speed.
    kappa_v: float = 0.25

    # Nominal maximum outflow capacity.
    q_max_veh_per_h: float = 3600.0

    # Critical density used in saturation shaping.
    rho_crit_veh_per_km: float = 45.0


@dataclass
class TrafficState:
    """Dynamic state variables updated every simulation step."""

    # Current density (vehicles per km)
    rho_veh_per_km: float
    # Current average speed (km/h)
    v_kmh: float
    # Memory term for persistent congestion pressure (0..1)
    queue_index: float = 0.0


def fundamental_speed(p: TrafficParams, rho: float) -> float:
    """Compute desired speed from density using a simple fundamental diagram.

    - At low density -> speed near free-flow.
    - At high density -> speed drops nonlinearly.
    """
    x = max(0.0, min(1.0, rho / p.rho_max_veh_per_km))
    return p.v_free_kmh * (1.0 - (x ** p.gamma))


def _capacity_shaped_outflow(p: TrafficParams, rho: float, speed: float, q_max: float) -> float:
    """Return outgoing flow constrained by both kinematics and capacity saturation.

    Intuition:
    - `base = rho * speed` behaves like unconstrained fundamental flow.
    - `sat` imposes a soft ceiling near critical density.
    - final outflow is the minimum of both.
    """
    base = max(0.0, rho) * max(0.0, speed)
    sat = q_max * (1.0 - math.exp(-max(0.0, rho) / max(1e-6, p.rho_crit_veh_per_km)))
    return min(base, sat)


def step_traffic(
    p: TrafficParams,
    s: TrafficState,
    q_in_veh_per_h: float,
    vms_speed_limit_kmh: float,
    capacity_factor: float = 1.0,
) -> Tuple[TrafficState, float]:
    """Advance traffic state by one time step.

    Args:
        p: static parameters
        s: current dynamic state
        q_in_veh_per_h: incoming demand
        vms_speed_limit_kmh: control-imposed speed limit
        capacity_factor: multiplicative reduction of road capacity

    Returns:
        Tuple `(state_next, q_out_veh_per_h)` with updated traffic state and
        the physically constrained outflow used in conservation update.
    """
    dt = p.dt_s
    dt_h = dt / 3600.0

    # Effective capacity considering incidents/weather.
    q_max = p.q_max_veh_per_h * max(0.1, capacity_factor)

    # Desired speed is the lower of traffic-physics and VMS control.
    v_des = min(fundamental_speed(p, s.rho_veh_per_km), vms_speed_limit_kmh)

    # Queue pressure captures sustained overload; this accelerates speed collapse
    # when inflow repeatedly exceeds capacity.
    queue_growth = max(0.0, q_in_veh_per_h - q_max) / max(1.0, p.q_max_veh_per_h)
    queue_next = min(1.0, max(0.0, 0.97 * s.queue_index + 0.08 * queue_growth))

    # Convert queue pressure into speed penalty.
    queue_drag = 1.0 - 0.45 * queue_next
    v_des *= max(0.25, queue_drag)

    # First-order relaxation toward desired speed (smooth transitions).
    v_next = s.v_kmh + p.kappa_v * (v_des - s.v_kmh) * dt

    # Compute outflow from current state and update density via conservation.
    q_out = _capacity_shaped_outflow(p, s.rho_veh_per_km, s.v_kmh, q_max)
    rho_next = s.rho_veh_per_km + (dt_h / p.segment_length_km) * (q_in_veh_per_h - q_out)

    # Clip to physically meaningful bounds.
    rho_next = max(0.0, min(p.rho_max_veh_per_km, rho_next))
    v_next = max(0.0, min(p.v_free_kmh, v_next))

    return TrafficState(rho_veh_per_km=rho_next, v_kmh=v_next, queue_index=queue_next), q_out
