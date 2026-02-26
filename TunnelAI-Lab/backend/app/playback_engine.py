from __future__ import annotations

import random
from dataclasses import dataclass

from .schemas import ActuatorState, EventState, PlaybackFrame, TimebaseState, VehicleState


@dataclass
class PlaybackState:
    scenario_id: str
    duration_s: float
    dt: float = 1.0
    speed_factor: float = 1.0
    paused: bool = False
    t: float = 0.0


def _clamp_time(value: float, duration: float) -> float:
    return max(0.0, min(value, duration))


class PlaybackEngine:
    def __init__(self, scenario_meta: dict):
        self.meta = scenario_meta
        self.state = PlaybackState(
            scenario_id=scenario_meta["id"],
            duration_s=float(scenario_meta["duration"]),
            dt=float(scenario_meta.get("dt", 1.0)),
        )
        self._rng = random.Random(scenario_meta["id"])
        self._base_vehicle_count = max(20, int(scenario_meta["source"].get("q_in_base_veh_per_h", 2000) // 30))
        self._stau_start = float(scenario_meta["source"].get("queue_start_s", 10.0))
        self._stau_end = float(scenario_meta["source"].get("queue_end_s", 120.0))
        self._brand_start = float(scenario_meta["source"].get("fire_start_s", 45.0))
        self._brand_end = float(scenario_meta["source"].get("fire_end_s", 90.0))

    def set_paused(self, paused: bool) -> None:
        self.state.paused = paused

    def set_speed(self, factor: float) -> None:
        self.state.speed_factor = max(0.1, factor)

    def seek(self, t: float) -> None:
        self.state.t = _clamp_time(t, self.state.duration_s)

    def step(self) -> PlaybackFrame:
        if not self.state.paused:
            self.state.t = min(
                self.state.duration_s,
                self.state.t + self.state.dt * self.state.speed_factor,
            )
        return self._build_frame()

    def _build_frame(self) -> PlaybackFrame:
        t = self.state.t
        vehicles = [self._vehicle_at(idx, t) for idx in range(self._vehicle_count)]
        return PlaybackFrame(
            scenario_id=self.state.scenario_id,
            mode="playback",
            t=t,
            dt=self.state.dt,
            timebase=TimebaseState(
                t=t,
                dt=self.state.dt,
                paused=self.state.paused,
                speed_factor=self.state.speed_factor,
            ),
            vehicles=vehicles,
            events=self._events_at(t),
            actuators=self._actuators_at(t),
        )

    def status(self) -> dict[str, float | bool | str]:
        return {
            "scenario_id": self.state.scenario_id,
            "t": self.state.t,
            "duration_s": self.state.duration_s,
            "paused": self.state.paused,
            "speed_factor": self.state.speed_factor,
            "dt": self.state.dt,
        }

    def _vehicle_at(self, idx: int, t: float) -> VehicleState:
        lane = idx % 2 + 1
        tube = idx % 2 + 1
        base_speed = 18.0 + (idx % 5) * 2.0
        if self._stau_active(t):
            base_speed = max(2.0, base_speed * (0.45 + 0.1 * ((idx + int(t)) % 3)))
        x = (idx * 37 + base_speed * t) % self.meta["tunnel_length"]
        if self._brand_active(t) and 620 <= x <= 680 and lane == 2:
            base_speed = max(3.0, base_speed * 0.25)
        return VehicleState(
            id=f"V{idx:04d}",
            tube=tube,
            lane=lane,
            x=round(x, 2),
            v=round(base_speed, 2),
            a=0.0,
            type="truck" if idx % 9 == 0 else "car",
        )

    def _stau_active(self, t: float) -> bool:
        return self._stau_start <= t <= self._stau_end

    def _brand_active(self, t: float) -> bool:
        return self._brand_start <= t <= self._brand_end

    def _vehicle_count_at(self, t: float) -> int:
        if self._stau_active(t):
            return int(self._base_vehicle_count * 1.6)
        return self._base_vehicle_count

    def _events_at(self, t: float) -> list[EventState]:
        events: list[EventState] = []
        if self._stau_active(t):
            events.append(
                EventState(
                    id="STAU_MAIN",
                    type="queue",
                    tube=1,
                    lane=2,
                    x0=500,
                    x1=900,
                    severity=0.7,
                    active=True,
                )
            )
        if self._brand_active(t):
            events.extend(
                [
                    EventState(
                        id="BRAND_MAIN",
                        type="fire",
                        tube=1,
                        lane=2,
                        x0=630,
                        x1=670,
                        severity=1.0,
                        active=True,
                    ),
                    EventState(
                        id="RAUCH_MAIN",
                        type="smoke",
                        tube=1,
                        lane=2,
                        x0=580,
                        x1=760,
                        severity=0.85,
                        active=True,
                    ),
                ]
            )
        return events

    def _actuators_at(self, t: float) -> list[ActuatorState]:
        brand = self._brand_active(t)
        return [
            ActuatorState(id="FAN_Z1_A", type="fan", zone=1, state="on" if brand else "off", value=90 if brand else 0),
            ActuatorState(id="FAN_Z1_B", type="fan", zone=1, state="on" if brand else "off", value=80 if brand else 0),
            ActuatorState(id="VMS_Z1", type="vms", zone=1, state="warning" if brand else "off", value=1 if brand else 0),
        ]

    def step(self) -> PlaybackFrame:
        if not self.state.paused:
            self.state.t = min(
                self.state.duration_s,
                self.state.t + self.state.dt * self.state.speed_factor,
            )
        t = self.state.t
        self._vehicle_count = self._vehicle_count_at(t)
        return self._build_frame()
