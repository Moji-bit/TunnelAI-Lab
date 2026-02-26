from __future__ import annotations

import random
from dataclasses import dataclass

from .schemas import EventState, PlaybackFrame, VehicleState


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
        self._vehicle_count = max(20, int(scenario_meta["source"].get("q_in_base_veh_per_h", 2000) // 30))

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
            vehicles=vehicles,
            events=self._events_at(t),
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
        x = (idx * 37 + base_speed * t) % self.meta["tunnel_length"]
        if self._incident_active(t) and 620 <= x <= 680 and lane == 2:
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

    def _incident_active(self, t: float) -> bool:
        source = self.meta["source"]
        return source.get("incident_start_s", 10e6) <= t <= source.get("incident_end_s", -1)

    def _events_at(self, t: float) -> list[EventState]:
        if not self._incident_active(t):
            return []
        return [
            EventState(
                id="INCIDENT_MAIN",
                type="incident",
                tube=1,
                lane=2,
                x0=620,
                x1=680,
                severity=0.8,
                active=True,
            )
        ]
