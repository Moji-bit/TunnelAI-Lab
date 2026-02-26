from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScenarioSummary(BaseModel):
    id: str
    name: str
    duration: float
    dt: float
    tubes: int = 2
    lanes: int = 2
    tags: list[str] = Field(default_factory=list)


class VehicleState(BaseModel):
    id: str
    tube: int
    lane: int
    x: float
    v: float
    a: float
    type: Literal["car", "truck", "emergency"] = "car"


class EventState(BaseModel):
    id: str
    type: Literal["incident", "queue", "closure", "fire", "smoke"]
    tube: int
    lane: int
    x0: float
    x1: float
    severity: float = 0.0
    active: bool = True


class ActuatorState(BaseModel):
    id: str
    type: Literal["fan", "vms"]
    zone: int
    state: Literal["off", "on", "warning"] = "off"
    value: float = 0.0


class TimebaseState(BaseModel):
    t: float
    dt: float
    paused: bool
    speed_factor: float


class PlaybackFrame(BaseModel):
    schema: Literal["tunnelai.viz.frame.v1"] = "tunnelai.viz.frame.v1"
    scenario_id: str
    mode: Literal["playback", "live"]
    t: float
    dt: float
    timebase: TimebaseState
    vehicles: list[VehicleState]
    events: list[EventState]
    actuators: list[ActuatorState] = Field(default_factory=list)
