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
    type: Literal["incident", "queue", "closure"]
    tube: int
    lane: int
    x0: float
    x1: float
    severity: float = 0.0
    active: bool = True


class PlaybackFrame(BaseModel):
    schema: Literal["tunnelai.viz.frame.v1"] = "tunnelai.viz.frame.v1"
    scenario_id: str
    mode: Literal["playback", "live"]
    t: float
    dt: float
    vehicles: list[VehicleState]
    events: list[EventState]
