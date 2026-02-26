from __future__ import annotations

import re

from fastapi import APIRouter

from .errors import api_error
from .playback_manager import PlaybackManager
from .scenario_store import ScenarioStore


SCENARIO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
ALLOWED_COMMANDS = {"play", "pause", "seek", "speed"}


def _validate_scenario_id(scenario_id: str) -> str:
    if not scenario_id or not SCENARIO_ID_PATTERN.match(scenario_id):
        api_error(400, "SC_INVALID_ID", "Invalid scenario id format.")
    return scenario_id


def _require_session(playback: PlaybackManager, session_id: str):
    try:
        return playback.get_session(session_id)
    except KeyError:
        api_error(404, "PB_SESSION_NOT_FOUND", f"Playback session '{session_id}' not found.")


def _apply_control_or_error(payload: dict, session) -> None:
    cmd = payload.get("cmd")
    if cmd not in ALLOWED_COMMANDS:
        api_error(400, "CMD_UNKNOWN", "Unknown playback command.", {"allowed": sorted(ALLOWED_COMMANDS)})

    if cmd == "pause":
        session.engine.set_paused(True)
        return

    if cmd == "play":
        session.engine.set_paused(False)
        return

    if cmd == "seek":
        if "t" not in payload:
            api_error(400, "CMD_PARAM_MISSING", "Parameter 't' required for seek command.")
        try:
            t = float(payload["t"])
        except (TypeError, ValueError):
            api_error(400, "CMD_INVALID_PARAM", "Parameter 't' must be numeric.")
        if t < 0:
            api_error(400, "CMD_INVALID_PARAM", "Seek time must be greater than or equal to zero.")
        session.engine.seek(t)
        return

    if "factor" not in payload:
        api_error(400, "CMD_PARAM_MISSING", "Parameter 'factor' required for speed command.")
    try:
        factor = float(payload["factor"])
    except (TypeError, ValueError):
        api_error(400, "CMD_INVALID_PARAM", "Speed factor must be numeric.")
    if factor <= 0:
        api_error(400, "CMD_INVALID_PARAM", "Speed factor must be greater than zero.")
    session.engine.set_speed(factor)


def build_api_router(store: ScenarioStore, playback: PlaybackManager) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/scenarios")
    def list_scenarios() -> list[dict]:
        return [scenario.model_dump() for scenario in store.list_scenarios()]

    @router.get("/scenarios/{scenario_id}/meta")
    def scenario_meta(scenario_id: str) -> dict:
        checked = _validate_scenario_id(scenario_id)
        try:
            return store.load_meta(checked)
        except FileNotFoundError:
            api_error(404, "SC_NOT_FOUND", f"Scenario '{checked}' does not exist.")

    @router.post("/playback/session")
    def create_playback_session(payload: dict) -> dict:
        scenario_id = _validate_scenario_id(str(payload.get("scenario_id", "")))
        try:
            session = playback.create_session(scenario_id)
        except FileNotFoundError:
            api_error(404, "SC_NOT_FOUND", f"Scenario '{scenario_id}' does not exist.")
        return {"session_id": session.id, **session.engine.status()}

    @router.get("/playback/session/{session_id}")
    def get_playback_session(session_id: str) -> dict:
        session = _require_session(playback, session_id)
        return {"session_id": session.id, **session.engine.status()}

    @router.post("/playback/session/{session_id}/control")
    def playback_control(session_id: str, payload: dict) -> dict:
        session = _require_session(playback, session_id)
        _apply_control_or_error(payload, session)
        return {"session_id": session.id, **session.engine.status()}

    @router.post("/playback/session/{session_id}/frame")
    def next_playback_frame(session_id: str) -> dict:
        session = _require_session(playback, session_id)
        try:
            return session.engine.step().model_dump()
        except Exception as exc:
            api_error(500, "ENG_STEP_FAILED", "Simulation step failed.", {"reason": str(exc)})

    return router
