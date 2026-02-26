from __future__ import annotations

from fastapi import APIRouter

from .playback_manager import PlaybackManager
from .scenario_store import ScenarioStore


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
        return store.load_meta(scenario_id)

    @router.post("/playback/session")
    def create_playback_session(payload: dict) -> dict:
        scenario_id = str(payload.get("scenario_id", ""))
        session = playback.create_session(scenario_id)
        return {"session_id": session.id, **session.engine.status()}

    @router.get("/playback/session/{session_id}")
    def get_playback_session(session_id: str) -> dict:
        session = playback.get_session(session_id)
        return {"session_id": session.id, **session.engine.status()}

    @router.post("/playback/session/{session_id}/control")
    def playback_control(session_id: str, payload: dict) -> dict:
        session = playback.get_session(session_id)
        cmd = payload.get("cmd")
        if cmd == "pause":
            session.engine.set_paused(True)
        elif cmd == "play":
            session.engine.set_paused(False)
        elif cmd == "seek":
            session.engine.seek(float(payload.get("t", 0.0)))
        elif cmd == "speed":
            session.engine.set_speed(float(payload.get("factor", 1.0)))
        return {"session_id": session.id, **session.engine.status()}

    @router.post("/playback/session/{session_id}/frame")
    def next_playback_frame(session_id: str) -> dict:
        session = playback.get_session(session_id)
        return session.engine.step().model_dump()

    return router
