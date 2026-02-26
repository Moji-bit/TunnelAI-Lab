from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

from .api import build_api_router
from .errors import build_error
from .playback_manager import PlaybackManager
from .scenario_store import ScenarioStore
from .ws_live import live_ws
from .ws_playback import control_ws

BASE_DIR = Path(__file__).resolve().parents[2]
store = ScenarioStore(BASE_DIR / "scenarios")
playback = PlaybackManager(store)

app = FastAPI(title="TunnelAI-Viz Backend", version="0.1.0")
app.include_router(build_api_router(store, playback))


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        payload = exc.detail
    else:
        payload = build_error("VAL_BAD_REQUEST", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=build_error("ENG_INTERNAL_ERROR", "Unexpected server error.", {"reason": str(exc)}),
    )


@app.websocket("/ws")
async def ws_control_route(websocket: WebSocket, scenario_id: str, session_id: str | None = None) -> None:
    await control_ws(websocket, playback=playback, scenario_id=scenario_id, session_id=session_id)


@app.websocket("/ws/live")
async def ws_live_route(websocket: WebSocket) -> None:
    await live_ws(websocket)
