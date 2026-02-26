from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

from .api import build_api_router
from .playback_manager import PlaybackManager
from .scenario_store import ScenarioStore
from .ws_live import live_ws
from .ws_playback import control_ws

BASE_DIR = Path(__file__).resolve().parents[2]
store = ScenarioStore(BASE_DIR / "scenarios")
playback = PlaybackManager(store)

app = FastAPI(title="TunnelAI-Viz Backend", version="0.1.0")
app.include_router(build_api_router(store, playback))


@app.websocket("/ws/playback")
async def ws_playback_route(websocket: WebSocket, scenario_id: str, session_id: str | None = None) -> None:
    await playback_ws(websocket, playback=playback, scenario_id=scenario_id, session_id=session_id)


@app.websocket("/ws/live")
async def ws_live_route(websocket: WebSocket) -> None:
    await live_ws(websocket)
