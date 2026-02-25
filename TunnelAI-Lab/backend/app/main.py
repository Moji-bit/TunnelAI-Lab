from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket

from .api import build_api_router
from .scenario_store import ScenarioStore
from .ws_live import live_ws
from .ws_playback import playback_ws

BASE_DIR = Path(__file__).resolve().parents[2]
store = ScenarioStore(BASE_DIR / "scenarios")

app = FastAPI(title="TunnelAI-Viz Backend", version="0.1.0")
app.include_router(build_api_router(store))


@app.websocket("/ws/playback")
async def ws_playback_route(websocket: WebSocket, scenario_id: str) -> None:
    await playback_ws(websocket, store=store, scenario_id=scenario_id)


@app.websocket("/ws/live")
async def ws_live_route(websocket: WebSocket) -> None:
    await live_ws(websocket)
