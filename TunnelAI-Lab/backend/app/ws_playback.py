from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .playback_engine import PlaybackEngine
from .scenario_store import ScenarioStore


async def playback_ws(websocket: WebSocket, store: ScenarioStore, scenario_id: str) -> None:
    await websocket.accept()
    meta = store.load_meta(scenario_id)
    engine = PlaybackEngine(meta)

    try:
        while True:
            await websocket.send_json(engine.step().model_dump())
            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                _apply_command(engine, incoming)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(max(0.01, engine.state.dt / 20))
    except WebSocketDisconnect:
        return


def _apply_command(engine: PlaybackEngine, raw_message: str) -> None:
    payload = json.loads(raw_message)
    cmd = payload.get("cmd")
    if cmd == "pause":
        engine.set_paused(True)
    elif cmd == "play":
        engine.set_paused(False)
    elif cmd == "seek":
        engine.seek(float(payload.get("t", 0.0)))
    elif cmd == "speed":
        engine.set_speed(float(payload.get("factor", 1.0)))
