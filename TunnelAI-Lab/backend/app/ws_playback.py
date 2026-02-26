from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .playback_engine import PlaybackEngine
from .playback_manager import PlaybackManager


async def playback_ws(websocket: WebSocket, playback: PlaybackManager, scenario_id: str, session_id: str | None) -> None:
    await websocket.accept()
    session = playback.get_or_create(scenario_id=scenario_id, session_id=session_id)
    engine: PlaybackEngine = session.engine

    try:
        while True:
            frame = engine.step().model_dump()
            frame["session_id"] = session.id
            await websocket.send_json(frame)
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
