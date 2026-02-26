from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .errors import build_error
from .playback_engine import PlaybackEngine
from .playback_manager import PlaybackManager


async def control_ws(websocket: WebSocket, playback: PlaybackManager, scenario_id: str, session_id: str | None) -> None:
    await websocket.accept()
    try:
        session = playback.get_or_create(scenario_id=scenario_id, session_id=session_id)
    except FileNotFoundError:
        await websocket.send_json({"type": "ERROR", **build_error("SC_NOT_FOUND", f"Scenario '{scenario_id}' does not exist.")})
        await websocket.close(code=1008)
        return
    except KeyError:
        await websocket.send_json({"type": "ERROR", **build_error("PB_SESSION_NOT_FOUND", f"Playback session '{session_id}' not found.")})
        await websocket.close(code=1008)
        return

    engine: PlaybackEngine = session.engine

    try:
        while True:
            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                ok, err = _apply_control(engine, incoming)
                if not ok and err is not None:
                    await websocket.send_json({"type": "ERROR", **err})
            except asyncio.TimeoutError:
                pass

            frame = engine.step().model_dump()
            await websocket.send_json({"type": "TICK", "session_id": session.id, "payload": frame})
            await asyncio.sleep(max(0.03, engine.state.dt / 20))
    except WebSocketDisconnect:
        return


def _apply_control(engine: PlaybackEngine, raw_message: str) -> tuple[bool, dict | None]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return False, build_error("VAL_BAD_JSON", "Invalid JSON message.")

    if payload.get("type") != "CONTROL":
        return False, build_error("CMD_UNKNOWN", "Unknown websocket message type.", {"expected": "CONTROL"})

    data = payload.get("payload", {})
    cmd = data.get("cmd")
    if cmd not in {"pause", "play", "seek", "speed"}:
        return False, build_error("CMD_UNKNOWN", "Unknown playback command.", {"allowed": ["play", "pause", "seek", "speed"]})

    if cmd == "pause":
        engine.set_paused(True)
        return True, None
    elif cmd == "play":
        engine.set_paused(False)
        return True, None
    elif cmd == "seek":
        if "t" not in data:
            return False, build_error("CMD_PARAM_MISSING", "Parameter 't' required for seek command.")
        try:
            t = float(data.get("t"))
        except (TypeError, ValueError):
            return False, build_error("CMD_INVALID_PARAM", "Parameter 't' must be numeric.")
        if t < 0:
            return False, build_error("CMD_INVALID_PARAM", "Seek time must be greater than or equal to zero.")
        engine.seek(t)
        return True, None
    elif cmd == "speed":
        if "factor" not in data:
            return False, build_error("CMD_PARAM_MISSING", "Parameter 'factor' required for speed command.")
        try:
            factor = float(data.get("factor"))
        except (TypeError, ValueError):
            return False, build_error("CMD_INVALID_PARAM", "Speed factor must be numeric.")
        if factor <= 0:
            return False, build_error("CMD_INVALID_PARAM", "Speed factor must be greater than zero.")
        engine.set_speed(factor)
        return True, None

    return False, build_error("CMD_UNKNOWN", "Unknown playback command.")
