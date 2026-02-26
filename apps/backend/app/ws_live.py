from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .schemas import PlaybackFrame, VehicleState


async def live_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    t = 0.0
    dt = 0.2
    start = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - start
            t = max(t + dt, elapsed)
            frame = PlaybackFrame(
                scenario_id="LIVE_STREAM",
                mode="live",
                t=t,
                dt=dt,
                vehicles=[
                    VehicleState(id="LIVE_001", tube=1, lane=1, x=(t * 25) % 1500, v=25.0, a=0.0),
                    VehicleState(id="LIVE_002", tube=2, lane=2, x=(t * 17 + 300) % 1500, v=17.0, a=0.0, type="truck"),
                ],
                events=[],
            )
            await websocket.send_json(frame.model_dump())
            await asyncio.sleep(dt)
    except WebSocketDisconnect:
        return
