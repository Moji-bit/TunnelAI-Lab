from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from .playback_engine import PlaybackEngine
from .scenario_store import ScenarioStore


@dataclass
class PlaybackSession:
    id: str
    scenario_id: str
    engine: PlaybackEngine


class PlaybackManager:
    def __init__(self, store: ScenarioStore):
        self._store = store
        self._sessions: dict[str, PlaybackSession] = {}
        self._lock = threading.Lock()

    def create_session(self, scenario_id: str) -> PlaybackSession:
        meta = self._store.load_meta(scenario_id)
        session = PlaybackSession(id=uuid.uuid4().hex[:12], scenario_id=scenario_id, engine=PlaybackEngine(meta))
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> PlaybackSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown playback session '{session_id}'")
        return session

    def get_or_create(self, scenario_id: str, session_id: str | None) -> PlaybackSession:
        if session_id:
            return self.get_session(session_id)
        return self.create_session(scenario_id)
