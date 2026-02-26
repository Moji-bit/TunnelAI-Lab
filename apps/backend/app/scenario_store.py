from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ScenarioSummary


class ScenarioStore:
    def __init__(self, scenarios_dir: str | Path):
        self._scenarios_dir = Path(scenarios_dir)

    def _scenario_paths(self) -> list[Path]:
        return sorted(self._scenarios_dir.glob("*.json"))

    def list_scenarios(self) -> list[ScenarioSummary]:
        summaries: list[ScenarioSummary] = []
        for scenario_path in self._scenario_paths():
            raw = self._load_raw(scenario_path.stem)
            summaries.append(self._to_summary(raw))
        return summaries

    def load_meta(self, scenario_id: str) -> dict[str, Any]:
        raw = self._load_raw(scenario_id)
        return {
            "id": raw["scenario_id"],
            "name": raw["scenario_id"],
            "schema": "tunnelai.viz.frame.v1",
            "duration": float(raw.get("duration_s", 0)),
            "dt": 1.0,
            "tunnel_length": 1500,
            "tubes": 2,
            "lanes": 2,
            "tags": ["generated", "playback"],
            "source": raw,
        }

    def _load_raw(self, scenario_id: str) -> dict[str, Any]:
        scenario_path = self._scenarios_dir / f"{scenario_id}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario '{scenario_id}' not found")
        with scenario_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _to_summary(raw: dict[str, Any]) -> ScenarioSummary:
        return ScenarioSummary(
            id=raw["scenario_id"],
            name=raw["scenario_id"],
            duration=float(raw.get("duration_s", 0)),
            dt=1.0,
            tubes=2,
            lanes=2,
            tags=["generated", "playback"],
        )
