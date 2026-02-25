from pathlib import Path

from backend.app.playback_engine import PlaybackEngine
from backend.app.scenario_store import ScenarioStore


ROOT = Path(__file__).resolve().parents[2]


def test_list_scenarios_not_empty() -> None:
    store = ScenarioStore(ROOT / "scenarios")
    scenarios = store.list_scenarios()
    assert scenarios
    assert scenarios[0].id.startswith("stau_case_")


def test_playback_frame_schema() -> None:
    store = ScenarioStore(ROOT / "scenarios")
    meta = store.load_meta("stau_case_00")
    engine = PlaybackEngine(meta)
    frame = engine.step()
    assert frame.schema == "tunnelai.viz.frame.v1"
    assert frame.mode == "playback"
    assert frame.vehicles
