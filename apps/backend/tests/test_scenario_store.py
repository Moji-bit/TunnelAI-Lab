from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.playback_engine import PlaybackEngine
from app.scenario_store import ScenarioStore


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
