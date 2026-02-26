# streaming/run_batch_record.py
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if __package__ is None or __package__ == "":
    sys.path.insert(0, BASE_DIR)

from streaming.run_record import record_to_csv, load_scenario


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def run_all_scenarios(
    scenario_dir="scenarios",
    out_dir="data/raw",
    start_time="2026-01-01T08:00:00+01:00",
):
    resolved_scenario_dir = _resolve_path(scenario_dir)
    resolved_out_dir = _resolve_path(out_dir)

    os.makedirs(resolved_out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(resolved_scenario_dir, "*.json")))

    for path in files:
        sc = load_scenario(path)
        out_csv = os.path.join(resolved_out_dir, f"{sc.scenario_id}.csv")
        record_to_csv(sc, out_csv, start_time)
        print("✔ Recorded:", out_csv)


if __name__ == "__main__":
    run_all_scenarios()
