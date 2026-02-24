# streaming/run_batch_record.py
import glob
import os
from datetime import datetime

from streaming.run_record import record_to_csv, load_scenario


def run_all_scenarios(
    scenario_dir="scenarios",
    out_dir="data/raw",
    start_time="2026-01-01T08:00:00+01:00",
):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(scenario_dir, "*.json")))

    for path in files:
        sc = load_scenario(path)
        out_csv = os.path.join(out_dir, f"{sc.scenario_id}.csv")
        record_to_csv(sc, out_csv, start_time)
        print("✔ Recorded:", out_csv)


if __name__ == "__main__":
    run_all_scenarios()