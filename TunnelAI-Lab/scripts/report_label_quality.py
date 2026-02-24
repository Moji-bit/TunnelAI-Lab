from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label quality report for incident labels in long-format CSV")
    p.add_argument("--csv", default="data/raw/all_runs.csv", help="Path to long-format CSV")
    p.add_argument("--h", type=int, default=60, help="Prediction horizon in seconds")
    return p.parse_args()


def load_active_series(csv_path: Path) -> dict[str, list[tuple[datetime, int]]]:
    per_scenario: dict[str, dict[datetime, int]] = defaultdict(dict)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("tag_id") != "Z3.EVT.Incident.Active":
                continue
            sid = row.get("scenario_id", "scenario_0")
            ts = datetime.fromisoformat(row["timestamp"])
            active = 1 if float(row["value"]) >= 0.5 else 0
            per_scenario[sid][ts] = active

    out: dict[str, list[tuple[datetime, int]]] = {}
    for sid, ts_map in per_scenario.items():
        out[sid] = sorted(ts_map.items(), key=lambda x: x[0])
    return out


def count_transitions(series: list[tuple[datetime, int]]) -> tuple[int, int, int]:
    if not series:
        return 0, 0, 0
    onset = 0
    offset = 0
    active_total = 0
    prev = 0
    for _, active in series:
        active_total += active
        if active == 1 and prev == 0:
            onset += 1
        if active == 0 and prev == 1:
            offset += 1
        prev = active
    return onset, offset, active_total


def horizon_positive_ratio(series: list[tuple[datetime, int]], horizon_s: int) -> float:
    if not series:
        return 0.0
    vals = [v for _, v in series]
    n = len(vals)
    if n <= horizon_s:
        return 0.0

    positives = 0
    total = 0
    for i in range(0, n - horizon_s):
        fut = vals[i + 1 : i + 1 + horizon_s]
        y = 1 if any(x >= 1 for x in fut) else 0
        positives += y
        total += 1
    return (positives / total) if total else 0.0


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parents[1] / csv_path

    data = load_active_series(csv_path)
    if not data:
        raise SystemExit(f"No Z3.EVT.Incident.Active labels found in {csv_path}")

    print("# Label Quality Report")
    print(f"csv: {csv_path}")
    print(f"horizon_H: {args.h} seconds")
    print(f"scenarios: {len(data)}")

    total_points = 0
    total_active = 0
    total_onsets = 0
    total_offsets = 0

    print("\nPer-scenario summary:")
    for sid, series in sorted(data.items()):
        onset, offset, active_total = count_transitions(series)
        ratio = horizon_positive_ratio(series, args.h)
        n = len(series)
        total_points += n
        total_active += active_total
        total_onsets += onset
        total_offsets += offset
        imbalance = (active_total / n) if n else 0.0
        print(
            f"- {sid}: n={n}, active_ratio={imbalance:.4f}, "
            f"onset={onset}, offset={offset}, horizon_pos_ratio={ratio:.4f}"
        )

    print("\nGlobal summary:")
    print(f"- total_points={total_points}")
    print(f"- active_points={total_active}")
    print(f"- active_ratio={total_active/total_points:.4f}")
    print(f"- onset_count={total_onsets}")
    print(f"- offset_count={total_offsets}")


if __name__ == "__main__":
    main()
