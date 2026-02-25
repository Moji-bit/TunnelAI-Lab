"""streaming/run_record.py

Command-line entry point to generate one scenario run and persist it as CSV.

In plain words:
1) load scenario configuration (JSON or defaults)
2) run simulator stream second-by-second
3) flatten snapshots into long-format rows
4) write rows into a CSV file for later analysis/training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple

# Make repository root importable when this file is executed directly.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if __package__ is None or __package__ == "":
    sys.path.insert(0, BASE_DIR)

from streaming.opcua_mock_server import generate_stream
from streaming.recorder import write_long_csv
from sim.event_generator import Scenario


def _resolve_path(path: str | None) -> str | None:
    """Resolve relative paths against repo base directory.

    This allows users to pass short paths like `data/raw/out.csv` from anywhere.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_scenario(path: str | None) -> Scenario:
    """Load a scenario from JSON file or return default `Scenario()`.

    JSON keys are expected to match dataclass field names in `Scenario`.
    """
    if path is None:
        return Scenario()

    resolved_path = _resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Scenario(**d)


def record_to_csv(
    scenario: Scenario,
    out_csv: str,
    start_time_iso: str,
    max_seconds: int | None = None,
) -> str:
    """Run stream generation and write long-format CSV.

    Output schema per row:
    - timestamp (ISO8601)
    - tag_id
    - value
    - quality
    - scenario_id

    Args:
        scenario: scenario object defining demand/incidents/weather/etc.
        out_csv: output path (relative or absolute)
        start_time_iso: wall-clock start timestamp for first sample
        max_seconds: optional truncation for quick demo runs

    Returns:
        Resolved absolute/normalized output path.
    """
    t0 = datetime.fromisoformat(start_time_iso)

    # In-memory collector for all generated tag rows.
    rows: List[Tuple[str, str, float, str, str]] = []

    # `n` counts seconds/snapshots, not number of rows.
    n = 0
    for snap in generate_stream(scenario, t0):
        ts_iso = snap.timestamp.isoformat()

        # One snapshot contains many tags; convert each tag to one CSV row.
        for tag_id, value in snap.tags.items():
            rows.append((ts_iso, tag_id, float(value), snap.quality, snap.scenario_id))

        n += 1
        if max_seconds is not None and n >= max_seconds:
            break

    resolved_out_csv = _resolve_path(out_csv) or out_csv
    os.makedirs(os.path.dirname(resolved_out_csv) or ".", exist_ok=True)
    write_long_csv(resolved_out_csv, rows)
    return resolved_out_csv


def main():
    """CLI wrapper so this module can be run directly from terminal."""
    p = argparse.ArgumentParser(
        description="TunnelAI-Lab: record mock OPC-UA stream to long-format CSV"
    )
    p.add_argument("--scenario", type=str, default=None, help="Path to scenario JSON (optional)")
    p.add_argument("--out", type=str, default="data/raw/stau_run_long.csv", help="Output CSV path")
    p.add_argument(
        "--start",
        type=str,
        default="2026-01-01T08:00:00+01:00",
        help="Start time ISO8601 with timezone",
    )
    p.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Limit recording length (seconds) for quick tests",
    )

    args = p.parse_args()

    scenario = load_scenario(args.scenario)
    out_csv = record_to_csv(
        scenario=scenario,
        out_csv=args.out,
        start_time_iso=args.start,
        max_seconds=args.max_seconds,
    )

    print("✅ Recorded stream to:", out_csv)
    print("Scenario:", scenario.scenario_id, "| duration_s =", scenario.duration_s)


if __name__ == "__main__":
    main()
