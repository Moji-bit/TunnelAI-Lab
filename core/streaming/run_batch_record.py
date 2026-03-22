# streaming/run_batch_record.py
from __future__ import annotations

import argparse
import os
import random
import tempfile
from collections import Counter
from typing import Dict, List

CORE_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, BASE_DIR)

from core.streaming.run_record import record_to_csv, load_scenario


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _parse_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _balanced_scenarios(
    n_runs_per_scenario_type: int,
    random_seed: int,
) -> List[object]:
    """Create balanced in-memory scenarios across requested event classes.

    Uses `generate_random_scenarios` as base generator and then enforces class labels.
    """
    total = n_runs_per_scenario_type * len(EVENT_CLASSES)

    with tempfile.TemporaryDirectory(prefix="tunnelai_batch_scenarios_") as tmp:
        paths = generate_random_scenarios(out_dir=tmp, n=total, seed=random_seed)
        scenarios = [load_scenario(p) for p in paths]

    rng = random.Random(random_seed)
    by_class: Dict[str, List[object]] = {k: [] for k in EVENT_CLASSES}

    for idx, sc in enumerate(scenarios):
        cls = EVENT_CLASSES[idx % len(EVENT_CLASSES)]
        sc.scenario_id = f"scenario_{idx + 1:04d}"

        # Keep diverse weather profiles from base generator.
        if cls == "none":
            sc.incident_type = "collision"  # unused while incident inactive
            sc.incident_start_s = sc.duration_s + 10
            sc.incident_end_s = sc.duration_s + 20
            sc.incident_capacity_drop = 0.0
            sc.incident_severity = 0.0
        else:
            sc.incident_type = cls
            # Randomize incident window but keep valid and non-trivial.
            start_lo = max(60, int(sc.duration_s * 0.15))
            start_hi = max(start_lo + 1, int(sc.duration_s * 0.55))
            sc.incident_start_s = rng.randint(start_lo, start_hi)
            end_lo = min(sc.duration_s - 1, sc.incident_start_s + 120)
            end_hi = min(sc.duration_s - 1, sc.incident_start_s + int(sc.duration_s * 0.35))
            sc.incident_end_s = max(end_lo, end_hi)
            sc.incident_capacity_drop = max(0.1, float(sc.incident_capacity_drop))
            sc.incident_severity = max(0.2, float(sc.incident_severity))

        by_class[cls].append(sc)

    # Guarantee exact balance by truncation/sampling per class.
    balanced: List[object] = []
    for cls in EVENT_CLASSES:
        cls_items = by_class[cls]
        if len(cls_items) < n_runs_per_scenario_type:
            raise RuntimeError(f"Not enough generated scenarios for class={cls}: {len(cls_items)}")
        balanced.extend(cls_items[:n_runs_per_scenario_type])

    rng.shuffle(balanced)
    return balanced


def run_all_scenarios(
    n_runs_per_scenario_type: int = 20,
    output_dir: str = "data/raw",
    random_seed: int = 42,
    exact_format: bool = True,
    start_time: str = "2026-01-01T08:00:00+01:00",
    max_seconds: int | None = None,
) -> List[str]:
    """Generate balanced bulk datasets for ML training.

    Output files:
      data/raw/scenario_0001.csv
      data/raw/scenario_0002.csv
      ...
    """
    out_dir = _resolve_path(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    scenarios = _balanced_scenarios(
        n_runs_per_scenario_type=n_runs_per_scenario_type,
        random_seed=random_seed,
    )

    class_counter: Counter[str] = Counter()
    out_files: List[str] = []

    for i, sc in enumerate(scenarios, start=1):
        cls = sc.incident_type if sc.incident_start_s < sc.duration_s else "none"
        class_counter[cls] += 1

        out_csv = os.path.join(out_dir, f"scenario_{i:04d}.csv")
        if exact_format:
            out_path = record_to_exact_csv(
                scenario=sc,
                out_csv=out_csv,
                start_time_iso=start_time,
                max_seconds=max_seconds,
            )
        else:
            out_path = record_to_csv(
                scenario=sc,
                out_csv=out_csv,
                start_time_iso=start_time,
                max_seconds=max_seconds,
            )
        out_files.append(out_path)

    print("✅ Batch generation finished")
    print(f"scenarios_total={len(scenarios)}")
    print("class_distribution=")
    for cls in EVENT_CLASSES:
        print(f"  {cls}: {class_counter.get(cls, 0)}")
    print(f"output_dir={out_dir}")

    return out_files


def main() -> None:
    p = argparse.ArgumentParser(description="Generate large balanced scenario datasets for ML.")
    p.add_argument("--n-runs-per-scenario-type", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="data/raw")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--exact-format", type=_parse_bool, default=True)
    p.add_argument("--start-time", type=str, default="2026-01-01T08:00:00+01:00")
    p.add_argument("--max-seconds", type=int, default=None)
    args = p.parse_args()

    run_all_scenarios(
        n_runs_per_scenario_type=args.n_runs_per_scenario_type,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        exact_format=args.exact_format,
        start_time=args.start_time,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
