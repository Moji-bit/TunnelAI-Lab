from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streaming.run_record import load_scenario, record_to_csv


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else BASE_DIR / pp


def make_out_name(scenario_name: str, seed: int) -> str:
    stem = Path(scenario_name).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}__seed{seed}__{ts}.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Standardized batch data generation (scenario-set + seeds)")
    ap.add_argument("--plan", default="configs/data_batch_plan.json")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--limit-seeds", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan_path = resolve_path(args.plan)
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    scenario_dir = resolve_path(plan["scenario_dir"])
    out_dir = resolve_path(plan["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(plan.get("scenarios", []))
    seeds = [int(s) for s in plan.get("seeds", [42])]

    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]
    if args.limit_seeds is not None:
        seeds = seeds[: args.limit_seeds]

    start_time = plan.get("start_time_iso", "2026-01-01T08:00:00+01:00")
    max_seconds = plan.get("max_seconds", None)

    print("# Standard Batch")
    print(f"plan={plan_path}")
    print(f"scenarios={len(scenarios)}, seeds={len(seeds)}")

    for scn_name in scenarios:
        scn_path = scenario_dir / scn_name
        if not scn_path.exists():
            print(f"[warn] missing scenario: {scn_path}")
            continue

        for seed in seeds:
            out_csv = out_dir / make_out_name(scn_name, seed)
            print(f"[run] scenario={scn_name}, seed={seed} -> {out_csv.name}")

            if args.dry_run:
                continue

            scn = load_scenario(str(scn_path))
            setattr(scn, "seed", int(seed))
            record_to_csv(
                scenario=scn,
                out_csv=str(out_csv),
                start_time_iso=start_time,
                max_seconds=(int(max_seconds) if max_seconds is not None else None),
            )


if __name__ == "__main__":
    main()
