from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if __package__ is None or __package__ == "":
    sys.path.insert(0, BASE_DIR)

REQUIRED_COLUMNS = {
    "run_id",
    "seed",
    "model",
    "backbone",
    "horizon_s",
    "feature_set",
    "noise_profile",
}


@dataclass(frozen=True)
class PlanRow:
    run_id: str
    model: str
    backbone: str
    horizon_s: int
    feature_set: str
    noise_profile: str
    seed: int


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_matrix(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return rows

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Matrix missing required columns: {sorted(missing)}")
    return rows


def iter_plan_rows(rows: list[dict[str, str]]) -> Iterable[PlanRow]:
    for row in rows:
        yield PlanRow(
            run_id=str(row["run_id"]),
            model=str(row["model"]),
            backbone=str(row["backbone"]),
            horizon_s=int(row["horizon_s"]),
            feature_set=str(row["feature_set"]),
            noise_profile=str(row["noise_profile"]),
            seed=int(row["seed"]),
        )


def print_plan(rows: list[dict[str, str]]) -> None:
    plan_rows = list(iter_plan_rows(rows))
    print("# Experiment Plan")
    print(f"Total runs: {len(plan_rows)}")

    if not plan_rows:
        return

    by_horizon: dict[int, int] = {}
    for r in plan_rows:
        by_horizon[r.horizon_s] = by_horizon.get(r.horizon_s, 0) + 1

    print("Runs by horizon:")
    for h in sorted(by_horizon):
        print(f"  H={h}s -> {by_horizon[h]} runs")

    print("\nFirst runs:")
    for r in plan_rows[:10]:
        print(
            f"  {r.run_id}: model={r.model}/{r.backbone}, H={r.horizon_s}, "
            f"features={r.feature_set}, noise={r.noise_profile}, seed={r.seed}"
        )


def build_datasets(rows: list[dict[str, str]], source_csv: str, out_root: str, L: int, stride: int) -> None:
    try:
        from dataset.dataset_builder import DatasetConfig, build_npz_from_csv
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependency")
        raise RuntimeError(
            f"Dataset build requires missing dependency: {missing}. "
            "Install environment.yml first, then rerun with --mode build-datasets."
        ) from exc

    horizons = sorted({int(r["horizon_s"]) for r in rows if r.get("horizon_s")})
    if not horizons:
        print("No horizons found in matrix; skipping dataset build.")
        return

    for h in horizons:
        out_dir = os.path.join(out_root, f"H{h}")
        cfg = DatasetConfig(L=L, H=h, stride=stride)
        print(f"\n[build] H={h} -> {out_dir}")
        build_npz_from_csv(csv_path=source_csv, out_dir=out_dir, cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and pre-build datasets from thesis_experiment_matrix.csv")
    parser.add_argument("--matrix", default="thesis_experiment_matrix.csv", help="Path to matrix CSV")
    parser.add_argument("--mode", choices=["plan", "build-datasets"], default="plan")
    parser.add_argument("--csv-path", default="data/raw/all_runs.csv", help="Long-format source CSV for dataset build")
    parser.add_argument("--out-root", default="data/processed/experiments", help="Output root for generated NPZ datasets")
    parser.add_argument("--L", type=int, default=300, help="Input window length")
    parser.add_argument("--stride", type=int, default=5, help="Window stride")
    args = parser.parse_args()

    matrix_path = resolve_path(args.matrix)
    rows = load_matrix(matrix_path)

    print_plan(rows)

    if args.mode == "build-datasets":
        source_csv = resolve_path(args.csv_path)
        out_root = resolve_path(args.out_root)
        try:
            build_datasets(rows, source_csv=source_csv, out_root=out_root, L=args.L, stride=args.stride)
        except RuntimeError as exc:
            print(f"[warn] {exc}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
