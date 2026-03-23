from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXCLUDE_FILES = {"all_runs.csv", "augmented_labels.csv", "augmented_sensors.csv", "augmented_scenarios.csv"}


def merge_all_csv(raw_dir: str = "data/raw", out_path: str = "data/raw/all_runs.csv") -> str:
    """Merge long-format raw CSV files into one dataset.

    Rules:
    - include only CSV files with columns: timestamp, tag_id, value
    - skip known generated files to avoid recursive/self merges
    """
    root = Path(raw_dir)
    files = sorted([p for p in root.glob("*.csv") if p.name not in EXCLUDE_FILES])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    chunks: list[pd.DataFrame] = []
    skipped: list[str] = []
    required = {"timestamp", "tag_id", "value"}

    for fp in files:
        df = pd.read_csv(fp)
        if not required.issubset(df.columns):
            skipped.append(fp.name)
            continue
        if "scenario_id" not in df.columns:
            df["scenario_id"] = fp.stem
        chunks.append(df)

    if skipped:
        print(f"[merge_csv] skipped non-long CSV files: {', '.join(skipped)}")
    if not chunks:
        raise RuntimeError("No valid long-format CSV files found to merge.")

    merged = pd.concat(chunks, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"✅ Merged {len(chunks)} files -> {out}")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge long-format raw CSV files into one all_runs.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-path", default="data/raw/all_runs.csv")
    args = parser.parse_args()
    merge_all_csv(raw_dir=args.raw_dir, out_path=args.out_path)
