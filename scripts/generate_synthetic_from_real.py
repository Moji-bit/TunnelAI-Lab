from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _choose_numeric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "event_type_code",
        "scenario_id",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _inject_event_labels(df: pd.DataFrame, rng: np.random.Generator, event_rate: float) -> pd.DataFrame:
    if event_rate <= 0.0:
        return df
    if "event_type" not in df.columns:
        return df

    out = df.copy()
    if "event_type_code" not in out.columns:
        out["event_type_code"] = 0

    n = len(out)
    if n < 10:
        return out

    if rng.random() < event_rate:
        span = int(max(5, min(n // 4, rng.integers(10, 80))))
        start = int(rng.integers(0, max(1, n - span)))
        end = start + span
        event_name = str(rng.choice(["collision", "stalled_vehicle", "wrong_way_driver", "vehicle_fire"]))
        event_code_map = {"collision": 1, "stalled_vehicle": 2, "wrong_way_driver": 3, "vehicle_fire": 4}
        out.loc[out.index[start:end], "event_type"] = event_name
        out.loc[out.index[start:end], "event_type_code"] = event_code_map[event_name]
    return out


def synthesize_runs(
    input_csv: str,
    out_dir: str,
    n_samples: int,
    seed: int,
    chunk_len: int,
    noise_std: float,
    scale_jitter: float,
    missing_rate: float,
    event_injection_rate: float,
    scenario_id_col: str = "scenario_id",
) -> list[str]:
    df = pd.read_csv(input_csv)
    if len(df) < 2:
        raise ValueError("Input CSV too small.")

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    num_cols = _choose_numeric_columns(df)
    if not num_cols:
        raise ValueError("No numeric columns found for synthesis.")

    files: list[str] = []
    n = len(df)
    use_len = int(min(max(20, chunk_len), n))

    for i in range(1, n_samples + 1):
        start = int(rng.integers(0, max(1, n - use_len + 1)))
        block = df.iloc[start : start + use_len].copy().reset_index(drop=True)

        for c in num_cols:
            x = pd.to_numeric(block[c], errors="coerce").astype(float)
            col_std = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) else 0.0
            scale = float(rng.normal(1.0, scale_jitter))
            noise = rng.normal(0.0, max(1e-9, noise_std * max(col_std, 1.0)), size=len(x))
            block[c] = x * scale + noise

            if missing_rate > 0:
                miss_mask = rng.random(len(block)) < missing_rate
                block.loc[miss_mask, c] = np.nan
                block[c] = block[c].ffill().bfill()

        block = _inject_event_labels(block, rng=rng, event_rate=event_injection_rate)

        if scenario_id_col in block.columns:
            block[scenario_id_col] = f"synth_{i:05d}"
        else:
            block.insert(0, scenario_id_col, f"synth_{i:05d}")

        out_path = os.path.join(out_dir, f"synth_{i:05d}.csv")
        block.to_csv(out_path, index=False)
        files.append(out_path)

    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic scenario CSVs from real raw sensor CSV.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", default="data/raw/synth")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_len", type=int, default=600)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--scale_jitter", type=float, default=0.10)
    ap.add_argument("--missing_rate", type=float, default=0.01)
    ap.add_argument("--event_injection_rate", type=float, default=0.35)
    ap.add_argument("--manifest", default="artifacts/synth_generation_manifest.json")
    args = ap.parse_args()

    files = synthesize_runs(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        chunk_len=args.chunk_len,
        noise_std=args.noise_std,
        scale_jitter=args.scale_jitter,
        missing_rate=args.missing_rate,
        event_injection_rate=args.event_injection_rate,
    )

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "input_csv": args.input_csv,
        "out_dir": args.out_dir,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "chunk_len": args.chunk_len,
        "noise_std": args.noise_std,
        "scale_jitter": args.scale_jitter,
        "missing_rate": args.missing_rate,
        "event_injection_rate": args.event_injection_rate,
        "files": files,
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Synthetic generation finished")
    print(f"generated_files={len(files)}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
