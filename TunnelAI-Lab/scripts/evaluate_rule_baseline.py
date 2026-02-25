from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.metrics import (
    mae,
    rmse,
    precision_recall_f1,
    pr_auc,
    brier_score,
    lead_time_seconds,
    bootstrap_ci,
)


def load_series(csv_path: Path):
    by_ts = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row["timestamp"]
            tag = row["tag_id"]
            val = float(row["value"])
            by_ts.setdefault(ts, {})[tag] = val
    return [by_ts[k] for k in sorted(by_ts.keys())]


def rule_score(x):
    speed = x.get("Z2.TRAF.Speed", x.get("Z2.TRAF.AGG.S01.Speed_10s", 100.0))
    density = x.get("Z2.TRAF.Density", x.get("Z2.TRAF.AGG.S01.Density_10s", 0.0))
    co = x.get("Z2.CO.S01.Value", x.get("Z2.ENV.AGG.S01.CO_10s", 0.0))

    score = 0.0
    score += max(0.0, min(1.0, (35.0 - speed) / 35.0))
    score += max(0.0, min(1.0, (density - 80.0) / 80.0))
    score += max(0.0, min(1.0, (co - 120.0) / 120.0))
    return min(1.0, score / 2.0)


def main():
    p = argparse.ArgumentParser(description="Evaluate rule baseline with thesis metrics")
    p.add_argument("--csv", default="data/raw/all_runs.csv")
    p.add_argument("--h", type=int, default=60)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parents[1] / csv_path

    series = load_series(csv_path)
    scores = [rule_score(x) for x in series]
    active = [
        1 if x.get("Z3.EVT.Incident.Active", x.get("Z2.EVENT.IncidentFlag", 0.0)) >= 0.5 else 0
        for x in series
    ]

    y_h = []
    s_h = []
    for i in range(0, max(0, len(series) - args.h)):
        future = active[i + 1 : i + 1 + args.h]
        y_h.append(1 if any(future) else 0)
        s_h.append(scores[i])

    y_pred = [1 if s >= args.threshold else 0 for s in s_h]
    evt = precision_recall_f1(y_h, y_pred)
    evt["pr_auc"] = pr_auc(y_h, s_h)
    evt["brier"] = brier_score(y_h, s_h)
    evt["lead_time_s"] = lead_time_seconds(active, scores, threshold=args.threshold, step_s=1)

    # forecast metrics: naive persistence baseline on key vars
    def persistence_metric(tag):
        vals = [x.get(tag) for x in series if tag in x]
        if len(vals) < 2:
            return (0.0, 0.0)
        y_true = vals[1:]
        y_pred = vals[:-1]
        return mae(y_true, y_pred), rmse(y_true, y_pred)

    sp_mae, sp_rmse = persistence_metric("Z2.TRAF.Speed")
    co_mae, co_rmse = persistence_metric("Z2.CO.S01.Value")
    vis_mae, vis_rmse = persistence_metric("Z2.VIS.S01.Value")

    # repeated-seed style uncertainty from per-run blocks by scenario_id if available
    # fallback: split into 5 contiguous chunks
    chunk = max(1, len(y_h) // 5)
    f1s = []
    for i in range(0, len(y_h), chunk):
        yt = y_h[i : i + chunk]
        yp = y_pred[i : i + chunk]
        if yt:
            f1s.append(precision_recall_f1(yt, yp)["f1"])
    ci_lo, ci_hi = bootstrap_ci(f1s, alpha=0.95)

    print("# Evaluation Report (Rule Baseline)")
    print(f"csv={csv_path}")
    print(f"H={args.h}, threshold={args.threshold}")
    print("\nEvent metrics:")
    print(evt)
    print("\nForecast metrics (naive persistence):")
    print({
        "speed_mae": sp_mae,
        "speed_rmse": sp_rmse,
        "co_mae": co_mae,
        "co_rmse": co_rmse,
        "vis_mae": vis_mae,
        "vis_rmse": vis_rmse,
    })
    print("\nUncertainty:")
    print({"f1_ci95": [ci_lo, ci_hi], "n_chunks": len(f1s)})


if __name__ == "__main__":
    main()
