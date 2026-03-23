from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.dataset.dataset_builder import DatasetConfig, make_windows_wide_csv
from core.dataset.torch_data import StandardScaler
from core.sim.scenario_generator import generate_random_scenarios
from core.streaming.run_record import load_scenario, record_to_exact_csv


class LSTMClassifier(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_layers: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_layers: int, n_heads: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        h = self.encoder(z)
        return self.head(h[:, -1, :])


def _pick_scenario(tmp_dir: Path, seed: int) -> Path:
    paths = generate_random_scenarios(out_dir=str(tmp_dir), n=1, seed=seed)
    return Path(paths[0])


def _detection_delay_seconds(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray, normal_id: int = 0) -> float:
    true_idx = np.where(y_true != normal_id)[0]
    pred_idx = np.where(y_pred != normal_id)[0]
    if len(true_idx) == 0:
        return 0.0
    if len(pred_idx) == 0:
        return float("inf")

    t_true = int(timestamps[true_idx[0]])
    t_pred = int(timestamps[pred_idx[0]])

    # timestamps are stored as int64 ns in wide_csv pipeline
    delay_s = (t_pred - t_true) / 1_000_000_000.0
    return float(delay_s)


def _accuracy_over_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    correct = (y_true == y_pred).astype(np.float32)
    return np.cumsum(correct) / np.arange(1, len(correct) + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run trained model on a freshly simulated scenario timeline")
    ap.add_argument("--model", default="artifacts/best_model.pt")
    ap.add_argument("--scenario", default=None, help="Optional scenario JSON path; if missing, generate one")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=int, default=600)
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--start-time", default="2026-01-01T08:00:00+01:00")
    ap.add_argument("--out-dir", default="artifacts/sim_eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate / load scenario
    if args.scenario:
        scenario_path = Path(args.scenario)
    else:
        scenario_path = _pick_scenario(out_dir, seed=args.seed)

    scenario = load_scenario(str(scenario_path))

    # 2) Run simulation -> exact wide CSV
    csv_path = out_dir / "simulation.csv"
    record_to_exact_csv(
        scenario=scenario,
        out_csv=str(csv_path),
        start_time_iso=args.start_time,
        max_seconds=args.max_seconds,
    )

    # 3) Convert to windows
    df = pd.read_csv(csv_path)
    cfg = DatasetConfig(
        input_format="wide_csv",
        L=args.L,
        H=1,
        stride=args.stride,
    )
    data = make_windows_wide_csv(df, cfg)
    X = data["X"].astype(np.float32)
    y_true = data["Y_event_cls"].astype(np.int64)
    timestamps = data["meta"]["timestamps"].astype(np.int64)
    class_names = [str(x) for x in data["meta"].get("event_class_names", np.array([], dtype=str)).tolist()]

    # 4) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    backbone = ckpt.get("backbone", "lstm")
    d_in = int(ckpt.get("d_in", X.shape[-1]))
    n_classes = int(ckpt.get("n_classes", len(class_names) if class_names else 2))

    if backbone == "transformer":
        model = TransformerClassifier(d_in=d_in, d_model=128, n_layers=2, n_heads=4, n_classes=n_classes)
    else:
        model = LSTMClassifier(d_in=d_in, d_model=128, n_layers=2, n_classes=n_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Scale with statistics from this simulation (best effort if train scaler is unavailable)
    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X).astype(np.float32)

    # 5) Predict
    with torch.no_grad():
        logits = model(torch.from_numpy(Xn).to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

    # 6) Output predicted vs true timeline
    timeline_path = out_dir / "pred_vs_true_timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "y_true", "y_pred", "y_true_name", "y_pred_name"])
        for ts, yt, yp in zip(timestamps, y_true, y_pred):
            yt_name = class_names[yt] if yt < len(class_names) else str(yt)
            yp_name = class_names[yp] if yp < len(class_names) else str(yp)
            w.writerow([int(ts), int(yt), int(yp), yt_name, yp_name])

    # 7) Compute metrics
    delay_s = _detection_delay_seconds(y_true, y_pred, timestamps=timestamps, normal_id=0)
    acc_t = _accuracy_over_time(y_true, y_pred)
    final_acc = float(acc_t[-1]) if len(acc_t) else 0.0

    print("=== Simulation Test ===")
    print(f"scenario: {scenario_path}")
    print(f"simulation_csv: {csv_path}")
    print(f"windows: X={X.shape}, y={y_true.shape}")
    print(f"timeline_csv: {timeline_path}")
    print(f"detection_delay_s: {delay_s}")
    print(f"accuracy_over_time_final: {final_acc:.4f}")

    # Save summary JSON
    summary = {
        "scenario_path": str(scenario_path),
        "simulation_csv": str(csv_path),
        "timeline_csv": str(timeline_path),
        "n_windows": int(len(y_true)),
        "detection_delay_s": delay_s,
        "accuracy_over_time_final": final_acc,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
