from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = BASE_DIR / "core"
for p in (BASE_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.dataset.torch_data import build_loaders
from core.models.multitask_model import MultiTaskModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, w_forecast: float, w_event: float, w_risk: float) -> dict:
    model.eval()
    n = 0
    sum_loss = 0.0
    sum_f = 0.0
    sum_e = 0.0
    sum_r = 0.0
    sum_acc = 0.0

    with torch.no_grad():
        for x, y_f, y_e in loader:
            x = x.to(device)
            y_f = y_f.to(device)
            y_e = y_e.to(device)

            p_f, p_e, p_r = model(x)
            p_e = p_e.squeeze(-1)
            p_r = p_r.squeeze(-1)

            loss_f = F.mse_loss(p_f, y_f)
            loss_e = F.binary_cross_entropy_with_logits(p_e, y_e)
            # risk-label proxy: same binary label as event head
            loss_r = F.binary_cross_entropy_with_logits(p_r, y_e)
            loss = w_forecast * loss_f + w_event * loss_e + w_risk * loss_r

            pred = (torch.sigmoid(p_e) >= 0.5).float()
            acc = (pred == y_e).float().mean()

            b = x.size(0)
            n += b
            sum_loss += float(loss.item()) * b
            sum_f += float(loss_f.item()) * b
            sum_e += float(loss_e.item()) * b
            sum_r += float(loss_r.item()) * b
            sum_acc += float(acc.item()) * b

    if n == 0:
        return {"loss": float("nan"), "loss_forecast": float("nan"), "loss_event": float("nan"), "loss_risk": float("nan"), "event_acc": float("nan")}

    return {
        "loss": sum_loss / n,
        "loss_forecast": sum_f / n,
        "loss_event": sum_e / n,
        "loss_risk": sum_r / n,
        "event_acc": sum_acc / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train MultiTaskModel on NPZ windows")
    p.add_argument("--train", default="data/processed/train.npz")
    p.add_argument("--val", default="data/processed/val.npz")
    p.add_argument("--test", default="data/processed/test.npz")

    p.add_argument("--backbone", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["mean", "last"], default="mean")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--w-forecast", type=float, default=1.0)
    p.add_argument("--w-event", type=float, default=1.0)
    p.add_argument("--w-risk", type=float, default=0.3)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="artifacts/multitask/best.pt")
    args = p.parse_args()

    set_seed(args.seed)

    train_loader, val_loader, test_loader, scaler, meta = build_loaders(
        train_path=str(BASE_DIR / args.train),
        val_path=str(BASE_DIR / args.val),
        test_path=str(BASE_DIR / args.test),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    d_in = len(meta["feature_tags"]) if meta.get("feature_tags") is not None else next(iter(train_loader))[0].shape[-1]
    m = len(meta["forecast_targets"]) if meta.get("forecast_targets") is not None else next(iter(train_loader))[1].shape[-1]

    train_npz = np.load(BASE_DIR / args.train, allow_pickle=False)
    H = int(train_npz["H"][0]) if "H" in train_npz.files else int(next(iter(train_loader))[1].shape[1])

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = MultiTaskModel(
        backbone=args.backbone,
        d_in=d_in,
        d_model=args.d_model,
        H=H,
        m=m,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y_f, y_e in train_loader:
            x = x.to(device)
            y_f = y_f.to(device)
            y_e = y_e.to(device)

            p_f, p_e, p_r = model(x)
            p_e = p_e.squeeze(-1)
            p_r = p_r.squeeze(-1)

            loss_f = F.mse_loss(p_f, y_f)
            loss_e = F.binary_cross_entropy_with_logits(p_e, y_e)
            loss_r = F.binary_cross_entropy_with_logits(p_r, y_e)
            loss = args.w_forecast * loss_f + args.w_event * loss_e + args.w_risk * loss_r

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader, device, args.w_forecast, args.w_event, args.w_risk)
        val_metrics = evaluate(model, val_loader, device, args.w_forecast, args.w_event, args.w_risk)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_event_acc={val_metrics['event_acc']:.3f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": args.backbone,
                    "d_in": d_in,
                    "d_model": args.d_model,
                    "H": H,
                    "m": m,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                    "pooling": args.pooling,
                    "seed": args.seed,
                    "scaler_mean": getattr(scaler, "mean_", None),
                    "scaler_std": getattr(scaler, "std_", None),
                    "feature_tags": meta.get("feature_tags"),
                    "forecast_targets": meta.get("forecast_targets"),
                    "history": history,
                    "best_val_loss": best_val,
                },
                out_path,
            )

    print(f"✅ best checkpoint: {out_path}")
    test_metrics = evaluate(model, test_loader, device, args.w_forecast, args.w_event, args.w_risk)
    print("test:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
