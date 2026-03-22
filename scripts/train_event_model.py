from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.dataset.torch_data import build_loaders
from core.evaluation.metrics import classification_report_dict


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
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last)


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
        h_last = h[:, -1, :]
        return self.head(h_last)


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Metrics:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    acc = float(tp.sum() / max(1.0, cm.sum()))
    return Metrics(
        accuracy=acc,
        precision_macro=float(np.mean(precision)),
        recall_macro=float(np.mean(recall)),
        f1_macro=float(np.mean(f1)),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> tuple[float, Metrics, np.ndarray]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_samples = 0
    ys_true = []
    ys_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            logits = model(x)
            loss = loss_fn(logits, y)

            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            n_samples += bs

            pred = torch.argmax(logits, dim=1)
            ys_true.append(y.cpu().numpy())
            ys_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(ys_true) if ys_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ys_pred) if ys_pred else np.array([], dtype=np.int64)
    cm = confusion_matrix_np(y_true, y_pred, n_classes=n_classes)
    m = metrics_from_cm(cm)
    mean_loss = total_loss / max(1, n_samples)
    return mean_loss, m, cm


def main() -> None:
    ap = argparse.ArgumentParser(description="Train event classifier on NPZ windows")
    ap.add_argument("--train", default="data/processed/train.npz")
    ap.add_argument("--val", default="data/processed/val.npz")
    ap.add_argument("--test", default="data/processed/test.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save_dir", default="artifacts")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    train_loader, val_loader, test_loader, _scaler, meta = build_loaders(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    feature_names = meta.get("feature_names")
    event_class_names = meta.get("event_class_names")
    n_classes = int(len(event_class_names)) if event_class_names is not None else 2

    d_in = int(train_loader.dataset.X.shape[-1])
    if args.backbone == "lstm":
        model = LSTMClassifier(d_in=d_in, d_model=args.d_model, n_layers=args.n_layers, n_classes=n_classes, dropout=args.dropout)
    else:
        model = TransformerClassifier(
            d_in=d_in,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_classes=n_classes,
            dropout=args.dropout,
        )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
        )

    best_f1 = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_model.pt")

    print(f"device={device}, backbone={args.backbone}, d_in={d_in}, classes={n_classes}")
    print(
        "training_settings="
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"optimizer={args.optimizer}, momentum={args.momentum}, weight_decay={args.weight_decay}, "
        f"early_stopping={args.early_stopping}, patience={args.patience}, seed={args.seed}"
    )
    if feature_names is not None:
        print(f"features={list(feature_names)}")
    if event_class_names is not None:
        print(f"event_class_names={list(event_class_names)}")

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            n_samples += bs

        train_loss = total_loss / max(1, n_samples)
        val_loss, val_m, _ = evaluate(model, val_loader, device=device, n_classes=n_classes)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={val_m.accuracy:.4f} prec_macro={val_m.precision_macro:.4f} "
            f"rec_macro={val_m.recall_macro:.4f} f1_macro={val_m.f1_macro:.4f}"
        )

        if val_m.f1_macro > best_f1:
            best_f1 = val_m.f1_macro
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "d_in": d_in,
                    "n_classes": n_classes,
                    "feature_names": feature_names,
                    "event_class_names": event_class_names,
                },
                best_path,
            )
            print(f"  ↳ saved best model: {best_path} (f1_macro={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if args.early_stopping and epochs_no_improve >= max(1, int(args.patience)):
                print(f"  ↳ early stopping at epoch {epoch} (patience={args.patience})")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_m, test_cm = evaluate(model, test_loader, device=device, n_classes=n_classes)
    print("\n=== Test Metrics ===")
    print(f"loss={test_loss:.4f}")
    print(f"accuracy={test_m.accuracy:.4f}")
    print(f"precision_macro={test_m.precision_macro:.4f}")
    print(f"recall_macro={test_m.recall_macro:.4f}")
    print(f"f1_macro={test_m.f1_macro:.4f}")

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(test_cm)

    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true_all.extend(y.cpu().numpy().tolist())
            y_pred_all.extend(pred.cpu().numpy().tolist())

    target_names = [str(x) for x in event_class_names] if event_class_names is not None else None
    report = classification_report_dict(y_true_all, y_pred_all, target_names=target_names)
    print("\n=== Classification Report ===")
    print(report)


if __name__ == "__main__":
    main()
