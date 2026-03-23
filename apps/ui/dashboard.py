"""apps/ui/dashboard.py

Streamlit page for end-to-end AI workflow in TunnelAI-Lab:
0) Scenario Generation
1) Dataset Builder
2) Training
3) Evaluation
4) Model Test
"""

from __future__ import annotations

import os
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Ensure repo root import path when launched from nested cwd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.streaming.run_record import load_scenario, record_to_csv, record_to_exact_csv

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROC_DIR = os.path.join(REPO_ROOT, "data", "processed")
SCENARIO_DIR = os.path.join(REPO_ROOT, "scenarios")
ART_DIR = os.path.join(REPO_ROOT, "artifacts")
DEFAULT_START_TIME = "2026-01-01T08:00:00+01:00"


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    return proc.returncode, proc.stdout


def _list_files(folder: str, suffix: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(suffix.lower())]
    files.sort()
    return files


def _list_dirs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    dirs.sort()
    return dirs


def _parse_train_log(log_text: str) -> pd.DataFrame:
    rows = []
    pat = re.compile(
        r"^\[epoch\s+(\d+)\]\s+train_loss=([0-9eE\.\-]+)\s+val_loss=([0-9eE\.\-]+)\s+"
        r"acc=([0-9eE\.\-]+)\s+prec_macro=([0-9eE\.\-]+)\s+rec_macro=([0-9eE\.\-]+)\s+f1_macro=([0-9eE\.\-]+)"
    )
    for line in log_text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        rows.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
                "accuracy": float(m.group(4)),
                "precision_macro": float(m.group(5)),
                "recall_macro": float(m.group(6)),
                "f1_macro": float(m.group(7)),
            }
        )
    return pd.DataFrame(rows)


def _extract_block(text: str, header: str) -> str:
    lines = text.splitlines()
    out = []
    capture = False
    for line in lines:
        if line.strip() == header.strip():
            capture = True
            continue
        if capture and line.startswith("==="):
            break
        if capture:
            out.append(line)
    return "\n".join(out).strip()


def _parse_kv_block(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def _feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    groups = {
        "traffic features": [],
        "environment features": [],
        "ventilation/control features": [],
        "static tunnel metadata": [],
    }
    for f in feature_names:
        low = str(f).lower()
        if any(k in low for k in ["flow", "speed", "vehicle", "traffic", "queue", "density", "volume"]):
            groups["traffic features"].append(f)
        elif any(k in low for k in ["temp", "humid", "co2", "co", "no2", "pm", "weather", "visibility"]):
            groups["environment features"].append(f)
        elif any(k in low for k in ["fan", "vent", "damper", "control", "setpoint", "jet", "power"]):
            groups["ventilation/control features"].append(f)
        elif any(k in low for k in ["tunnel", "lane", "length", "slope", "grade", "segment", "portal"]):
            groups["static tunnel metadata"].append(f)
    return groups


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_classes: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pooling: str = "last",
    ):
        super().__init__()
        hidden_size = d_model // 2 if bidirectional else d_model
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        if self.pooling == "mean":
            p = h.mean(dim=1)
        else:
            p = h[:, -1, :]
        return self.head(p)


class _TransformerClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_classes: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        pooling: str = "last",
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward if dim_feedforward is not None else (d_model * 4),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pooling = pooling
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        h = self.encoder(z)
        if self.pooling == "mean":
            p = h.mean(dim=1)
        else:
            p = h[:, -1, :]
        return self.head(p)


def _build_model_from_ckpt(ckpt: dict) -> nn.Module | None:
    if "model_state_dict" not in ckpt:
        return None
    backbone = str(ckpt.get("backbone", "lstm")).lower()
    d_in = int(ckpt.get("d_in", 1))
    n_classes = int(ckpt.get("n_classes", 2))
    d_model = int(ckpt.get("d_model", 128))
    n_layers = int(ckpt.get("n_layers", 2))
    n_heads = int(ckpt.get("n_heads", 4))
    dropout = float(ckpt.get("dropout", 0.1))
    pooling = str(ckpt.get("pooling", "last"))
    bidirectional = bool(ckpt.get("bidirectional", False))
    dim_feedforward = ckpt.get("dim_feedforward", None)

    if backbone == "transformer":
        model = _TransformerClassifier(
            d_in=d_in,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_classes=n_classes,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            pooling=pooling,
        )
    else:
        model = _LSTMClassifier(
            d_in=d_in,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            pooling=pooling,
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _layer_table(model: nn.Module, sample_len: int = 60, d_in: int = 8) -> pd.DataFrame:
    rows = []
    outputs = {}
    hooks = []

    def make_hook(name: str):
        def _hook(_module, _inp, out):
            if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                outputs[name] = tuple(out[0].shape)
            elif isinstance(out, torch.Tensor):
                outputs[name] = tuple(out.shape)
            else:
                outputs[name] = str(type(out))
        return _hook

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        x = torch.zeros(1, sample_len, d_in, dtype=torch.float32)
        try:
            _ = model(x)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    for idx, (name, module) in enumerate(model.named_modules()):
        if name == "":
            continue
        pcount = sum(p.numel() for p in module.parameters(recurse=False))
        rows.append(
            {
                "index": idx,
                "name": name,
                "type": module.__class__.__name__,
                "output_shape": outputs.get(name, "-"),
                "param_count": int(pcount),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="TunnelAI-Lab – Unified Dashboard", layout="wide")
st.title("🤖 TunnelAI-Lab – Unified Dashboard")
st.caption("Scenario Generation • Dataset Builder • Training • Evaluation • Model Test")

DEFAULT_ARCH = {
    "backbone": "transformer",
    "d_model": 128,
    "n_layers": 2,
    "dropout": 0.1,
    "pooling": "mean",
    "n_heads": 4,
    "dim_feedforward": 512,
    "bidirectional": False,
}
PRESET_ARCH = {
    "transformer_base": {
        **DEFAULT_ARCH,
        "backbone": "transformer",
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "dim_feedforward": 512,
    },
    "transformer_large": {
        **DEFAULT_ARCH,
        "backbone": "transformer",
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "dim_feedforward": 1024,
        "dropout": 0.15,
    },
    "lstm_base": {
        **DEFAULT_ARCH,
        "backbone": "lstm",
        "d_model": 128,
        "n_layers": 2,
        "bidirectional": False,
    },
    "lstm_bidir": {
        **DEFAULT_ARCH,
        "backbone": "lstm",
        "d_model": 128,
        "n_layers": 2,
        "bidirectional": True,
    },
}

with st.sidebar:
    st.header("📁 Paths")
    st.code(f"RAW_DIR = {RAW_DIR}")
    st.code(f"PROC_DIR = {PROC_DIR}")
    st.code(f"ART_DIR = {ART_DIR}")


# -----------------------------------------------------------------------------
# Scenario Generation
# -----------------------------------------------------------------------------
st.header("0) Scenario Generation")
with st.container(border=True):
    st.caption("Generate many raw scenario CSVs from one base scenario (e.g., 1 real scenario × 1000 seeds).")

    scenario_files = _list_files(SCENARIO_DIR, ".json")
    if not scenario_files:
        st.warning(f"No scenario JSON found in: {SCENARIO_DIR}")
    else:
        g1, g2, g3 = st.columns(3)
        base_scenario = g1.selectbox("base scenario", scenario_files, index=0)
        run_count = g2.number_input("number of runs", min_value=1, max_value=100_000, value=1000, step=1)
        seed_start = g3.number_input("seed start", min_value=0, max_value=10_000_000, value=42, step=1)

        g4, g5, g6 = st.columns(3)
        start_time_iso = g4.text_input("start time (ISO8601)", value=DEFAULT_START_TIME)
        max_seconds_gen = g5.number_input("max seconds", min_value=10, max_value=24 * 3600, value=600, step=10)
        use_exact_format = g6.toggle("write exact wide csv format", value=True)

        out_raw_dir = st.text_input("output dir", value=RAW_DIR, key="scenario_gen_out_dir")
        manifest_path = st.text_input(
            "run manifest json",
            value=os.path.join(ART_DIR, "scenario_generation_last_run.json"),
            key="scenario_gen_manifest",
        )

        if st.button("🎬 Generate Scenario Batch", use_container_width=True):
            os.makedirs(out_raw_dir, exist_ok=True)
            base_path = os.path.join(SCENARIO_DIR, base_scenario)
            base_name = os.path.splitext(base_scenario)[0]
            out_files = []
            progress = st.progress(0.0)

            for i in range(int(run_count)):
                seed_i = int(seed_start) + i
                scn = load_scenario(base_path)
                setattr(scn, "seed", seed_i)
                out_name = f"{base_name}__seed{seed_i:07d}.csv"
                out_csv = os.path.join(out_raw_dir, out_name)
                if use_exact_format:
                    out_path = record_to_exact_csv(
                        scenario=scn,
                        out_csv=out_csv,
                        start_time_iso=start_time_iso,
                        max_seconds=int(max_seconds_gen),
                    )
                else:
                    out_path = record_to_csv(
                        scenario=scn,
                        out_csv=out_csv,
                        start_time_iso=start_time_iso,
                        max_seconds=int(max_seconds_gen),
                    )
                out_files.append(out_path)
                if i % max(1, int(run_count) // 100) == 0 or i == int(run_count) - 1:
                    progress.progress((i + 1) / float(run_count))

            manifest = {
                "created_at_utc": datetime.utcnow().isoformat(),
                "base_scenario": base_scenario,
                "run_count": int(run_count),
                "seed_start": int(seed_start),
                "seed_end": int(seed_start) + int(run_count) - 1,
                "start_time_iso": start_time_iso,
                "max_seconds": int(max_seconds_gen),
                "exact_format": bool(use_exact_format),
                "output_dir": out_raw_dir,
                "files": out_files,
            }
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            st.success(f"Generated {len(out_files)} scenarios in {out_raw_dir}")
            st.json(
                {
                    "first_file": out_files[0] if out_files else None,
                    "last_file": out_files[-1] if out_files else None,
                    "manifest": manifest_path,
                }
            )


# -----------------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------------
st.header("1) Dataset Builder")
with st.container(border=True):
    c0, c1, c2, c3, c4 = st.columns(5)
    raw_subdirs = _list_dirs(RAW_DIR)
    raw_dir_options = [RAW_DIR] + [os.path.join(RAW_DIR, d) for d in raw_subdirs]
    selected_raw_dir = c0.selectbox("raw dir", raw_dir_options, index=0)
    input_format = c1.selectbox("input_format", ["wide_csv", "scenario_csv", "long_tag_csv"], index=0)
    L = c2.number_input("window size (L)", min_value=5, max_value=5000, value=60, step=5)
    H = c3.number_input("H (forecast horizon)", min_value=1, max_value=1000, value=1, step=1)
    stride = c4.number_input("stride", min_value=1, max_value=200, value=5, step=1)

    csv_or_dir_default = selected_raw_dir if input_format in {"wide_csv", "scenario_csv"} else os.path.join(selected_raw_dir, "stau_run_long.csv")
    csv_or_dir = st.text_input("CSV path or directory", value=csv_or_dir_default)
    out_dir = st.text_input("Output directory", value=PROC_DIR)

    if st.button("🧱 Build Dataset", use_container_width=True):
        py = (
            "from core.dataset.dataset_builder import DatasetConfig, build_npz_from_csv;"
            f"cfg=DatasetConfig(input_format='{input_format}',L={int(L)},H={int(H)},stride={int(stride)});"
            f"build_npz_from_csv(csv_path={csv_or_dir!r}, out_dir={out_dir!r}, cfg=cfg)"
        )
        code, out = _run_cmd([sys.executable, "-c", py])
        if code == 0:
            st.success("NPZ build finished.")
        else:
            st.error("NPZ build failed.")
        st.code(out)

        if code == 0:
            train_npz = os.path.join(out_dir, "train.npz")
            val_npz = os.path.join(out_dir, "val.npz")
            test_npz = os.path.join(out_dir, "test.npz")
            if all(os.path.isfile(p) for p in [train_npz, val_npz, test_npz]):
                import numpy as np

                dtr = np.load(train_npz, allow_pickle=False)
                dva = np.load(val_npz, allow_pickle=False)
                dte = np.load(test_npz, allow_pickle=False)

                y_all = np.concatenate(
                    [
                        dtr["Y_event_cls"].astype(np.int64),
                        dva["Y_event_cls"].astype(np.int64),
                        dte["Y_event_cls"].astype(np.int64),
                    ]
                )
                vals, cnts = np.unique(y_all, return_counts=True)
                class_names = dtr["event_class_names"].tolist() if "event_class_names" in dtr.files else []
                dist = {}
                for v, c in zip(vals, cnts):
                    key = class_names[int(v)] if int(v) < len(class_names) else str(int(v))
                    dist[key] = int(c)

                st.markdown("**Dataset Summary**")
                st.write({"num_windows": int(len(y_all)), "class_distribution": dist})


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
st.header("2) Training")
with st.container(border=True):
    if "arch_cfg" not in st.session_state:
        st.session_state.arch_cfg = dict(DEFAULT_ARCH)
    if "arch_text" not in st.session_state:
        st.session_state.arch_text = json.dumps(st.session_state.arch_cfg, indent=2)
    if "experiment_logs" not in st.session_state:
        st.session_state.experiment_logs = []

    st.markdown("**Experiment Settings**")
    e1, e2, e3, e4 = st.columns(4)
    exp_window_length = e1.number_input("window length L", min_value=5, max_value=5000, value=60, step=5)
    exp_stride = e2.number_input("stride", min_value=1, max_value=200, value=5, step=1)
    exp_class_weighting = e3.toggle("class weighting", value=False)
    exp_val_split = e4.slider("validation split", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

    train_npz = st.text_input("train.npz", value=os.path.join(PROC_DIR, "train.npz"))
    val_npz = st.text_input("val.npz (optional)", value=os.path.join(PROC_DIR, "val.npz"))
    test_npz = st.text_input("test.npz", value=os.path.join(PROC_DIR, "test.npz"))
    save_dir = st.text_input("save_dir", value=ART_DIR)

    available_features = []
    if os.path.isfile(train_npz):
        try:
            d = np.load(train_npz, allow_pickle=False)
            if "feature_names" in d.files:
                available_features = [str(x) for x in d["feature_names"].tolist()]
            elif "feature_tags" in d.files:
                available_features = [str(x) for x in d["feature_tags"].tolist()]
        except Exception:
            available_features = []

    group_map = _feature_groups(available_features)
    fg_cols = st.columns(4)
    selected_by_group = set()
    for i, (group_name, group_feats) in enumerate(group_map.items()):
        enabled = fg_cols[i].checkbox(group_name, value=bool(group_feats))
        if enabled:
            selected_by_group.update(group_feats)
    default_features = sorted(selected_by_group) if selected_by_group else available_features
    selected_features = st.multiselect(
        "feature selection",
        options=available_features,
        default=default_features,
        help="Pick feature groups or manually override selected features.",
    )

    st.markdown("**Model Builder**")
    p1, p2 = st.columns([1, 2])
    preset = p1.selectbox("Presets", options=list(PRESET_ARCH.keys()), index=0)
    if p2.button("Load Preset", use_container_width=True):
        st.session_state.arch_cfg = dict(PRESET_ARCH[preset])
        st.session_state.arch_text = json.dumps(st.session_state.arch_cfg, indent=2)
        st.rerun()

    qc = st.session_state.arch_cfg
    q1, q2, q3, q4, q5 = st.columns(5)
    qb = q1.selectbox("backbone type", ["lstm", "transformer"], index=(0 if qc.get("backbone") == "lstm" else 1))
    qd = q2.number_input("hidden dimension", min_value=16, max_value=2048, value=int(qc.get("d_model", 128)), step=16)
    ql = q3.number_input("number of layers", min_value=1, max_value=24, value=int(qc.get("n_layers", 2)), step=1)
    qdrop = q4.number_input("dropout", min_value=0.0, max_value=0.9, value=float(qc.get("dropout", 0.1)), step=0.01)
    qpool = q5.selectbox("pooling", ["mean", "last"], index=(0 if qc.get("pooling", "mean") == "mean" else 1))

    q_extra = {"backbone": qb, "d_model": int(qd), "n_layers": int(ql), "dropout": float(qdrop), "pooling": qpool}
    if qb == "transformer":
        t1, t2 = st.columns(2)
        q_heads = t1.number_input("number of heads", min_value=1, max_value=32, value=int(qc.get("n_heads", 4)), step=1)
        q_ff = t2.number_input("feedforward dimension", min_value=32, max_value=8192, value=int(qc.get("dim_feedforward", 512)), step=32)
        q_extra["n_heads"] = int(q_heads)
        q_extra["dim_feedforward"] = int(q_ff)
        q_extra["bidirectional"] = False
    else:
        l1 = st.columns(1)[0]
        q_bidir = l1.toggle("bidirectional", value=bool(qc.get("bidirectional", False)))
        q_extra["bidirectional"] = bool(q_bidir)
        q_extra["n_heads"] = int(qc.get("n_heads", 4))
        q_extra["dim_feedforward"] = int(qc.get("dim_feedforward", 512))

    # quick controls -> architecture JSON sync
    if q_extra != st.session_state.arch_cfg:
        st.session_state.arch_cfg = dict(q_extra)
        st.session_state.arch_text = json.dumps(st.session_state.arch_cfg, indent=2)

    st.markdown("**Architecture JSON (manual editor)**")
    arch_text = st.text_area("architecture_json", value=st.session_state.arch_text, height=180)
    c_apply, c_reset = st.columns(2)
    if c_apply.button("Apply Architecture JSON", use_container_width=True):
        try:
            parsed = json.loads(arch_text)
            if isinstance(parsed, dict) and "backbone" in parsed:
                merged = dict(DEFAULT_ARCH)
                merged.update(parsed)
                st.session_state.arch_cfg = merged
                st.session_state.arch_text = json.dumps(merged, indent=2)
                st.success("Architecture JSON applied.")
                st.rerun()
            else:
                st.error("JSON must be an object with at least the 'backbone' field.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
    if c_reset.button("Reset Architecture Editor", use_container_width=True):
        st.session_state.arch_text = json.dumps(st.session_state.arch_cfg, indent=2)
        st.rerun()

    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("epochs", min_value=1, max_value=1000, value=20)
    batch_size = c2.number_input("batch_size", min_value=1, max_value=2048, value=64)
    lr = c3.number_input("learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")

    with st.expander("⚙️ Training Settings", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        optimizer = s1.selectbox("optimizer", ["adam", "adamw", "sgd"], index=0)
        weight_decay = s2.number_input("weight decay", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.6f")
        early_stopping = s3.toggle("early stopping", value=False)
        patience = s4.number_input("patience", min_value=1, max_value=200, value=5, step=1, disabled=not early_stopping)

        s5, s6, s7, s8 = st.columns(4)
        if optimizer in {"adam", "adamw"}:
            momentum = s5.number_input("momentum", min_value=0.0, max_value=1.0, value=0.0, step=0.01, disabled=True)
        else:
            momentum = s5.number_input("momentum", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
        use_cuda = s6.toggle("use_cuda", value=True)
        device = s7.selectbox("device", ["auto", "cpu", "cuda"], index=0)
        seed = s8.number_input("random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    training_cfg = {
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "optimizer": str(optimizer),
        "momentum": 0.0 if optimizer in {"adam", "adamw"} else float(momentum),
        "weight_decay": float(weight_decay),
        "early_stopping": bool(early_stopping),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "use_cuda": bool(use_cuda),
        "device": str(device),
        "random_seed": int(seed),
        "backbone": str(st.session_state.arch_cfg.get("backbone", "lstm")),
        "architecture": dict(st.session_state.arch_cfg),
        "experiment_settings": {
            "window_length_L": int(exp_window_length),
            "stride": int(exp_stride),
            "feature_selection": selected_features,
            "feature_groups_selected": [k for k, v in group_map.items() if any(f in selected_features for f in v)],
            "class_weighting": bool(exp_class_weighting),
            "validation_split": float(exp_val_split),
            "selected_feature_count": int(len(selected_features)),
        },
    }

    st.caption("Selected training config")
    st.json(training_cfg)

    if st.button("🏋️ Train Model", use_container_width=True):
        st.info("Starting training with selected settings…")
        st.code(str(training_cfg))
        feature_idx = []
        if available_features and selected_features:
            idx_map = {name: i for i, name in enumerate(available_features)}
            feature_idx = [idx_map[f] for f in selected_features if f in idx_map]
        cmd = [
            sys.executable,
            os.path.join("scripts", "train_event_model.py"),
            "--train", train_npz,
            "--test", test_npz,
            "--epochs", str(training_cfg["epochs"]),
            "--batch_size", str(training_cfg["batch_size"]),
            "--lr", str(training_cfg["learning_rate"]),
            "--backbone", training_cfg["backbone"],
            "--optimizer", training_cfg["optimizer"],
            "--momentum", str(training_cfg["momentum"]),
            "--weight_decay", str(training_cfg["weight_decay"]),
            "--patience", str(training_cfg["patience"]),
            "--device", training_cfg["device"],
            "--seed", str(training_cfg["random_seed"]),
            "--d_model", str(int(training_cfg["architecture"].get("d_model", 128))),
            "--n_layers", str(int(training_cfg["architecture"].get("n_layers", 2))),
            "--dropout", str(float(training_cfg["architecture"].get("dropout", 0.1))),
            "--pooling", str(training_cfg["architecture"].get("pooling", "mean")),
            "--val_split", str(float(training_cfg["experiment_settings"]["validation_split"])),
            "--save_dir", save_dir,
        ]
        if val_npz.strip():
            cmd.extend(["--val", val_npz])
        if feature_idx:
            cmd.extend(["--feature_indices", ",".join(str(i) for i in sorted(set(feature_idx)))])
        if bool(training_cfg["experiment_settings"]["class_weighting"]):
            cmd.append("--class_weighting")
        if training_cfg["architecture"].get("backbone") == "transformer":
            cmd.extend(
                [
                    "--n_heads", str(int(training_cfg["architecture"].get("n_heads", 4))),
                    "--dim_feedforward", str(int(training_cfg["architecture"].get("dim_feedforward", 512))),
                ]
            )
        else:
            if bool(training_cfg["architecture"].get("bidirectional", False)):
                cmd.append("--bidirectional")
        if training_cfg["early_stopping"]:
            cmd.append("--early_stopping")
        if training_cfg["use_cuda"]:
            cmd.append("--use_cuda")
        code, out = _run_cmd(cmd)
        if code == 0:
            st.success("Training finished.")
        else:
            st.error("Training failed.")
        st.code(out)

        hist = _parse_train_log(out)
        if not hist.empty:
            st.markdown("**Training Curves**")
            st.line_chart(hist.set_index("epoch")[["train_loss", "val_loss"]])

            st.markdown("**Validation Metrics**")
            st.line_chart(hist.set_index("epoch")[["accuracy", "f1_macro"]])

            last = hist.iloc[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Val Loss", f"{last['val_loss']:.4f}")
            m2.metric("Final Accuracy", f"{last['accuracy']:.4f}")
            m3.metric("Final F1 Macro", f"{last['f1_macro']:.4f}")

        # Show test outputs from training script
        cm_text = _extract_block(out, "=== Confusion Matrix (rows=true, cols=pred) ===")
        rep_text = _extract_block(out, "=== Classification Report ===")
        test_metrics = _extract_block(out, "=== Test Metrics ===")

        if test_metrics:
            st.markdown("**Test Accuracy / Metrics**")
            st.code(test_metrics)
        val_metrics = _extract_block(out, "=== Validation Metrics ===")
        if val_metrics:
            st.markdown("**Validation Metrics**")
            st.code(val_metrics)
            vm = _parse_kv_block(val_metrics)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("validation accuracy", f"{float(vm.get('accuracy', 0.0)):.4f}")
            mc2.metric("macro precision", f"{float(vm.get('precision_macro', 0.0)):.4f}")
            mc3.metric("macro recall", f"{float(vm.get('recall_macro', 0.0)):.4f}")
            mc4.metric("macro F1", f"{float(vm.get('f1_macro', 0.0)):.4f}")
        val_cm_text = _extract_block(out, "=== Validation Confusion Matrix (rows=true, cols=pred) ===")
        if val_cm_text:
            st.markdown("**Validation Confusion Matrix**")
            st.code(val_cm_text)
        if cm_text:
            st.markdown("**Confusion Matrix**")
            st.code(cm_text)
        if rep_text:
            st.markdown("**Classification Report**")
            st.code(rep_text)

        # Persist small experiment log
        os.makedirs(os.path.join(ART_DIR, "experiment_logs"), exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(ART_DIR, "experiment_logs", f"experiment_{ts}.json")
        log_payload = {
            "timestamp_utc": ts,
            "settings": training_cfg,
            "validation_metrics": _parse_kv_block(val_metrics) if val_metrics else {},
            "test_metrics": _parse_kv_block(test_metrics) if test_metrics else {},
            "command": cmd,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)
        st.success(f"Saved experiment log: {log_path}")
        st.session_state.experiment_logs.append(log_path)

    if st.session_state.experiment_logs:
        st.markdown("**Recent Experiment Logs**")
        st.json(st.session_state.experiment_logs[-5:])


# -----------------------------------------------------------------------------
# Layer Inspection
# -----------------------------------------------------------------------------
st.header("🧠 Layer Inspection")
with st.container(border=True):
    inspect_ckpt = st.text_input("Model checkpoint for inspection", value=os.path.join(ART_DIR, "best_model.pt"))
    inspect_len = st.number_input("Sample sequence length (for output shape probe)", min_value=5, max_value=5000, value=60, step=5)

    if st.button("🔍 Inspect Layers", use_container_width=True):
        if not os.path.isfile(inspect_ckpt):
            st.error(f"Checkpoint not found: {inspect_ckpt}")
        else:
            ckpt = torch.load(inspect_ckpt, map_location="cpu")
            model = _build_model_from_ckpt(ckpt)
            if model is None:
                st.error("Unsupported checkpoint format (missing model_state_dict).")
            else:
                d_in = int(ckpt.get("d_in", 8))
                layer_df = _layer_table(model, sample_len=int(inspect_len), d_in=d_in)

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.markdown("**Model Summary**")
                c1, c2 = st.columns(2)
                c1.metric("Total Params", f"{total_params:,}")
                c2.metric("Trainable Params", f"{trainable_params:,}")

                st.markdown("**Layers**")
                st.dataframe(layer_df, use_container_width=True)

                if not layer_df.empty:
                    layer_choice = st.selectbox(
                        "Layer selector",
                        options=layer_df["name"].tolist(),
                        index=0,
                    )
                    selected_module = dict(model.named_modules()).get(layer_choice)
                    if selected_module is not None:
                        st.markdown("**Selected Layer Details**")
                        st.json(
                            {
                                "name": layer_choice,
                                "type": selected_module.__class__.__name__,
                                "repr": repr(selected_module),
                                "param_count": int(sum(p.numel() for p in selected_module.parameters(recurse=False))),
                            }
                        )

                        # Weight visualizations (best-effort)
                        weight = getattr(selected_module, "weight", None)
                        if isinstance(weight, torch.Tensor):
                            w = weight.detach().cpu().numpy()
                            if w.ndim == 2:
                                st.markdown("**Linear Weight Heatmap**")
                                try:
                                    import matplotlib.pyplot as plt

                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    im = ax.imshow(w, aspect="auto", cmap="viridis")
                                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                    ax.set_xlabel("Input dim")
                                    ax.set_ylabel("Output dim")
                                    st.pyplot(fig, clear_figure=True)
                                except Exception:
                                    st.info("Heatmap visualization not available in this environment.")
                            elif w.ndim in (3, 4):
                                st.markdown("**Conv Filter Preview (first filter)**")
                                filt = w[0]
                                if filt.ndim == 3:
                                    filt = filt[0]
                                f_min, f_max = float(filt.min()), float(filt.max())
                                norm = (filt - f_min) / (f_max - f_min + 1e-9)
                                st.image(norm, caption=f"Filter[0] shape={filt.shape}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
st.header("3) Evaluation")
with st.container(border=True):
    report_out = st.text_input("dataset report JSON", value=os.path.join(PROC_DIR, "dataset_report.json"))
    eval_train = st.text_input("eval train.npz", value=os.path.join(PROC_DIR, "train.npz"), key="eval_train")
    eval_val = st.text_input("eval val.npz", value=os.path.join(PROC_DIR, "val.npz"), key="eval_val")
    eval_test = st.text_input("eval test.npz", value=os.path.join(PROC_DIR, "test.npz"), key="eval_test")

    if st.button("📊 Analyze Dataset", use_container_width=True):
        cmd = [
            sys.executable,
            os.path.join("scripts", "analyze_dataset.py"),
            "--train", eval_train,
            "--val", eval_val,
            "--test", eval_test,
            "--out", report_out,
        ]
        code, out = _run_cmd(cmd)
        if code == 0:
            st.success("Evaluation report created.")
        else:
            st.error("Evaluation failed.")
        st.code(out)


# -----------------------------------------------------------------------------
# Model Test
# -----------------------------------------------------------------------------
st.header("4) Model Test")
with st.container(border=True):
    model_path = st.text_input("model checkpoint", value=os.path.join(ART_DIR, "best_model.pt"))
    scenario_files = _list_files(SCENARIO_DIR, ".json")
    default_scn = ""
    scenario_choice = st.selectbox("Scenario JSON (optional)", ["<auto-generate>"] + scenario_files, index=0)
    if scenario_choice != "<auto-generate>":
        default_scn = os.path.join(SCENARIO_DIR, scenario_choice)

    c1, c2, c3 = st.columns(3)
    max_seconds = c1.number_input("max_seconds", min_value=10, max_value=24 * 3600, value=600, step=10)
    test_L = c2.number_input("L", min_value=5, max_value=2000, value=60, step=5)
    test_stride = c3.number_input("stride", min_value=1, max_value=200, value=5, step=1)
    test_out_dir = st.text_input(
        "test output dir",
        value=os.path.join(ART_DIR, f"sim_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )

    if st.button("🧪 Run Model Test on Simulation", use_container_width=True):
        cmd = [
            sys.executable,
            os.path.join("scripts", "test_model_on_simulation.py"),
            "--model", model_path,
            "--max-seconds", str(int(max_seconds)),
            "--L", str(int(test_L)),
            "--stride", str(int(test_stride)),
            "--out-dir", test_out_dir,
        ]
        if default_scn:
            cmd.extend(["--scenario", default_scn])

        code, out = _run_cmd(cmd)
        if code == 0:
            st.success("Model test finished.")
        else:
            st.error("Model test failed.")

        with st.expander("Command", expanded=False):
            st.code(" ".join(shlex.quote(x) for x in cmd))
        st.code(out)
