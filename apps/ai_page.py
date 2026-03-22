"""apps/ai_page.py

Streamlit page for end-to-end AI workflow in TunnelAI-Lab:
1) Dataset Builder
2) Training
3) Evaluation
4) Model Test
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import pandas as pd

# Ensure repo root import path when launched from nested cwd
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROC_DIR = os.path.join(REPO_ROOT, "data", "processed")
SCENARIO_DIR = os.path.join(REPO_ROOT, "scenarios")
ART_DIR = os.path.join(REPO_ROOT, "artifacts")


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


st.set_page_config(page_title="TunnelAI-Lab – AI Page", layout="wide")
st.title("🤖 TunnelAI-Lab – AI Workflow")
st.caption("Dataset Builder • Training • Evaluation • Model Test")

with st.sidebar:
    st.header("📁 Paths")
    st.code(f"RAW_DIR = {RAW_DIR}")
    st.code(f"PROC_DIR = {PROC_DIR}")
    st.code(f"ART_DIR = {ART_DIR}")


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
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("epochs", min_value=1, max_value=1000, value=20)
    batch_size = c2.number_input("batch_size", min_value=1, max_value=2048, value=64)
    lr = c3.number_input("learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
    backbone = c4.selectbox("backbone", ["lstm", "transformer"], index=0)

    train_npz = st.text_input("train.npz", value=os.path.join(PROC_DIR, "train.npz"))
    val_npz = st.text_input("val.npz", value=os.path.join(PROC_DIR, "val.npz"))
    test_npz = st.text_input("test.npz", value=os.path.join(PROC_DIR, "test.npz"))
    save_dir = st.text_input("save_dir", value=ART_DIR)

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
        "backbone": str(backbone),
    }

    st.caption("Selected training config")
    st.json(training_cfg)

    if st.button("🏋️ Train Model", use_container_width=True):
        st.info("Starting training with selected settings…")
        st.code(str(training_cfg))
        cmd = [
            sys.executable,
            os.path.join("scripts", "train_event_model.py"),
            "--train", train_npz,
            "--val", val_npz,
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
            "--save_dir", save_dir,
        ]
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
        if cm_text:
            st.markdown("**Confusion Matrix**")
            st.code(cm_text)
        if rep_text:
            st.markdown("**Classification Report**")
            st.code(rep_text)


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
