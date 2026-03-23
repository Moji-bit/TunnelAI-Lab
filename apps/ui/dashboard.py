"""Streamlit dashboard for raw TunnelAI playback.

This page is intentionally lightweight and resilient:
- Works even when the `scenarios/` folder is empty or removed.
- Focuses on existing CSV files in `data/raw/`.
"""

from __future__ import annotations

import os
import time
from typing import List

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
SCENARIO_DIR = os.path.join(REPO_ROOT, "scenarios")


def list_csv_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort()
    return files


def list_json_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    files.sort()
    return files


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure we always have a numeric index-like axis for plotting.
    if "timestamp" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


st.set_page_config(page_title="TunnelAI-Lab – Raw Playback", layout="wide")
st.title("🚇 TunnelAI-Lab – Raw Playback Dashboard")

scenario_files = list_json_files(SCENARIO_DIR)
if not scenario_files:
    st.info(
        "Keine Szenario-Dateien gefunden. Das ist okay: der Runner ist deaktiviert und nur Raw-Playback ist aktiv."
    )

with st.sidebar:
    st.header("📂 Playback (data/raw)")

    csv_files = list_csv_files(RAW_DIR)
    if not csv_files:
        st.error(f"Keine CSV-Dateien gefunden in: {RAW_DIR}")
        st.stop()

    selected_csv = st.selectbox("CSV wählen", csv_files, index=len(csv_files) - 1)
    speed = st.slider("Playback-Geschwindigkeit", min_value=1, max_value=20, value=6, step=1)
    window = st.slider("Chart-Fenster (Samples)", min_value=25, max_value=2000, value=300, step=25)

    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("▶️ Start", use_container_width=True)
    pause_btn = c2.button("⏸️ Pause", use_container_width=True)
    reset_btn = c3.button("🔄 Reset", use_container_width=True)

# Backward-compatibility flags for older dashboard fragments.
# Some stale local copies referenced `run_btn`/`start_play`/`pause_play`/`reset_play`.
run_btn = False
start_play = start_btn
pause_play = pause_btn
reset_play = reset_btn

csv_path = os.path.join(RAW_DIR, selected_csv)
df = load_csv(csv_path)

if df.empty:
    st.warning("Die gewählte CSV ist leer.")
    st.stop()

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.warning("Keine numerischen Spalten zum Plotten gefunden.")
    st.dataframe(df.head(50), use_container_width=True)
    st.stop()

st.session_state.setdefault("playing", False)
st.session_state.setdefault("i", 0)

if start_btn:
    st.session_state.playing = True
if pause_btn:
    st.session_state.playing = False
if reset_btn:
    st.session_state.playing = False
    st.session_state.i = 0

max_i = len(df) - 1
st.session_state.i = max(0, min(st.session_state.i, max_i))

left, right = st.columns([2.2, 1])

with left:
    st.subheader("📈 Live Charts")
    i = st.session_state.i
    start_i = max(0, i - window)
    view = df.iloc[start_i : i + 1]
    st.line_chart(view[numeric_cols[:8]], use_container_width=True)

with right:
    st.subheader("🧾 Status")
    st.write(f"Datei: `{selected_csv}`")
    st.write(f"Zeile: **{st.session_state.i + 1} / {len(df)}**")
    st.write(f"Numerische Signale: **{len(numeric_cols)}**")
    st.dataframe(df.iloc[[st.session_state.i]].T, use_container_width=True, height=420)

if st.session_state.playing and st.session_state.i < max_i:
    st.session_state.i += 1
    time.sleep(max(0.01, 0.2 / speed))
    st.rerun()
