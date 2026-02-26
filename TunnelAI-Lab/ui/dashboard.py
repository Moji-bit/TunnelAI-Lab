"""ui/dashboard.py

Streamlit user interface for non-technical exploration of generated tunnel data.

Main capabilities:
1) Run a configured scenario and create a CSV file
2) Load an existing CSV and replay it like a live stream
3) Filter/select signals using metadata from tags.yaml
4) Display status and simple risk-oriented tag ranking

This file intentionally keeps UI logic in one place for readability during thesis work.
"""

# ui/dashboard.py
from __future__ import annotations

import os
import sys
import time
import requests

from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st
import yaml

# ensure repo root is importable when Streamlit launches from nested paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from streaming.run_record import load_scenario, record_to_csv


# -------------------------
# Paths / Constants
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCENARIO_DIR = os.path.join(BASE_DIR, "scenarios")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TAGS_YAML = os.path.join(BASE_DIR, "tags", "tags.yaml")

DEFAULT_START = "2026-01-01T08:00:00+01:00"
DEFAULT_MAX_SECONDS = 300  # Quick demo; set None for full duration

CRIT_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}
BACKEND_BASE_URL = os.getenv("TUNNEL_BACKEND_URL", "http://127.0.0.1:8000")


# -------------------------
# File utilities
# -------------------------
def list_json_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    files.sort()
    return files


def list_csv_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort()
    return files


def make_out_csv_path(scenario_path: str, seed: int) -> str:
    base = os.path.splitext(os.path.basename(scenario_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RAW_DIR, f"{base}__seed{seed}__{ts}.csv")


# -------------------------
# Data handling: long -> wide
# -------------------------
@st.cache_data(show_spinner=False)
def load_long_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot_table(
        index="timestamp",
        columns="tag_id",
        values="value",
        aggfunc="mean",
    ).sort_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


# -------------------------
# Tags.yaml helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_tags_yaml(path: str = TAGS_YAML) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tag_index(cfg: dict) -> dict:
    idx = {}
    for t in cfg.get("tags", []):
        idx[t["tag_id"]] = t
    return idx


def tag_label(tag_id: str, meta: dict) -> str:
    unit = meta.get("unit")
    parts = tag_id.split(".")
    seg = next((p for p in parts if p.startswith("S") and len(p) == 3), "")
    signal = parts[-1]
    zone = f"Z{meta.get('zone')}" if meta.get("zone") else parts[0]
    subsystem = meta.get("subsystem", parts[1] if len(parts) > 1 else "")
    u = f" [{unit}]" if unit else ""
    if seg:
        return f"{zone} | {subsystem} | {seg} | {signal}{u}"
    return f"{zone} | {subsystem} | {signal}{u}"


def limit_status(value: float, meta: dict) -> str:
    limits = meta.get("limits")
    if not limits:
        return "⚪️"

    vmin = limits.get("min", None)
    vmax = limits.get("max", None)

    if (vmin is not None and value < vmin) or (vmax is not None and value > vmax):
        return "🔴"

    band = 0.10
    if vmin is not None and vmax is not None and vmax > vmin:
        span = vmax - vmin
        if value < vmin + band * span or value > vmax - band * span:
            return "🟡"

    return "🟢"


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="TunnelAI-Lab – App", layout="wide")
st.title("🚇 TunnelAI-Lab – App (Szenario-Runner + Playback)")

with st.sidebar:
    st.header("⚙️ Szenario ausführen")

    scenarios = list_json_files(SCENARIO_DIR)
    if not scenarios:
        st.error(f"Keine Szenario-JSONs gefunden in: {os.path.relpath(SCENARIO_DIR, BASE_DIR)}")
        st.stop()

    selected_scn = st.selectbox("Szenario wählen", scenarios, index=0)
    scenario_path = os.path.join(SCENARIO_DIR, selected_scn)

    start_time = st.text_input("Startzeit (ISO8601)", value=DEFAULT_START)
    max_seconds = st.number_input(
        "Max seconds (Playback-Länge)",
        min_value=10,
        max_value=24 * 3600,
        value=DEFAULT_MAX_SECONDS,
        step=10,
        help="Für schnelle Tests. Für volle Länge später auf None/leer umstellen.",
    )
    seed = st.number_input(
        "Seed (Reproduzierbarkeit)",
        min_value=0,
        max_value=10_000_000,
        value=42,
        step=1,
        help="Gleicher Seed => gleiche CSV. Anderer Seed => neue, aber reproduzierbare Variante.",
    )
    
    run_btn = st.button("▶️ Run Scenario → CSV erzeugen", use_container_width=True)

    st.markdown("---")
    st.header("📂 Playback")

    csv_files = list_csv_files(RAW_DIR)
    selected_csv = st.selectbox(
        "CSV wählen (data/raw)",
        csv_files,
        index=max(0, len(csv_files) - 1) if csv_files else 0,
        disabled=(len(csv_files) == 0),
    )

    play_speed = st.slider("Playback Speed", 1, 20, 6, 1)
    window = st.slider("Chart-Fenster (letzte N Samples)", 50, 2000, 400, 50)

    c1, c2, c3 = st.columns(3)
    start_play = c1.button("▶️ Start")
    pause_play = c2.button("⏸️ Pause")
    reset_play = c3.button("🔄 Reset")


# -------------------------
# Session state init
# -------------------------
st.session_state.setdefault("playing", False)
st.session_state.setdefault("i", 0)
st.session_state.setdefault("last_csv_path", None)
st.session_state.setdefault("wide", None)
st.session_state.setdefault("long", None)
st.session_state.setdefault("status_by_ts", None)
st.session_state.setdefault("backend_session_id", None)
st.session_state.setdefault("backend_last_frame", None)


# -------------------------
# Run scenario -> create CSV
# -------------------------
if run_btn:
    scn = load_scenario(scenario_path)
    setattr(scn, "seed", int(seed))
    out_csv = make_out_csv_path(scenario_path, int(seed))

    out_csv = record_to_csv(
        scenario=scn,
        out_csv=out_csv,
        start_time_iso=start_time,
        max_seconds=int(max_seconds) if max_seconds else None,
    )

    st.success(f"✅ CSV erzeugt: {out_csv}")
    st.session_state.last_csv_path = None
    st.session_state.i = 0


# -------------------------
# Load selected CSV (long->wide)
# -------------------------
if selected_csv:
    csv_path = os.path.join(RAW_DIR, selected_csv)

    if st.session_state.last_csv_path != csv_path:
        df_long = load_long_csv(csv_path)
        df_wide = long_to_wide(df_long)

        status_by_ts = (
            df_long.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")[["scenario_id", "quality"]]
        )

        st.session_state.long = df_long
        st.session_state.wide = df_wide
        st.session_state.status_by_ts = status_by_ts
        st.session_state.last_csv_path = csv_path
        st.session_state.i = 0


df_long: Optional[pd.DataFrame] = st.session_state.long
df_wide: Optional[pd.DataFrame] = st.session_state.wide
status_by_ts: Optional[pd.DataFrame] = st.session_state.status_by_ts

if df_wide is None or df_wide.empty:
    st.info("Noch keine Daten geladen. Erzeuge ein Szenario oder wähle eine CSV.")
    st.stop()


# -------------------------
# Tags.yaml -> Filter + Labels
# -------------------------
cfg = load_tags_yaml()
tag_idx = build_tag_index(cfg)

available_tags = [c for c in df_wide.columns if c in tag_idx]

zones = sorted({tag_idx[t].get("zone") for t in available_tags})
subs = sorted({tag_idx[t].get("subsystem") for t in available_tags})

with st.sidebar:
    st.markdown("---")
    st.header("🧩 Tag-Filter (aus tags.yaml)")

    zone_sel = st.multiselect("Zone", zones, default=zones)
    subs_sel = st.multiselect("Subsystem", subs, default=subs)

    segs = sorted({
        p
        for t in available_tags
        for p in t.split(".")
        if p.startswith("S") and len(p) == 3
    })
    
    seg_sel = st.multiselect("Segment", segs, default=segs)

    selected_tags = []
    for t in available_tags:
        m = tag_idx[t]
        z_ok = m.get("zone") in zone_sel
        s_ok = m.get("subsystem") in subs_sel
        seg = next((p for p in t.split(".") if p.startswith("S") and len(p) == 3), None)
        seg_ok = (seg in seg_sel) if seg else True
        if z_ok and s_ok and seg_ok:
            selected_tags.append(t)

    label_map = {tag_label(t, tag_idx[t]): t for t in selected_tags}
    shown = st.multiselect(
        "Signals für Chart",
        options=list(label_map.keys()),
        default=list(label_map.keys())[:8],
    )
    chart_tags = [label_map[x] for x in shown]


# -------------------------
# Playback controls
# -------------------------
if start_play:
    try:
        if not st.session_state.backend_session_id:
            resp = requests.post(
                f"{BACKEND_BASE_URL}/api/playback/session",
                json={"scenario_id": os.path.splitext(selected_scn)[0]},
                timeout=3,
            )
            resp.raise_for_status()
            st.session_state.backend_session_id = resp.json()["session_id"]

        requests.post(
            f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
            json={"cmd": "play"},
            timeout=3,
        ).raise_for_status()
        requests.post(
            f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
            json={"cmd": "speed", "factor": float(play_speed)},
            timeout=3,
        ).raise_for_status()
        st.session_state.playing = True
    except Exception as ex:
        st.error(f"Backend-Playback konnte nicht gestartet werden: {ex}")
if pause_play:
    st.session_state.playing = False
    if st.session_state.backend_session_id:
        try:
            requests.post(
                f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
                json={"cmd": "pause"},
                timeout=3,
            )
        except Exception:
            pass
if reset_play:
    st.session_state.playing = False
    st.session_state.i = 0
    st.session_state.backend_last_frame = None
    if st.session_state.backend_session_id:
        try:
            requests.post(
                f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
                json={"cmd": "seek", "t": 0},
                timeout=3,
            )
            requests.post(
                f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
                json={"cmd": "pause"},
                timeout=3,
            )
        except Exception:
            pass


# -------------------------
# Main layout
# -------------------------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("📈 Live Charts")
    chart_area = st.empty()
    table_area = st.empty()

with right:
    st.subheader("🧾 Status / Info")
    status_area = st.empty()
    tags_area = st.empty()
    st.markdown("---")
    st.write(f"CSV: `{os.path.basename(st.session_state.last_csv_path)}`")
    st.write(f"Samples: {len(df_wide):,}")
    st.write(f"Tags: {len(df_wide.columns):,}")


def render_frame(i: int) -> None:
    i = max(0, min(i, len(df_wide) - 1))
    start_i = max(0, i - window)

    view = df_wide.iloc[start_i : i + 1].copy()
    current = df_wide.iloc[i, :]  # <- Series

    # Chart (selected signals)
    plot_df = view[chart_tags] if chart_tags else view.iloc[:, :8]
    # Normalize chart input to avoid Altair/jsonschema recursion issues with mixed dtypes/index objects.
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    plot_df = plot_df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="all")

    if plot_df.empty:
        chart_area.info("Keine numerischen Werte für die aktuelle Auswahl vorhanden.")
    else:
        # Use simple RangeIndex to keep the backend payload strictly tabular/numeric.
        chart_area.line_chart(plot_df.reset_index(drop=True))

    # Table (nur sichtbare/ausgewählte Spalten, um UI-Lag zu reduzieren)
    table_cols = list(plot_df.columns)[: min(12, len(plot_df.columns))]
    table_area.dataframe(view[table_cols].tail(12) if table_cols else pd.DataFrame(), use_container_width=True)

    # Status
    ts = df_wide.index[i]
    if status_by_ts is not None and ts in status_by_ts.index:
        scenario_id = status_by_ts.at[ts, "scenario_id"]
        quality = status_by_ts.at[ts, "quality"]
    else:
        scenario_id = "-"
        quality = "-"

    status_area.markdown(
        f"""
        **Timestamp:** `{ts}`  
        **Scenario:** `{scenario_id}`  
        **Quality:** `{quality}`
        """
    )

    # Top-Tags by (criticality + limit proximity)
    rows = []
    for t in plot_df.columns:  # nur die angezeigten Tags bewerten
        meta = tag_idx.get(t)
        if meta is None:
            continue

        val = current.get(t)
        if pd.isna(val):
            continue

        v = float(val)
        lamp = limit_status(v, meta)
        crit = meta.get("criticality", "low")
        w = CRIT_WEIGHT.get(crit, 1)

        viol = 2 if lamp == "🔴" else (1 if lamp == "🟡" else 0)
        score = w + viol

        rows.append(
            {
                "score": score,
                "lamp": lamp,
                "tag": tag_label(t, meta),
                "value": v,
                "unit": meta.get("unit", ""),
                "criticality": crit,
            }
        )

    if rows:
        top_df = pd.DataFrame(rows).sort_values("score", ascending=False).head(12)
        tags_area.dataframe(
            top_df[["lamp", "tag", "value", "unit", "criticality"]],
            use_container_width=True,
        )
    else:
        tags_area.info("Keine Tag-Metadaten/Values für Ranking verfügbar.")


# -------------------------
# Erst rendern, dann ggf. nächsten Tick planen
# -------------------------
render_frame(st.session_state.i)

if st.session_state.backend_session_id:
    st.info(
        "🔗 Frontend Sync: Öffne das Web-Frontend mit URL-Parameter `?session_id="
        f"{st.session_state.backend_session_id}` um denselben Playback-Stream zu sehen."
    )

if st.session_state.playing and st.session_state.backend_session_id:
    try:
        requests.post(
            f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/control",
            json={"cmd": "speed", "factor": float(play_speed)},
            timeout=3,
        ).raise_for_status()
        frame_resp = requests.post(
            f"{BACKEND_BASE_URL}/api/playback/session/{st.session_state.backend_session_id}/frame",
            timeout=3,
        )
        frame_resp.raise_for_status()
        st.session_state.backend_last_frame = frame_resp.json()

        with right:
            st.markdown("---")
            st.subheader("🚗 Backend-Frame (für Frontend + Dashboard)")
            frame = st.session_state.backend_last_frame
            st.write(f"Session: `{st.session_state.backend_session_id}`")
            st.write(f"t={frame.get('t', 0):.1f}s | Fahrzeuge={len(frame.get('vehicles', []))}")
            st.dataframe(pd.DataFrame(frame.get("vehicles", [])).head(12), use_container_width=True)
    except Exception as ex:
        st.warning(f"Backend-Frame konnte nicht geladen werden: {ex}")

if st.session_state.playing:
    interval_ms = max(800, int(1000 / play_speed))  # langsameres Tick-Tempo reduziert Blinken/Frieren
    time.sleep(interval_ms / 1000.0)

    if st.session_state.i < len(df_wide) - 1:
        st.session_state.i += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.success("Run fertig ✅")
