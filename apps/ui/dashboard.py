"""ui/dashboard.py

Single-app mode notice.

The interactive workflow has been consolidated into `apps/ai_page.py`.
"""

from __future__ import annotations

import os
import sys

import streamlit as st
import yaml
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.streaming.run_record import load_scenario, record_to_csv


# -------------------------
# Paths / Constants
# -------------------------
SCENARIO_DIR = os.path.join(REPO_ROOT, "scenarios")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
TAGS_YAML = os.path.join(REPO_ROOT, "core", "tags", "tags.yaml")

DEFAULT_START = "2026-01-01T08:00:00+01:00"
DEFAULT_MAX_SECONDS = 300  # Quick demo; set None for full duration

CRIT_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class LSTMClassifier(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_layers: int, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_layers: int, n_heads: int, n_classes: int):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        h = self.encoder(z)
        return self.head(h[:, -1, :])


@st.cache_resource(show_spinner=False)
def load_ai_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    backbone = ckpt.get("backbone", "lstm")
    d_in = int(ckpt.get("d_in", 1))
    n_classes = int(ckpt.get("n_classes", 2))
    if backbone == "transformer":
        model = TransformerClassifier(d_in=d_in, d_model=128, n_layers=2, n_heads=4, n_classes=n_classes)
    else:
        model = LSTMClassifier(d_in=d_in, d_model=128, n_layers=2, n_classes=n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


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
        st.error(f"Keine Szenario-JSONs gefunden in: {os.path.relpath(SCENARIO_DIR, REPO_ROOT)}")
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

    st.markdown("---")
    st.header("🤖 AI Prediction Mode")
    ai_mode = st.toggle("AI live prediction", value=False)
    model_path = st.text_input("Model path", value=os.path.join(REPO_ROOT, "artifacts", "best_model.pt"))
    ai_window = st.number_input("AI window size (L)", min_value=5, max_value=5000, value=60, step=5)


# -------------------------
# Session state init
# -------------------------
st.session_state.setdefault("playing", False)
st.session_state.setdefault("i", 0)
st.session_state.setdefault("last_csv_path", None)
st.session_state.setdefault("wide", None)
st.session_state.setdefault("long", None)
st.session_state.setdefault("status_by_ts", None)


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

    segs = sorted(
        {
            p
            for t in available_tags
            for p in t.split(".")
            if p.startswith("S") and len(p) == 3
        }
    )

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
    st.session_state.playing = True
if pause_play:
    st.session_state.playing = False
if reset_play:
    st.session_state.playing = False
    st.session_state.i = 0


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
    ai_area = st.empty()
    tags_area = st.empty()
    st.markdown("---")
    st.write(f"CSV: `{os.path.basename(st.session_state.last_csv_path)}`")
    st.write(f"Samples: {len(df_wide):,}")
    st.write(f"Tags: {len(df_wide.columns):,}")


def render_frame(i: int) -> None:
    i = max(0, min(i, len(df_wide) - 1))
    start_i = max(0, i - window)

    view = df_wide.iloc[start_i : i + 1].copy()
    current = df_wide.iloc[i, :]

    # Chart (selected signals)
    plot_df = view[chart_tags] if chart_tags else view.iloc[:, :8]
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    plot_df = plot_df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="all")

    if plot_df.empty:
        chart_area.info("Keine numerischen Werte für die aktuelle Auswahl vorhanden.")
    else:
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

    # AI live prediction
    if ai_mode:
        if not os.path.isfile(model_path):
            ai_area.warning(f"Model nicht gefunden: {model_path}")
        else:
            try:
                model, ckpt = load_ai_model(model_path)
                feature_names = [str(x) for x in ckpt.get("feature_names", [])]
                class_names = [str(x) for x in ckpt.get("event_class_names", [])]
                if not feature_names:
                    ai_area.warning("Im Modell keine feature_names gespeichert.")
                else:
                    end_i = i + 1
                    start_ai = max(0, end_i - int(ai_window))
                    block = df_wide.iloc[start_ai:end_i]
                    if len(block) < int(ai_window):
                        ai_area.info(f"AI wartet auf genug Samples ({len(block)}/{int(ai_window)}).")
                    else:
                        use_feats = [f for f in feature_names if f in block.columns]
                        if len(use_feats) != len(feature_names):
                            missing = sorted(set(feature_names) - set(use_feats))
                            ai_area.warning(f"Fehlende Feature-Spalten: {missing[:6]}")
                        if not use_feats:
                            ai_area.error("Keine passenden Modell-Features in der CSV.")
                        else:
                            x_np = block[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
                            if x_np.shape[1] != int(ckpt.get("d_in", x_np.shape[1])):
                                ai_area.error("Feature-Dimension passt nicht zum Modell.")
                            else:
                                x_t = torch.from_numpy(x_np).unsqueeze(0)
                                with torch.no_grad():
                                    logits = model(x_t)
                                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                pred_idx = int(probs.argmax())
                                conf = float(probs[pred_idx])
                                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                                risk_high = pred_name not in {"normal", "none", "0"} and conf >= 0.5
                                if risk_high:
                                    ai_area.error(f"⚠️ Predicted Event: **{pred_name}** | Confidence: **{conf:.2%}**")
                                else:
                                    ai_area.success(f"Predicted Event: **{pred_name}** | Confidence: **{conf:.2%}**")
            except Exception as e:
                ai_area.error(f"AI Prediction Fehler: {e}")

    # Top-Tags by (criticality + limit proximity)
    rows = []
    for t in plot_df.columns:
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

if st.session_state.playing:
    interval_ms = max(800, int(1000 / play_speed))
    time.sleep(interval_ms / 1000.0)

    if st.session_state.i < len(df_wide) - 1:
        st.session_state.i += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.success("Run fertig ✅")
