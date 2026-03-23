from __future__ import annotations

import json
import os
from io import BytesIO

import pandas as pd
import streamlit as st

from apps.ui.services.augmentation_engine import AugmentationConfig, generate_augmented_dataset
from apps.ui.services.data_loader import (
    load_ground_truth,
    load_scenario_metadata,
    load_timeseries,
    load_tunnel_config,
)
from apps.ui.services.data_merger import build_merged_dataset
from apps.ui.services.export_service import export_augmented_files
from apps.ui.services.scenario_generator import list_presets
from apps.ui.services.schema_validator import validate_cross_file_consistency, validate_schema

st.set_page_config(page_title="TunnelAI Dataset Builder", layout="wide")
st.title("🧪 Dataset Builder + Data Augmentation")

st.caption(
    "Lädt tunnel_config/scenario_metadata/timeseries/ground_truth, validiert die Konsistenz und erzeugt "
    "tausende plausible augmentierte Szenarien."
)

uploads = {
    "tunnel_config": st.file_uploader("1) tunnel_config.csv", type=["csv"], key="up_tunnel"),
    "scenario_metadata": st.file_uploader("2) scenario_metadata.csv", type=["csv"], key="up_scenario"),
    "timeseries": st.file_uploader("3) timeseries.csv", type=["csv"], key="up_timeseries"),
    "ground_truth": st.file_uploader("4) ground_truth.csv", type=["csv"], key="up_ground_truth"),
}

if not all(uploads.values()):
    st.info("Bitte alle vier CSV-Dateien hochladen.")
    st.stop()


def _to_df(uploaded, loader_func):
    return loader_func(BytesIO(uploaded.getvalue()))


source = {
    "tunnel_config": _to_df(uploads["tunnel_config"], load_tunnel_config),
    "scenario_metadata": _to_df(uploads["scenario_metadata"], load_scenario_metadata),
    "timeseries": _to_df(uploads["timeseries"], load_timeseries),
    "ground_truth": _to_df(uploads["ground_truth"], load_ground_truth),
}

st.subheader("📄 Datenvorschau")
preview_tabs = st.tabs(["tunnel_config", "scenario_metadata", "timeseries", "ground_truth"])
for key, tab in zip(source.keys(), preview_tabs):
    with tab:
        st.write(f"Rows: {len(source[key])}, Columns: {len(source[key].columns)}")
        st.dataframe(source[key].head(100), use_container_width=True)

st.subheader("✅ Schema-Validierungsreport")
report_rows = []
for key, df in source.items():
    report_rows.extend(validate_schema(key, df).as_rows())
report_rows.extend(
    validate_cross_file_consistency(
        source["tunnel_config"],
        source["scenario_metadata"],
        source["timeseries"],
        source["ground_truth"],
    ).as_rows()
)

report_df = pd.DataFrame(report_rows)
if report_df.empty:
    st.success("Keine Validierungsprobleme erkannt.")
else:
    st.dataframe(report_df, use_container_width=True)
    if (report_df["level"] == "error").any():
        st.error("Bitte Fehler beheben, bevor Augmentation gestartet wird.")
        st.stop()

st.subheader("📊 Szenario-Zusammenfassung")
scn = source["scenario_metadata"]
summary_cols = st.columns(5)
summary_cols[0].metric("Anzahl Szenarien", scn["scenario_id"].nunique())
summary_cols[1].metric("Event-Typen", scn["event_type"].nunique())
summary_cols[2].metric("Wettertypen", scn["weather_type"].nunique())
summary_cols[3].metric("Ø Dauer (s)", f"{pd.to_numeric(scn['simulation_duration_s'], errors='coerce').mean():.1f}")
summary_cols[4].metric("Ø Time step (s)", f"{pd.to_numeric(scn['time_step_s'], errors='coerce').mean():.2f}")

c1, c2 = st.columns(2)
with c1:
    st.bar_chart(scn["event_type"].astype(str).value_counts())
with c2:
    st.bar_chart(scn["event_severity"].astype(str).value_counts())

st.subheader("⚙️ Augmentation-Panel")
col_a, col_b, col_c = st.columns(3)
with col_a:
    target_scenarios = st.number_input("Anzahl Ziel-Szenarien", min_value=10, max_value=200000, value=5000, step=100)
    augmentation_strength = st.slider("Augmentationsstärke", 0.01, 0.80, 0.25, 0.01)
    noise_level = st.slider("Noise-Level", 0.0, 0.20, 0.03, 0.005)
with col_b:
    event_shift_range_s = st.slider("Event-Shift-Range (s)", 0, 240, 30, 5)
    missing_rate = st.slider("Missing-Rate", 0.0, 0.10, 0.01, 0.001)
    lag_max_steps = st.slider("Sensor-Lag max. Steps", 0, 10, 2, 1)
with col_c:
    seed = st.number_input("Random Seed", min_value=1, max_value=2_000_000_000, value=42, step=1)
    preset = st.selectbox("Preset", options=list_presets(), index=0)
    allowed_weather = st.multiselect(
        "Erlaubte Wettervariationen",
        options=["clear", "rain", "snow", "fog", "storm"],
        default=["clear", "rain", "snow", "fog"],
    )

st.markdown("**Class-Balance Ziele (event_type → Anteil)**")
cb_cols = st.columns(5)
class_balance_targets = {
    "normal": cb_cols[0].number_input("normal", 0.0, 1.0, 0.40, 0.01),
    "congestion": cb_cols[1].number_input("congestion", 0.0, 1.0, 0.18, 0.01),
    "accident": cb_cols[2].number_input("accident", 0.0, 1.0, 0.16, 0.01),
    "fire": cb_cols[3].number_input("fire", 0.0, 1.0, 0.12, 0.01),
    "sensor_fault": cb_cols[4].number_input("sensor_fault", 0.0, 1.0, 0.14, 0.01),
}

cfg_dict = {
    "target_scenarios": int(target_scenarios),
    "augmentation_strength": float(augmentation_strength),
    "noise_level": float(noise_level),
    "event_shift_range_s": int(event_shift_range_s),
    "missing_rate": float(missing_rate),
    "lag_max_steps": int(lag_max_steps),
    "allowed_weather": allowed_weather,
    "class_balance_targets": class_balance_targets,
    "random_seed": int(seed),
    "preset": preset,
}
st.code(json.dumps(cfg_dict, ensure_ascii=False, indent=2), language="json")

if st.button("Generate Augmented Dataset", type="primary"):
    cfg = AugmentationConfig(
        target_scenarios=int(target_scenarios),
        augmentation_strength=float(augmentation_strength),
        noise_level=float(noise_level),
        event_shift_range_s=int(event_shift_range_s),
        missing_rate=float(missing_rate),
        lag_max_steps=int(lag_max_steps),
        allowed_weather=allowed_weather,
        class_balance_targets=class_balance_targets,
        random_seed=int(seed),
    )

    generated = generate_augmented_dataset(
        tunnel_config=source["tunnel_config"],
        scenario_metadata=source["scenario_metadata"],
        timeseries=source["timeseries"],
        ground_truth=source["ground_truth"],
        cfg=cfg,
    )

    merged = build_merged_dataset(
        tunnel_config=generated["tunnel_config"],
        scenario_metadata=generated["scenario_metadata"],
        timeseries=generated["timeseries"],
        ground_truth=generated["ground_truth"],
    )
    generated["merged"] = merged

    st.session_state["generated"] = generated
    st.success("Augmentiertes Dataset erfolgreich erzeugt.")

if "generated" not in st.session_state:
    st.stop()

generated = st.session_state["generated"]

st.subheader("🔍 Preview before export (Original vs. augmentiert)")
orig_id = str(source["scenario_metadata"]["scenario_id"].iloc[0])
aug_id = str(generated["scenario_metadata"]["scenario_id"].iloc[0])

compare_orig = source["timeseries"].loc[source["timeseries"]["scenario_id"].astype(str) == orig_id]
compare_aug = generated["timeseries"].loc[generated["timeseries"]["scenario_id"].astype(str) == aug_id]

for signal in ["speed_mean_kmh", "flow_veh_h", "occupancy_pct", "co_ppm", "visibility_m", "queue_length_m"]:
    if signal not in compare_orig.columns or signal not in compare_aug.columns:
        continue
    st.markdown(f"**{signal}**")
    left, right = st.columns(2)
    with left:
        st.caption(f"Original ({orig_id})")
        st.line_chart(compare_orig.set_index("timestamp_s")[[signal]], use_container_width=True)
    with right:
        st.caption(f"Augmentiert ({aug_id})")
        st.line_chart(compare_aug.set_index("timestamp_s")[[signal]], use_container_width=True)

st.subheader("📦 Export")
export_dir = st.text_input("Export-Verzeichnis", value=os.path.join("data", "augmented"))
if st.button("Export augmented CSVs"):
    written = export_augmented_files(generated, export_dir)
    st.success(f"Dateien exportiert: {len(written)}")
    st.json(written)

for key, df in generated.items():
    if key not in {"timeseries", "ground_truth", "scenario_metadata", "tunnel_config", "merged"}:
        continue
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {key}.csv",
        data=csv,
        file_name=f"augmented_{key}.csv" if key != "merged" else "merged_training_dataset.csv",
        mime="text/csv",
    )
