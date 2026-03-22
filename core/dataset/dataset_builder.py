# dataset/dataset_builder.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ALLOWED_INPUT_FORMATS = {"long_tag_csv", "scenario_csv"}

DEFAULT_FEATURE_TAGS = [
    "Z2.TRAF.Speed",
    "Z2.TRAF.Density",
    "Z2.CO.S01.Value",
    "Z2.VIS.S01.Value",
    "Z2.VMS.SpeedLimit",
    "Z2.FAN.StageCmd",
]

DEFAULT_FORECAST_TARGETS = [
    "Z2.TRAF.Speed",
    "Z2.CO.S01.Value",
    "Z2.VIS.S01.Value",
]

DEFAULT_SCENARIO_DYNAMIC_FEATURES = [
    "speed_kmh",
    "flow_veh_h",
    "occupancy_pct",
    "co_ppm",
    "no2_ppm",
    "pm25",
    "temp_c",
    "humidity_pct",
    "visibility_m",
    "air_velocity",
    "air_pressure",
    "fire_alarm_state",
    "jet_fan_active_count",
    "barrier_entry_state",
]

DEFAULT_SCENARIO_STATIC_FEATURES = [
    "tunnel_length_m",
    "gradient_pct",
    "curvature_radius_m",
    "profile",
    "tubes",
    "lanes_per_tube",
    "direction_mode",
    "aadt",
    "heavy_vehicle_pct",
    "speed_limit_kmh",
    "vent_system",
    "jet_fan_count",
    "weather",
]


@dataclass
class DatasetConfig:
    """Configuration for dataset building from long-tag or scenario CSV inputs."""

    # Windowing
    L: int = 300
    H: int = 60
    stride: int = 5

    # Input mode
    input_format: str = "long_tag_csv"  # allowed: long_tag_csv, scenario_csv

    # Long-tag mode columns
    feature_tags: List[str] | None = None
    forecast_targets: List[str] | None = None
    event_tag: str = "Z3.EVT.Incident.Active"
    keep_quality: str | None = "GOOD"

    # Scenario-CSV mode columns
    scenario_id_col: str = "scenario_id"
    timestamp_col: str = "timestamp"
    timestamp_s_col: str = "timestamp_s"
    event_type_col: str = "event_type"
    risk_level_col: str = "risk_level"
    scenario_dynamic_features: List[str] | None = None
    scenario_static_features: List[str] | None = None

    # Splitting
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42


def _validate_input_format(fmt: str) -> None:
    if fmt not in ALLOWED_INPUT_FORMATS:
        raise ValueError(f"input_format must be one of {sorted(ALLOWED_INPUT_FORMATS)}, got: {fmt}")


def load_long_csv(csv_path: str, keep_quality: str | None = "GOOD") -> pd.DataFrame:
    """Load legacy long-format tag CSV (timestamp, tag_id, value, ...)."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "quality" in df.columns and keep_quality is not None:
        df = df[df["quality"].astype(str).str.upper().eq(keep_quality.upper())]

    required = {"timestamp", "tag_id", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "scenario_id" not in df.columns:
        df["scenario_id"] = "scenario_0"
    return df


def load_scenario_csv_dir(raw_dir: str, scenario_id_col: str = "scenario_id") -> pd.DataFrame:
    """Load and concatenate one-CSV-per-scenario files from a directory.

    If `scenario_id` column is missing in a file, filename stem is used.
    """
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_dir}")

    chunks: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp)
        if scenario_id_col not in df.columns:
            df[scenario_id_col] = fp.stem
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long tag table to wide scenario/timestamp matrix."""
    return (
        df_long.pivot_table(
            index=["scenario_id", "timestamp"],
            columns="tag_id",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )


def _encode_categorical_columns(df: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Encode string columns into numeric values; return mapping metadata."""
    out = df.copy()
    enc_meta: Dict[str, List[str]] = {}
    for col in cols:
        if col in out.columns and out[col].dtype == object:
            cats = sorted(out[col].fillna("unknown").astype(str).unique().tolist())
            mapping = {name: i for i, name in enumerate(cats)}
            out[col] = out[col].fillna("unknown").astype(str).map(mapping).astype(float)
            enc_meta[col] = cats
    return out, enc_meta


def _normalize_event_type(x: object) -> str:
    s = str(x).strip().lower()
    if s in {"", "none", "normal", "0"}:
        return "normal"
    aliases = {
        "accident": "accident",
        "collision": "accident",
        "fire": "fire",
        "vehicle_fire": "fire",
        "breakdown": "breakdown",
        "stalled_vehicle": "standstill",
        "standstill": "standstill",
        "ghostdriver": "ghostdriver",
        "wrong_way_driver": "ghostdriver",
    }
    return aliases.get(s, s)


def _event_mapping(values: Sequence[str]) -> tuple[Dict[str, int], np.ndarray]:
    classes = sorted({_normalize_event_type(v) for v in values} | {"normal"})
    mapping = {name: i for i, name in enumerate(classes)}
    return mapping, np.array(classes, dtype=str)


def _window_label_from_future(future_labels: np.ndarray, normal_id: int) -> int:
    """Choose multiclass event label for a window horizon.

    If any non-normal class appears in the horizon, use the first non-normal class,
    otherwise use normal.
    """
    non_normal = future_labels[future_labels != normal_id]
    if len(non_normal) > 0:
        return int(non_normal[0])
    return int(normal_id)


def make_windows_long(wide: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, object]:
    """Build windows from legacy wide tag matrix (pivoted from long CSV)."""
    feature_tags = cfg.feature_tags or DEFAULT_FEATURE_TAGS
    forecast_targets = cfg.forecast_targets or []
    event_tag = cfg.event_tag

    needed = set(feature_tags) | {event_tag} | set(forecast_targets)
    missing = [t for t in needed if t not in wide.columns]
    if missing:
        raise KeyError(f"Missing tags in wide data: {missing}")

    X_list: List[np.ndarray] = []
    Yf_list: List[np.ndarray] = []
    Ye_cls_list: List[int] = []
    ts_list: List[np.datetime64] = []
    sid_list: List[str] = []

    for scenario_id, block in wide.groupby(level=0):
        block = block.droplevel(0).sort_index()

        feat = block[feature_tags].astype(float)
        ev = (block[event_tag].astype(float) >= 0.5).astype(int)

        if forecast_targets:
            targ = block[forecast_targets].astype(float)
            ok = feat.notna().all(axis=1) & targ.notna().all(axis=1) & ev.notna()
            targ = targ[ok]
        else:
            targ = None
            ok = feat.notna().all(axis=1) & ev.notna()

        feat = feat[ok]
        ev = ev[ok]

        if len(feat) < (cfg.L + cfg.H):
            continue

        feat_np = feat.to_numpy()
        ev_np = ev.to_numpy()
        ts_np = feat.index.to_numpy()
        targ_np = targ.to_numpy() if targ is not None else None

        max_start = len(feat_np) - (cfg.L + cfg.H) + 1
        for start in range(0, max_start, cfg.stride):
            end_x = start + cfg.L
            end_y = end_x + cfg.H

            X_list.append(feat_np[start:end_x, :])
            future_ev = ev_np[end_x:end_y]
            Ye_cls_list.append(1 if np.any(future_ev >= 1) else 0)
            if targ_np is not None:
                Yf_list.append(targ_np[end_x:end_y, :])
            ts_list.append(ts_np[end_x - 1])
            sid_list.append(str(scenario_id))

    if not X_list:
        raise RuntimeError("No windows generated. Check L/H/stride and selected tags.")

    X = np.stack(X_list).astype(np.float32)
    Y_event_cls = np.array(Ye_cls_list, dtype=np.int64)
    Y_forecast = np.stack(Yf_list).astype(np.float32) if Yf_list else None

    meta = {
        "feature_names": np.array(feature_tags, dtype=str),
        "forecast_targets": np.array(forecast_targets, dtype=str),
        "event_class_names": np.array(["normal", "incident"], dtype=str),
        "timestamps": np.array(ts_list).astype("datetime64[ns]").astype(np.int64),
        "scenario_ids": np.array(sid_list, dtype=str),
        "L": np.array([cfg.L], dtype=np.int64),
        "H": np.array([cfg.H], dtype=np.int64),
        "stride": np.array([cfg.stride], dtype=np.int64),
    }

    return {
        "X": X,
        "Y_event_cls": Y_event_cls,
        "Y_event": (Y_event_cls > 0).astype(np.float32),
        "Y_forecast": Y_forecast,
        "meta": meta,
    }


def make_windows_scenario_csv(df: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, object]:
    """Build windows directly from scenario-wise timeseries CSV rows."""
    required = {cfg.scenario_id_col, cfg.event_type_col}
    time_col = cfg.timestamp_col if cfg.timestamp_col in df.columns else cfg.timestamp_s_col
    if time_col not in df.columns:
        raise ValueError(
            f"scenario_csv requires either '{cfg.timestamp_col}' or '{cfg.timestamp_s_col}' column"
        )
    required.add(time_col)

    dynamic_features = cfg.scenario_dynamic_features or DEFAULT_SCENARIO_DYNAMIC_FEATURES
    static_features = cfg.scenario_static_features or DEFAULT_SCENARIO_STATIC_FEATURES
    feature_cols = dynamic_features + static_features

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"scenario_csv missing required columns: {missing_required}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"scenario_csv missing feature columns: {missing_features}")

    forecast_targets = cfg.forecast_targets or []
    missing_targets = [c for c in forecast_targets if c not in df.columns]
    if missing_targets:
        raise ValueError(f"scenario_csv missing forecast target columns: {missing_targets}")

    event_mapping, event_class_names = _event_mapping(df[cfg.event_type_col].astype(str).tolist())
    normal_id = int(event_mapping["normal"])

    df_local = df.copy()
    df_local[cfg.event_type_col] = df_local[cfg.event_type_col].map(_normalize_event_type)
    df_local["_event_id"] = df_local[cfg.event_type_col].map(event_mapping).astype(int)

    # ensure numeric matrix for features (encode string columns like profile/weather/vent_system)
    feat_frame, cat_meta = _encode_categorical_columns(df_local[feature_cols], feature_cols)
    feat_frame = feat_frame.apply(pd.to_numeric, errors="coerce")

    X_list: List[np.ndarray] = []
    Yf_list: List[np.ndarray] = []
    Ye_cls_list: List[int] = []
    ts_list: List[np.int64] = []
    sid_list: List[str] = []

    grouped = df_local.groupby(cfg.scenario_id_col)
    for sid, block in grouped:
        order = np.argsort(block[time_col].to_numpy())
        block = block.iloc[order]

        feat_block = feat_frame.loc[block.index]
        ev_block = block["_event_id"]

        if forecast_targets:
            targ_block = block[forecast_targets].apply(pd.to_numeric, errors="coerce")
            ok = feat_block.notna().all(axis=1) & targ_block.notna().all(axis=1) & ev_block.notna()
            targ_block = targ_block[ok]
        else:
            targ_block = None
            ok = feat_block.notna().all(axis=1) & ev_block.notna()

        feat_block = feat_block[ok]
        ev_block = ev_block[ok]
        ts_block = block.loc[ok, time_col]

        if len(feat_block) < (cfg.L + cfg.H):
            continue

        feat_np = feat_block.to_numpy(dtype=np.float32)
        ev_np = ev_block.to_numpy(dtype=np.int64)
        targ_np = targ_block.to_numpy(dtype=np.float32) if targ_block is not None else None

        # timestamps: store int seconds when timestamp_s exists, else unix ns
        if time_col == cfg.timestamp_s_col:
            ts_np = pd.to_numeric(ts_block, errors="coerce").fillna(0).astype(np.int64).to_numpy()
        else:
            ts_np = pd.to_datetime(ts_block).astype(np.int64).to_numpy()

        max_start = len(feat_np) - (cfg.L + cfg.H) + 1
        for start in range(0, max_start, cfg.stride):
            end_x = start + cfg.L
            end_y = end_x + cfg.H

            X_list.append(feat_np[start:end_x, :])
            future_labels = ev_np[end_x:end_y]
            Ye_cls_list.append(_window_label_from_future(future_labels, normal_id=normal_id))
            if targ_np is not None:
                Yf_list.append(targ_np[end_x:end_y, :])
            ts_list.append(int(ts_np[end_x - 1]))
            sid_list.append(str(sid))

    if not X_list:
        raise RuntimeError("No windows generated from scenario_csv. Check L/H/stride and column completeness.")

    X = np.stack(X_list).astype(np.float32)
    Y_event_cls = np.array(Ye_cls_list, dtype=np.int64)
    Y_forecast = np.stack(Yf_list).astype(np.float32) if Yf_list else None

    meta = {
        "feature_names": np.array(feature_cols, dtype=str),
        "forecast_targets": np.array(forecast_targets, dtype=str),
        "event_class_names": event_class_names,
        "timestamps": np.array(ts_list, dtype=np.int64),
        "scenario_ids": np.array(sid_list, dtype=str),
        "L": np.array([cfg.L], dtype=np.int64),
        "H": np.array([cfg.H], dtype=np.int64),
        "stride": np.array([cfg.stride], dtype=np.int64),
    }
    if cat_meta:
        for name, cats in cat_meta.items():
            meta[f"cat_{name}"] = np.array(cats, dtype=str)

    return {
        "X": X,
        "Y_event_cls": Y_event_cls,
        "Y_event": (Y_event_cls != normal_id).astype(np.float32),
        "Y_forecast": Y_forecast,
        "meta": meta,
    }


def split_by_scenario(data: Dict[str, object], cfg: DatasetConfig):
    """Split by scenario_id to avoid leakage across train/val/test."""
    sids = data["meta"]["scenario_ids"]
    unique = np.unique(sids)

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))

    train_s = set(unique[:n_train])
    val_s = set(unique[n_train:n_train + n_val])
    test_s = set(unique[n_train + n_val:])

    mask_train = np.array([sid in train_s for sid in sids])
    mask_val = np.array([sid in val_s for sid in sids])
    mask_test = np.array([sid in test_s for sid in sids])

    def subset(mask: np.ndarray) -> Dict[str, object]:
        sub_meta = dict(data["meta"])
        sub_meta["timestamps"] = data["meta"]["timestamps"][mask]
        sub_meta["scenario_ids"] = data["meta"]["scenario_ids"][mask]
        return {
            "X": data["X"][mask],
            "Y_event_cls": data["Y_event_cls"][mask],
            "Y_event": data["Y_event"][mask],
            "Y_forecast": data["Y_forecast"][mask] if data["Y_forecast"] is not None else None,
            "meta": sub_meta,
        }

    return subset(mask_train), subset(mask_val), subset(mask_test)


def print_class_distribution(y_event_cls: np.ndarray, class_names: np.ndarray, prefix: str = "") -> None:
    """Print class histogram for quick dataset sanity checks."""
    unique, counts = np.unique(y_event_cls, return_counts=True)
    total = max(1, int(counts.sum()))
    print(f"{prefix}class distribution:")
    for cls_id, c in zip(unique, counts):
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
        print(f"  - {name} ({int(cls_id)}): {int(c)} ({100.0 * c / total:.2f}%)")


def save_npz(path: str, part: Dict[str, object]) -> str:
    """Save split dataset to NPZ with backward compatible and new keys."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = part["meta"]

    payload: Dict[str, object] = {
        "X": part["X"],
        "Y_event_cls": part["Y_event_cls"],
        "Y_event": part["Y_event"],  # backward compatible binary target
        "scenario_ids": meta["scenario_ids"],
        "timestamps": meta["timestamps"],
        "feature_names": meta["feature_names"],
        "event_class_names": meta["event_class_names"],
        "L": meta["L"],
        "H": meta["H"],
        "stride": meta["stride"],
        # backward compatible aliases
        "feature_tags": meta["feature_names"],
        "forecast_targets": meta["forecast_targets"],
    }
    if part["Y_forecast"] is not None:
        payload["Y_forecast"] = part["Y_forecast"]

    # include optional categorical metadata keys
    for k, v in meta.items():
        if k.startswith("cat_"):
            payload[k] = v

    np.savez_compressed(path, **payload)
    return path


def _build_data_from_input(csv_path: str, cfg: DatasetConfig) -> Dict[str, object]:
    _validate_input_format(cfg.input_format)
    if cfg.input_format == "long_tag_csv":
        if cfg.feature_tags is None:
            cfg.feature_tags = DEFAULT_FEATURE_TAGS
        df_long = load_long_csv(csv_path, keep_quality=cfg.keep_quality)
        wide = pivot_to_wide(df_long)
        return make_windows_long(wide, cfg)

    # scenario_csv mode expects a directory of scenario csv files OR one csv file
    p = Path(csv_path)
    if p.is_dir():
        df = load_scenario_csv_dir(str(p), scenario_id_col=cfg.scenario_id_col)
    else:
        df = pd.read_csv(p)
        if cfg.scenario_id_col not in df.columns:
            df[cfg.scenario_id_col] = p.stem
    return make_windows_scenario_csv(df, cfg)


def build_npz_from_csv(
    csv_path: str,
    out_dir: str = "data/processed",
    cfg: DatasetConfig | None = None,
) -> Tuple[str, str, str]:
    """Build train/val/test NPZ files from either long-tag CSV or scenario CSV input."""
    cfg = cfg or DatasetConfig()
    data = _build_data_from_input(csv_path, cfg)

    print_class_distribution(data["Y_event_cls"], data["meta"]["event_class_names"], prefix="all | ")

    train, val, test = split_by_scenario(data, cfg)

    train_path = save_npz(os.path.join(out_dir, "train.npz"), train)
    val_path = save_npz(os.path.join(out_dir, "val.npz"), val)
    test_path = save_npz(os.path.join(out_dir, "test.npz"), test)

    print_class_distribution(train["Y_event_cls"], train["meta"]["event_class_names"], prefix="train | ")
    print_class_distribution(val["Y_event_cls"], val["meta"]["event_class_names"], prefix="val | ")
    print_class_distribution(test["Y_event_cls"], test["meta"]["event_class_names"], prefix="test | ")

    yf_shape = lambda part: part["Y_forecast"].shape if part["Y_forecast"] is not None else None
    print("✅ Saved NPZ:")
    print("  train:", train_path, train["X"].shape, yf_shape(train), train["Y_event_cls"].shape)
    print("  val:  ", val_path, val["X"].shape, yf_shape(val), val["Y_event_cls"].shape)
    print("  test: ", test_path, test["X"].shape, yf_shape(test), test["Y_event_cls"].shape)

    return train_path, val_path, test_path


if __name__ == "__main__":
    cfg = DatasetConfig(L=300, H=60, stride=5, input_format="long_tag_csv")
    build_npz_from_csv(
        csv_path="data/raw/stau_run_long.csv",
        out_dir="data/processed",
        cfg=cfg,
    )
