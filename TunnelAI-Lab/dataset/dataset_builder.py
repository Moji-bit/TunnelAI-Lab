# dataset/dataset_builder.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class DatasetConfig:
    # Windowing
    L: int = 300
    H: int = 60
    stride: int = 5  # sample every 5s -> less redundancy

    # Tags
    feature_tags: List[str] = None
    forecast_targets: List[str] = None
    event_tag: str = "Z3.EVT.Incident.Active"

    # Quality filter
    keep_quality: str = "GOOD"

    # Splitting
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42


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


def load_long_csv(csv_path: str, keep_quality: str = "GOOD") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "quality" in df.columns and keep_quality is not None:
        df = df[df["quality"].astype(str).str.upper().eq(keep_quality.upper())]

    # enforce required columns
    required = {"timestamp", "tag_id", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "scenario_id" not in df.columns:
        # fallback: single scenario
        df["scenario_id"] = "scenario_0"

    return df


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    # MultiIndex: (scenario_id, timestamp) -> columns: tag_id
    wide = (
        df_long.pivot_table(
            index=["scenario_id", "timestamp"],
            columns="tag_id",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )
    return wide


def first_sustained_true(mask: np.ndarray, hold: int) -> int | None:
    """
    mask: boolean array length T
    returns start index of first run of True with length >= hold
    """
    if len(mask) < hold:
        return None
    run = np.convolve(mask.astype(int), np.ones(hold, dtype=int), mode="valid")
    if not np.any(run >= hold):
        return None
    return int(np.argmax(run >= hold))


def make_windows(
    wide: pd.DataFrame,
    cfg: DatasetConfig,
) -> Dict[str, object]:
    feature_tags = cfg.feature_tags or DEFAULT_FEATURE_TAGS
    forecast_targets = cfg.forecast_targets or DEFAULT_FORECAST_TARGETS
    event_tag = cfg.event_tag

    needed = set(feature_tags) | set(forecast_targets) | {event_tag}
    missing = [t for t in needed if t not in wide.columns]
    if missing:
        raise KeyError(f"Missing tags in wide data: {missing}")

    X_list: List[np.ndarray] = []
    Yf_list: List[np.ndarray] = []
    Ye_list: List[float] = []
    ts_list: List[np.datetime64] = []
    sid_list: List[str] = []

    # process scenario-wise (prevents leakage)
    for scenario_id, block in wide.groupby(level=0):
        block = block.droplevel(0).sort_index()  # index: timestamp

        feat = block[feature_tags].astype(float)
        targ = block[forecast_targets].astype(float)
        ev = block[event_tag].astype(float)

        # drop rows with any NaNs in used fields
        ok = feat.notna().all(axis=1) & targ.notna().all(axis=1) & ev.notna()
        feat = feat[ok]
        targ = targ[ok]
        ev = ev[ok]

        if len(feat) < (cfg.L + cfg.H):
            continue

        feat_np = feat.to_numpy()         # (T, d)
        targ_np = targ.to_numpy()         # (T, m)
        ev_np = ev.to_numpy()             # (T,)
        ts = feat.index.to_numpy()        # (T,)

        max_start = len(feat_np) - (cfg.L + cfg.H) + 1
        for start in range(0, max_start, cfg.stride):
            end_x = start + cfg.L
            end_y = end_x + cfg.H

            x = feat_np[start:end_x, :]          # (L, d)
            y_f = targ_np[end_x:end_y, :]        # (H, m)

            # event label: any incident in next H seconds
            future = ev_np[end_x:end_y]
            y_e = 1.0 if np.any(future >= 0.5) else 0.0

            X_list.append(x)
            Yf_list.append(y_f)
            Ye_list.append(y_e)
            ts_list.append(ts[end_x - 1])
            sid_list.append(str(scenario_id))

    if not X_list:
        raise RuntimeError("No windows generated. Check L/H/stride and missing tags.")

    X = np.stack(X_list).astype(np.float32)
    Y_forecast = np.stack(Yf_list).astype(np.float32)
    Y_event = np.array(Ye_list, dtype=np.float32)

    meta = {
        "feature_tags": np.array(feature_tags, dtype=str),
        "forecast_targets": np.array(forecast_targets, dtype=str),
        "event_tag": np.array([event_tag], dtype=str),
        "L": np.array([cfg.L], dtype=np.int64),
        "H": np.array([cfg.H], dtype=np.int64),
        "stride": np.array([cfg.stride], dtype=np.int64),
        "timestamps": np.array(ts_list).astype("datetime64[ns]").astype(np.int64),  # ns int
        "scenario_ids": np.array(sid_list, dtype=str),
    }

    return {"X": X, "Y_forecast": Y_forecast, "Y_event": Y_event, "meta": meta}


def split_by_scenario(data: Dict[str, object], cfg: DatasetConfig):
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

    def subset(mask):
        sub = {
            "X": data["X"][mask],
            "Y_forecast": data["Y_forecast"][mask],
            "Y_event": data["Y_event"][mask],
            "meta": dict(data["meta"]),
        }
        sub["meta"]["timestamps"] = data["meta"]["timestamps"][mask]
        sub["meta"]["scenario_ids"] = data["meta"]["scenario_ids"][mask]
        return sub

    return subset(mask_train), subset(mask_val), subset(mask_test)


def save_npz(path: str, part: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    meta = part["meta"]
    np.savez_compressed(
        path,
        X=part["X"],
        Y_forecast=part["Y_forecast"],
        Y_event=part["Y_event"],
        timestamps=meta["timestamps"],
        scenario_ids=meta["scenario_ids"],
        feature_tags=meta["feature_tags"],
        forecast_targets=meta["forecast_targets"],
        event_tag=meta["event_tag"],
        L=meta["L"],
        H=meta["H"],
        stride=meta["stride"],
    )
    return path


def build_npz_from_csv(
    csv_path: str,
    out_dir: str = "data/processed",
    cfg: DatasetConfig | None = None,
) -> Tuple[str, str, str]:
    cfg = cfg or DatasetConfig()
    if cfg.feature_tags is None:
        cfg.feature_tags = DEFAULT_FEATURE_TAGS
    if cfg.forecast_targets is None:
        cfg.forecast_targets = DEFAULT_FORECAST_TARGETS

    df_long = load_long_csv(csv_path, keep_quality=cfg.keep_quality)
    wide = pivot_to_wide(df_long)

    data = make_windows(wide, cfg)
    train, val, test = split_by_scenario(data, cfg)

    train_path = save_npz(os.path.join(out_dir, "train.npz"), train)
    val_path = save_npz(os.path.join(out_dir, "val.npz"), val)
    test_path = save_npz(os.path.join(out_dir, "test.npz"), test)

    print("✅ Saved NPZ:")
    print("  train:", train_path, train["X"].shape, train["Y_forecast"].shape, train["Y_event"].shape)
    print("  val:  ", val_path,   val["X"].shape,   val["Y_forecast"].shape,   val["Y_event"].shape)
    print("  test: ", test_path,  test["X"].shape,  test["Y_forecast"].shape,  test["Y_event"].shape)

    return train_path, val_path, test_path


if __name__ == "__main__":
    # Default run for your case
    cfg = DatasetConfig(L=300, H=60, stride=5)
    build_npz_from_csv(
        csv_path="data/raw/stau_run_long.csv",
        out_dir="data/processed",
        cfg=cfg,
    )