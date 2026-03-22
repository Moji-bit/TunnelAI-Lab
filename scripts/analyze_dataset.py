from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_npz(path: str) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def class_distribution(y: np.ndarray, class_names: list[str] | None = None) -> dict[str, int]:
    y_int = y.astype(np.int64)
    values, counts = np.unique(y_int, return_counts=True)
    out: dict[str, int] = {}
    for cls, cnt in zip(values, counts):
        key = str(int(cls))
        if class_names is not None and int(cls) < len(class_names):
            key = class_names[int(cls)]
        out[key] = int(cnt)
    return out


def summarize_split(name: str, npz: dict[str, np.ndarray], class_names: list[str] | None) -> dict[str, Any]:
    X = npz["X"].astype(np.float32)
    y = npz["Y_event_cls"].astype(np.int64)
    feature_names = npz.get("feature_names", npz.get("feature_tags", np.array([], dtype=str)))
    scenario_ids = npz.get("scenario_ids", np.array([], dtype=str)).astype(str)

    stats = {
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
    }

    print(f"\n[{name}]")
    print(f"  X shape: {X.shape}")
    print(f"  Y_event_cls shape: {y.shape}")
    print(f"  features ({len(feature_names)}): {list(feature_names)}")
    if class_names is not None:
        print(f"  class_names: {class_names}")
    dist = class_distribution(y, class_names)
    print(f"  class_distribution: {dist}")
    print(f"  stats: {stats}")

    return {
        "name": name,
        "shape_X": list(X.shape),
        "shape_Y_event_cls": list(y.shape),
        "feature_names": [str(x) for x in feature_names.tolist()],
        "class_distribution": dist,
        "stats": stats,
        "scenario_ids": scenario_ids.tolist(),
    }


def check_overlap(train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> dict[str, list[str]]:
    tv = sorted(train_ids & val_ids)
    tt = sorted(train_ids & test_ids)
    vt = sorted(val_ids & test_ids)
    return {
        "train_val": tv,
        "train_test": tt,
        "val_test": vt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze NPZ dataset splits and export JSON report")
    ap.add_argument("--train", default="data/processed/train.npz")
    ap.add_argument("--val", default="data/processed/val.npz")
    ap.add_argument("--test", default="data/processed/test.npz")
    ap.add_argument("--out", default="data/processed/dataset_report.json")
    args = ap.parse_args()

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    class_names_arr = train.get("event_class_names", None)
    class_names = [str(x) for x in class_names_arr.tolist()] if class_names_arr is not None else None

    print("=== Dataset Analysis ===")
    train_info = summarize_split("train", train, class_names)
    val_info = summarize_split("val", val, class_names)
    test_info = summarize_split("test", test, class_names)

    overlap = check_overlap(
        set(train_info["scenario_ids"]),
        set(val_info["scenario_ids"]),
        set(test_info["scenario_ids"]),
    )
    no_overlap = all(len(v) == 0 for v in overlap.values())
    print("\n[overlap_check]")
    print(f"  no_scenario_id_overlap: {no_overlap}")
    print(f"  overlaps: {overlap}")

    # overall summary
    y_all = np.concatenate(
        [
            train["Y_event_cls"].astype(np.int64),
            val["Y_event_cls"].astype(np.int64),
            test["Y_event_cls"].astype(np.int64),
        ]
    )
    overall_dist = class_distribution(y_all, class_names)

    report = {
        "splits": {
            "train": {k: v for k, v in train_info.items() if k != "scenario_ids"},
            "val": {k: v for k, v in val_info.items() if k != "scenario_ids"},
            "test": {k: v for k, v in test_info.items() if k != "scenario_ids"},
        },
        "class_names": class_names,
        "overall_class_distribution": overall_dist,
        "overlap_check": {
            "no_scenario_id_overlap": no_overlap,
            "details": overlap,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved report: {out_path}")


if __name__ == "__main__":
    main()
