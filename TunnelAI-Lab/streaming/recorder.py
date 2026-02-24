# streaming/recorder.py
import csv
import os
from typing import Dict, Iterable, Tuple

def write_long_csv(
    out_path: str,
    rows: Iterable[Tuple[str, str, float, str, str]],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "tag_id", "value", "quality", "scenario_id"])
        for r in rows:
            w.writerow(list(r))