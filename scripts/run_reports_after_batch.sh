#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('pandas') else 1)
PY
then
  python dataset/merge_csv.py
else
  echo "[warn] pandas missing: skipping merge_csv.py, using existing data/raw/all_runs.csv if present."
fi

python scripts/report_label_quality.py --csv data/raw/all_runs.csv --h 60
python scripts/evaluate_rule_baseline.py --csv data/raw/all_runs.csv --h 60 --threshold 0.5
python scripts/run_robustness_tests.py --csv data/raw/all_runs.csv --threshold 0.5
