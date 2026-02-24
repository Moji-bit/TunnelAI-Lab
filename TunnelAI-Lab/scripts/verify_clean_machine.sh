#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/verify_clean_machine.sh
# Verifies the reproducible-environment checklist on a clean machine.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not installed or not on PATH."
  exit 1
fi

ENV_NAME="tunnelai"

echo "[1/5] Removing old env if exists: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

echo "[2/5] Creating env from environment.lock.yml"
conda env create -f environment.lock.yml

echo "[3/5] Running import sanity checks"
conda run -n "$ENV_NAME" python - <<'PY'
import numpy
import pandas
import yaml
import streamlit
print('imports_ok')
PY

echo "[4/5] Running matrix planner"
conda run -n "$ENV_NAME" python scripts/run_experiment_matrix.py --mode plan

echo "[5/5] Running recorder smoke test"
conda run -n "$ENV_NAME" python streaming/run_record.py --scenario scenarios/stau_case_00.json --out data/raw/_clean_check.csv --max-seconds 2

echo "[OK] Clean-machine verification passed."
