@echo off
setlocal

cd /d %~dp0\..

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pandas') else 1)"
if errorlevel 1 (
  echo [warn] pandas missing: skipping merge_csv.py, using existing data/raw/all_runs.csv if present.
) else (
  python dataset/merge_csv.py
)

python scripts/report_label_quality.py --csv data/raw/all_runs.csv --h 60
if errorlevel 1 exit /b 1
python scripts/evaluate_rule_baseline.py --csv data/raw/all_runs.csv --h 60 --threshold 0.5
if errorlevel 1 exit /b 1
python scripts/run_robustness_tests.py --csv data/raw/all_runs.csv --threshold 0.5
if errorlevel 1 exit /b 1

endlocal
