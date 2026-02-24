# TunnelAI-Lab Streamlit App

## Start (One-Click)
- Windows: double-click `run_app.bat`

## Start (Manual)
```bash
conda env create -f environment.yml
conda activate tunnelai
streamlit run ui/dashboard.py
```

## Reproducible Setup (recommended for thesis)
```bash
conda env create -f environment.lock.yml
conda activate tunnelai
bash scripts/verify_clean_machine.sh
```

## Thesis Workflow
- Experiment matrix: `thesis_experiment_matrix.md` / `thesis_experiment_matrix.csv`
- Automation helper: `python scripts/run_experiment_matrix.py --mode plan`
- Thesis DoD checklist: `THESIS_DEFINITION_OF_DONE.md`
- Label quality spec: `LABEL_QUALITY.md`
- Label quality report: `python scripts/report_label_quality.py --csv data/raw/all_runs.csv --h 60`
- Model baselines: `MODEL_BASELINES.md`
- Run rule baseline: `python scripts/run_baselines.py --model rule --csv data/raw/all_runs.csv`
- Evaluation protocol: `EVALUATION_METRICS.md`
- Run evaluation: `python scripts/evaluate_rule_baseline.py --csv data/raw/all_runs.csv --h 60 --threshold 0.5`
- Robustness tests: `ROBUSTNESS_TESTS.md`
- Run robustness: `python scripts/run_robustness_tests.py --csv data/raw/all_runs.csv --threshold 0.5`
- SCADA hybrid model: `SCADA_HYBRID_MODEL.md`

## Standardized Data Generation (recommended before big batch)
```bash
# 1) Quick reproducibility pre-check
bash scripts/verify_clean_machine.sh  # Linux/macOS

# 2) Standardized scenario-set + seeds (dry-run first)
python scripts/run_standard_data_batch.py --plan configs/data_batch_plan.json --dry-run
python scripts/run_standard_data_batch.py --plan configs/data_batch_plan.json

# 3) Auto-run quality/evaluation/robustness reports
bash scripts/run_reports_after_batch.sh  # Linux/macOS
python scripts/run_reports_after_batch.py  # cross-platform fallback
```

### Windows Batch Commands
```bat
python scripts\run_standard_data_batch.py --plan configs\data_batch_plan.json --dry-run
python scripts\run_standard_data_batch.py --plan configs\data_batch_plan.json
.\scripts\run_reports_after_batch.bat
python scripts\run_reports_after_batch.py
run_reports_after_batch.bat
```

## Troubleshooting (Windows)
- If you get `bash` not found: use `.bat` scripts from Anaconda Prompt.
- If `scripts\run_reports_after_batch.bat` is not found, try either:
  - `.\scripts\run_reports_after_batch.bat`
  - `run_reports_after_batch.bat`
  - `python scripts\run_reports_after_batch.py`
- If you still see old errors (e.g. `fan_stage_dyn`), update local repo first:
```bat
git pull
```
