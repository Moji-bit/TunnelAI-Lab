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
