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

