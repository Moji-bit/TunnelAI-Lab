# TunnelAI-Lab Streamlit App

## Start (One-Click)
- Windows: double-click `run_app.bat`

## Start (Manual)
```bash
conda env create -f environment.yml
conda activate tunnelai
streamlit run apps/ui/dashboard.py
```

## Reproducible Setup (recommended for thesis)
```bash
conda env create -f environment.lock.yml
conda activate tunnelai
bash scripts/verify_clean_machine.sh
```

## Experiment Workflow
- Default matrix: `configs/experiment_matrix.csv`
- Automation helper: `python scripts/run_experiment_matrix.py --mode plan`
- Label quality spec: `LABEL_QUALITY.md`
- Label quality report: `python scripts/report_label_quality.py --csv data/raw/all_runs.csv --h 60`
- Model baselines: `MODEL_BASELINES.md`
- Run rule baseline: `python scripts/run_baselines.py --model rule --csv data/raw/all_runs.csv`
- Build NPZ datasets: `python core/dataset/dataset_builder.py`
- Train MultiTask model: `python scripts/train_multitask.py --train data/processed/train.npz --val data/processed/val.npz --test data/processed/test.npz`
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

## External Sensor Data Augmentation (4-file format)

Wenn du externe Rohdaten in vier CSV-Dateien hast (Labels, Sensors, Tunnel, Scenario-Meta),
kannst du daraus direkt 10.000 augmentierte Szenarien erzeugen:

```bash
python scripts/augment_external_scenarios.py \
  --labels-csv data/external/labels.csv \
  --sensors-csv data/external/sensors.csv \
  --tunnel-csv data/external/tunnel.csv \
  --scenario-csv data/external/scenario.csv \
  --out-dir data/augmented \
  --n-scenarios 10000 \
  --seed 42
```

Outputs:
- `data/augmented/augmented_scenarios.csv`
- `data/augmented/augmented_sensors.csv`
- `data/augmented/augmented_labels.csv`
- `data/augmented/augmented_tunnels.csv`
- `data/augmented/augmentation_manifest.json`


## Dataset Builder (4-CSV Upload + Augmentation)
Nach dem Start der Streamlit-App findest du eine zusätzliche Seite **Dataset Builder + Data Augmentation**.

Workflow:
1. Lade die vier Dateien hoch: `tunnel_config.csv`, `scenario_metadata.csv`, `timeseries.csv`, `ground_truth.csv`.
2. Prüfe den Schema-/Konsistenzreport.
3. Konfiguriere Zielanzahl, Noise, Event-Shift, Missing-Rate und Class-Balance.
4. Klicke **Generate Augmented Dataset**.
5. Nutze Preview (Original vs. augmentiert) und exportiere:
   - `augmented_timeseries.csv`
   - `augmented_ground_truth.csv`
   - `augmented_scenario_metadata.csv`
   - `augmented_tunnel_config.csv`
   - `merged_training_dataset.csv`

## Troubleshooting (Windows)
- If you get `bash` not found: use `.bat` scripts from Anaconda Prompt.
- If `scripts\run_reports_after_batch.bat` is not found, try either:
  - `.\scripts\run_reports_after_batch.bat`
  - `run_reports_after_batch.bat`
  - `python scripts\run_reports_after_batch.py`
- If you still see old errors, update local repo first:
```bat
git pull
```

## Scope (vereinfacht)
- Kein separates Backend/Frontend mehr.
- Fokus nur auf Streamlit-App (`apps/ui/dashboard.py`), KI-Modelle, Parameter, Tags und Auswertungs-Skripte.
