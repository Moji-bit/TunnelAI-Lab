# Thesis Experiment Matrix (Tunnel Incident Prediction)

This file provides a ready-to-use experiment plan for your master thesis.

## Research Question
Can AI predict tunnel incidents earlier and more reliably than rule-based baselines under varying traffic and environmental conditions?

## Common Setup
- Input window length `L`: 300 s
- Forecast horizons `H`: 30, 60, 120, 300 s
- Stride: 5 s
- Split strategy: scenario-wise split to avoid leakage
- Seeds: 42, 43, 50, 51, 60

## Datasets
- Train/Val/Test from generated scenarios in `scenarios/*.json`
- Include scenario variability:
  - inflow level + amplitude
  - incident start/end
  - capacity drop
  - speed-limit and fan-stage settings

## Model Families
1. Rule-based threshold baseline
2. Logistic Regression / XGBoost baseline
3. LSTM multitask model
4. Transformer multitask model

## Metrics
### Event Detection
- Precision, Recall, F1, FAR
- PR-AUC
- Lead-time to incident onset
- False alarms per hour

### Forecasting
- MAE, RMSE for selected targets (Speed, CO, VIS)

### Reliability
- Brier score / calibration curve

## Ablation Dimensions
- Backbone: LSTM vs Transformer
- Horizon: 30/60/120/300
- Feature sets:
  - traffic-only
  - traffic+environment
  - full (traffic+environment+actuators)
- Noise robustness:
  - nominal
  - high sensor noise
  - missing values (dropout)

## Table Template
Use `thesis_experiment_matrix.csv` for run tracking.

Columns:
- run_id
- seed
- split_id
- model
- backbone
- horizon_s
- feature_set
- noise_profile
- threshold
- precision
- recall
- f1
- far
- pr_auc
- lead_time_s
- false_alarms_per_h
- mae_speed
- rmse_speed
- mae_co
- rmse_co
- mae_vis
- rmse_vis
- brier
- notes


## Automation Helper
Use the helper script to inspect and pre-build datasets by horizon:

```bash
python scripts/run_experiment_matrix.py --mode plan
python scripts/run_experiment_matrix.py --mode build-datasets --csv-path data/raw/all_runs.csv
```

## Minimum Experiment Set (recommended)
- 4 model families × 4 horizons × 3 seeds = 48 core runs
- + 24 robustness runs (noise/missingness)
- + 16 ablation runs (feature subsets)
- Total ≈ 88 runs

## Reporting Structure (Thesis)
1. Data generation and assumptions
2. Model setup and baselines
3. Main performance table (event + forecast)
4. Early-warning analysis (lead-time)
5. Robustness analysis
6. Operational implications (false alarms vs detection gain)
