# Metrics and Evaluation Protocol

This document fulfills thesis checklist section **5) Metrics and Evaluation**.

## Event Metrics
Computed in `scripts/evaluate_rule_baseline.py` using helpers in `evaluation/metrics.py`:
- Precision / Recall / F1 / FAR
- PR-AUC
- Brier score (reliability)
- Lead-time in seconds before incident onset

## Forecast Metrics
Naive persistence forecast baseline reports:
- MAE / RMSE for:
  - `Z2.TRAF.Speed`
  - `Z2.CO.S01.Value`
  - `Z2.VIS.S01.Value`

## Confidence Intervals / Repeated Statistics
- Script computes chunk-based repeated F1 estimates and reports a 95% CI via bootstrap quantiles (`bootstrap_ci`).
- Recommended thesis practice: replace chunks with per-seed/per-scenario repeated runs and aggregate CI over those runs.

## Run Command
```bash
python scripts/evaluate_rule_baseline.py --csv data/raw/all_runs.csv --h 60 --threshold 0.5
```

## Notes
- Event horizon label: positive if any incident is active in `(t, t+H]`.
- If only legacy event tag exists (`Z2.EVENT.IncidentFlag`), evaluation script automatically falls back to it.
