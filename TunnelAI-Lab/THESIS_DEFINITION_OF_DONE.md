# Thesis Definition of Done (Tunnel Incident Prediction)

Use this checklist to decide when your app/system is "thesis-ready".

## 1) Reproducible Environment
- [ ] `environment.yml` installs successfully on a clean machine.
- [ ] Exact Python/package versions are pinned and documented.
- [ ] A fresh clone can run the full pipeline end-to-end.

## 2) Simulation/Data Quality
- [ ] At least 3 incident types are simulated (e.g., crash, stopped vehicle, smoke).
- [ ] Incident severity and duration vary across scenarios.
- [ ] Multiple seeds are used and logged for reproducibility.
- [ ] Train/val/test split is scenario-wise (no leakage).

## 3) Label Quality
- [ ] Event labels are clearly defined (onset, active, offset).
- [ ] Prediction horizon labels (`H`) are documented and validated.
- [ ] Class imbalance is measured and reported.

## 4) Baselines and Models
- [ ] Rule-based baseline implemented and evaluated.
- [ ] At least one classical ML baseline (e.g., Logistic/XGBoost).
- [ ] At least two deep baselines (LSTM + Transformer).
- [ ] Hyperparameter search space and final configs are documented.

## 5) Metrics and Evaluation
- [ ] Event metrics: Precision, Recall, F1, FAR, PR-AUC.
- [ ] Forecast metrics: MAE, RMSE for key variables.
- [ ] Early-warning metric: lead-time before incident onset.
- [ ] Reliability metric: Brier score or calibration curve.
- [ ] Results include confidence intervals or repeated-seed statistics.

## 6) Robustness / Stress Tests
- [ ] Sensor noise robustness test completed.
- [ ] Missing-data robustness test completed.
- [ ] Scenario shift test completed (unseen traffic patterns).
- [ ] False alarms per hour evaluated for operational realism.

## 7) App/UI Readiness
- [ ] Playback runs without severe flicker/freezes on target hardware.
- [ ] Scenario runner reliably writes output CSVs.
- [ ] Dashboard correctly displays status and event context.
- [ ] Key controls are documented (start/pause/reset, speed, window).

## 8) Documentation & Thesis Artifacts
- [ ] Experiment matrix is filled (`thesis_experiment_matrix.csv`).
- [ ] Methodology is reproducible from repo docs alone.
- [ ] Limitations and threats to validity are clearly stated.
- [ ] Ethical/safety implications for tunnel operations are discussed.

## 9) Final Acceptance Criteria (Go/No-Go)
Declare thesis system "done" only if all are true:
- [ ] End-to-end run succeeds on clean setup.
- [ ] Baseline and AI models are both evaluated on same splits.
- [ ] AI improves at least one key KPI (e.g., F1 or lead-time) over baseline.
- [ ] Operational trade-off is reported (detection gain vs false alarms).

---

## Quick Weekly Review (recommended)
- Week objective met? [ ]
- New failure mode discovered? [ ]
- Dataset/labels changed (and versioned)? [ ]
- Re-ran baseline for fair comparison? [ ]
- Thesis text updated with latest evidence? [ ]
