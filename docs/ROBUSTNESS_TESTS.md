# Robustness / Stress Tests

This document covers thesis checklist section **6) Robustness / Stress Tests**.

## Implemented Tests

1. **Sensor noise robustness**
   - Adds Gaussian noise to traffic speed, density, and CO signals.
   - Reports event metrics and false alarms per hour.

2. **Missing-data robustness**
   - Randomly drops key sensor inputs (`Speed`, `Density`, `CO`) with configurable drop probability.
   - Re-evaluates event metrics and false alarms per hour.

3. **Scenario shift (unseen patterns)**
   - Splits data by scenario ID (train scenarios vs unseen test scenarios).
   - Selects threshold on train set and evaluates on unseen test scenarios.

4. **Operational realism**
   - Reports **false alarms per hour** in all robustness outputs.

## Run Command
```bash
python scripts/run_robustness_tests.py --csv data/raw/all_runs.csv --threshold 0.5
```

## Output
The script prints:
- `base` metrics
- `noise_stress` metrics
- `missing_data_stress` metrics
- `scenario_shift` metrics

Each block includes:
- precision, recall, f1, far
- confusion counts
- false_alarms_per_hour
