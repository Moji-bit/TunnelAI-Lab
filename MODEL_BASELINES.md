# Baselines and Models

## Implemented Baselines
- Rule-based baseline: `scripts/run_baselines.py --model rule`
- Classical ML baseline (optional deps): `scripts/run_baselines.py --model logreg`

## Deep Baselines
- LSTM multitask model: `models/multitask_model.py` with `backbone="lstm"`
- Transformer multitask model: `models/multitask_model.py` with `backbone="transformer"`

## Hyperparameter Search Space

### Rule-based
- Speed threshold: [25, 35, 45] km/h
- Density threshold: [70, 80, 90] veh/km
- CO threshold: [100, 120, 150] ppm

### Logistic Regression
- `C`: [0.1, 1.0, 10.0]
- `class_weight`: [None, 'balanced']
- `max_iter`: [200, 500]

### LSTM
- `d_model`: [64, 128]
- `n_layers`: [1, 2]
- `dropout`: [0.1, 0.3]
- `lr`: [1e-3, 3e-4]
- `batch_size`: [32, 64]

### Transformer
- `d_model`: [64, 128]
- `n_heads`: [2, 4]
- `n_layers`: [2, 3]
- `dropout`: [0.1, 0.2]
- `lr`: [1e-3, 3e-4]
- `batch_size`: [32, 64]

## Final Configs (recommended starting point)
- Rule-based: speed<35 OR density>80 OR CO>120
- Logistic Regression: C=1.0, class_weight='balanced', max_iter=200
- LSTM: d_model=128, n_layers=2, dropout=0.1, lr=1e-3, batch_size=64
- Transformer: d_model=128, n_heads=4, n_layers=3, dropout=0.1, lr=3e-4, batch_size=64
