from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_timeseries(rows):
    by_ts = {}
    for row in rows:
        ts = row['timestamp']
        tag = row['tag_id']
        val = float(row['value'])
        by_ts.setdefault(ts, {})[tag] = val
    return [by_ts[k] for k in sorted(by_ts.keys())]


def rule_based_predictions(series):
    y_true = []
    y_pred = []
    for x in series:
        speed = x.get('Z2.TRAF.Speed', 100.0)
        density = x.get('Z2.TRAF.Density', 0.0)
        co = x.get('Z2.CO.S01.Value', 0.0)
        event = x.get('Z2.EVENT.IncidentFlag', x.get('Z3.EVT.Incident.Active', 0.0))
        pred = 1 if (speed < 35.0 or density > 80.0 or co > 120.0) else 0
        y_true.append(1 if event >= 0.5 else 0)
        y_pred.append(pred)
    return y_true, y_pred


def prf(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    far = fp / (fp + tn + eps)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'far': far, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    p = argparse.ArgumentParser(description='Run baseline models on long-format CSV')
    p.add_argument('--csv', default='data/raw/all_runs.csv')
    p.add_argument('--model', choices=['rule', 'logreg'], default='rule')
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parents[1] / csv_path

    rows = load_rows(csv_path)
    series = to_timeseries(rows)

    if args.model == 'rule':
        y_true, y_pred = rule_based_predictions(series)
        print('model=rule')
        print(prf(y_true, y_pred))
        return

    # optional classical baseline
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as exc:
        print(f'[warn] Missing dependency for logreg baseline: {getattr(exc, "name", "unknown")}.')
        print('Install scikit-learn and numpy in your thesis env to run --model logreg.')
        return

    X, y = [], []
    for x in series:
        X.append([
            x.get('Z2.TRAF.Speed', 100.0),
            x.get('Z2.TRAF.Density', 0.0),
            x.get('Z2.CO.S01.Value', 0.0),
            x.get('Z2.VIS.S01.Value', 100.0),
            x.get('Z2.VMS.SpeedLimit', 90.0),
            x.get('Z2.FAN.StageCmd', 0.0),
        ])
        y.append(1 if x.get('Z2.EVENT.IncidentFlag', x.get('Z3.EVT.Incident.Active', 0.0)) >= 0.5 else 0)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(X)
    n_train = int(0.8 * n)
    clf = LogisticRegression(max_iter=200, class_weight='balanced', random_state=42)
    clf.fit(X[:n_train], y[:n_train])
    y_pred = clf.predict(X[n_train:])
    m = prf(y[n_train:].tolist(), y_pred.tolist())
    print('model=logreg')
    print(m)


if __name__ == '__main__':
    main()
