from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_series(csv_path: Path):
    by_key = {}
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row.get('scenario_id', 'scenario_0')
            ts = row['timestamp']
            key = (sid, ts)
            by_key.setdefault(key, {})[row['tag_id']] = float(row['value'])

    records = []
    for (sid, ts), tags in by_key.items():
        rec = {'scenario_id': sid, 'timestamp': ts, **tags}
        records.append(rec)
    records.sort(key=lambda x: (x['scenario_id'], x['timestamp']))
    return records


def rule_score(x):
    speed = x.get('Z2.TRAF.AGG.S01.Speed_10s', x.get('Z2.TRAF.Speed', 100.0))
    density = x.get('Z2.TRAF.AGG.S01.Density_10s', x.get('Z2.TRAF.Density', 0.0))
    co = x.get('Z2.ENV.AGG.S01.CO_10s', x.get('Z2.CO.S01.Value', 0.0))

    score = 0.0
    score += max(0.0, min(1.0, (35.0 - float(speed)) / 35.0))
    score += max(0.0, min(1.0, (float(density) - 80.0) / 80.0))
    score += max(0.0, min(1.0, (float(co) - 120.0) / 120.0))
    return min(1.0, score / 2.0)


def true_label(x):
    return 1 if x.get('Z3.EVT.Incident.Active', x.get('Z2.EVENT.IncidentFlag', 0.0)) >= 0.5 else 0


def metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    eps = 1e-9
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    far = fp / (fp + tn + eps)
    return {'precision': p, 'recall': r, 'f1': f1, 'far': far, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def false_alarms_per_hour(records, y_pred):
    # per scenario duration from first/last timestamp
    grouped = defaultdict(list)
    for rec, pred in zip(records, y_pred):
        grouped[rec['scenario_id']].append((rec['timestamp'], true_label(rec), pred))

    fp_total = 0
    hours_total = 0.0
    for _, rows in grouped.items():
        rows.sort(key=lambda x: x[0])
        fp_total += sum(1 for _, y, p in rows if y == 0 and p == 1)
        if len(rows) >= 2:
            t0 = datetime.fromisoformat(rows[0][0])
            t1 = datetime.fromisoformat(rows[-1][0])
            hours_total += max(1e-6, (t1 - t0).total_seconds() / 3600.0)
        else:
            hours_total += 1.0 / 3600.0
    return fp_total / max(hours_total, 1e-6)


def eval_records(records, threshold=0.5):
    y_true = [true_label(x) for x in records]
    scores = [rule_score(x) for x in records]
    y_pred = [1 if s >= threshold else 0 for s in scores]
    m = metrics(y_true, y_pred)
    m['false_alarms_per_hour'] = false_alarms_per_hour(records, y_pred)
    return m


def with_noise(records, sigma_speed=3.0, sigma_density=5.0, sigma_co=8.0, seed=42):
    rng = random.Random(seed)
    out = []
    for x in records:
        y = dict(x)
        if 'Z2.TRAF.AGG.S01.Speed_10s' in y:
            y['Z2.TRAF.AGG.S01.Speed_10s'] = max(0.0, y['Z2.TRAF.AGG.S01.Speed_10s'] + rng.gauss(0.0, sigma_speed))
        elif 'Z2.TRAF.Speed' in y:
            y['Z2.TRAF.Speed'] = max(0.0, y['Z2.TRAF.Speed'] + rng.gauss(0.0, sigma_speed))
        if 'Z2.TRAF.AGG.S01.Density_10s' in y:
            y['Z2.TRAF.AGG.S01.Density_10s'] = max(0.0, y['Z2.TRAF.AGG.S01.Density_10s'] + rng.gauss(0.0, sigma_density))
        elif 'Z2.TRAF.Density' in y:
            y['Z2.TRAF.Density'] = max(0.0, y['Z2.TRAF.Density'] + rng.gauss(0.0, sigma_density))
        if 'Z2.ENV.AGG.S01.CO_10s' in y:
            y['Z2.ENV.AGG.S01.CO_10s'] = max(0.0, y['Z2.ENV.AGG.S01.CO_10s'] + rng.gauss(0.0, sigma_co))
        elif 'Z2.CO.S01.Value' in y:
            y['Z2.CO.S01.Value'] = max(0.0, y['Z2.CO.S01.Value'] + rng.gauss(0.0, sigma_co))
        out.append(y)
    return out


def with_missing(records, p_drop=0.1, seed=42):
    rng = random.Random(seed)
    keys = ['Z2.TRAF.AGG.S01.Speed_10s', 'Z2.TRAF.AGG.S01.Density_10s', 'Z2.ENV.AGG.S01.CO_10s', 'Z2.TRAF.Speed', 'Z2.TRAF.Density', 'Z2.CO.S01.Value']
    out = []
    for x in records:
        y = dict(x)
        for k in keys:
            if k in y and rng.random() < p_drop:
                del y[k]
        out.append(y)
    return out


def scenario_shift_test(records, threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7)):
    sids = sorted({x['scenario_id'] for x in records})
    cut = max(1, int(len(sids) * 0.7))
    train_ids = set(sids[:cut])
    test_ids = set(sids[cut:])
    if not test_ids:
        test_ids = set(sids[-1:])

    train = [x for x in records if x['scenario_id'] in train_ids]
    test = [x for x in records if x['scenario_id'] in test_ids]

    best_t = 0.5
    best_f1 = -1.0
    for t in threshold_grid:
        m = eval_records(train, threshold=t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_t = t

    m_test = eval_records(test, threshold=best_t)
    m_test['selected_threshold'] = best_t
    m_test['train_scenarios'] = len(train_ids)
    m_test['test_scenarios'] = len(test_ids)
    return m_test


def main():
    ap = argparse.ArgumentParser(description='Run robustness/stress tests for rule baseline')
    ap.add_argument('--csv', default='data/raw/all_runs.csv')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parents[1] / csv_path

    records = load_series(csv_path)

    base = eval_records(records, threshold=args.threshold)
    noise = eval_records(with_noise(records), threshold=args.threshold)
    missing = eval_records(with_missing(records), threshold=args.threshold)
    shift = scenario_shift_test(records)

    print('# Robustness Report')
    print(f'csv={csv_path}')
    print('\nbase:', base)
    print('\nnoise_stress:', noise)
    print('\nmissing_data_stress:', missing)
    print('\nscenario_shift:', shift)


if __name__ == '__main__':
    main()
