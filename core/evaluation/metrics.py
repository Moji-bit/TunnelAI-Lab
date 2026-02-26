# evaluation/metrics.py
from __future__ import annotations

import math


def _to_float_list(xs):
    return [float(x) for x in xs]


def mae(y_true, y_pred) -> float:
    yt = _to_float_list(y_true)
    yp = _to_float_list(y_pred)
    if not yt:
        return 0.0
    return sum(abs(a - b) for a, b in zip(yt, yp)) / len(yt)


def rmse(y_true, y_pred) -> float:
    yt = _to_float_list(y_true)
    yp = _to_float_list(y_pred)
    if not yt:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(yt, yp)) / len(yt))


def precision_recall_f1(y_true_bin, y_pred_bin):
    y_true = [1 if int(float(v)) >= 1 else 0 for v in y_true_bin]
    y_pred = [1 if int(float(v)) >= 1 else 0 for v in y_pred_bin]
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    far = fp / (fp + tn + eps)
    return {"precision": prec, "recall": rec, "f1": f1, "far": far, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def brier_score(y_true_bin, y_prob):
    y_true = _to_float_list(y_true_bin)
    y_prob = [min(1.0, max(0.0, float(p))) for p in y_prob]
    if not y_true:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_prob)) / len(y_true)


def pr_auc(y_true_bin, y_score):
    pairs = sorted([(float(s), int(float(y) >= 1)) for y, s in zip(y_true_bin, y_score)], reverse=True)
    total_pos = sum(y for _, y in pairs)
    if total_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    prec = [1.0]
    rec = [0.0]
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        prec.append(tp / max(1, tp + fp))
        rec.append(tp / total_pos)

    area = 0.0
    for i in range(1, len(prec)):
        dx = rec[i] - rec[i - 1]
        area += dx * (prec[i] + prec[i - 1]) / 2
    return area


def lead_time_seconds(event_active, pred_score, threshold=0.5, step_s=1):
    active = [1 if int(float(x)) >= 1 else 0 for x in event_active]
    score = [float(x) for x in pred_score]
    onsets = [i for i in range(1, len(active)) if active[i] == 1 and active[i - 1] == 0]
    if not onsets:
        return 0.0

    leads = []
    for onset in onsets:
        first_alarm = None
        for i in range(0, onset + 1):
            if score[i] >= threshold:
                first_alarm = i
                break
        if first_alarm is None:
            leads.append(0.0)
        else:
            leads.append(max(0.0, (onset - first_alarm) * step_s))
    return sum(leads) / len(leads)


def bootstrap_ci(values, alpha=0.95):
    arr = sorted(float(v) for v in values)
    if not arr:
        return (0.0, 0.0)
    lo = (1 - alpha) / 2
    hi = 1 - lo
    ilo = min(len(arr) - 1, max(0, int(lo * (len(arr) - 1))))
    ihi = min(len(arr) - 1, max(0, int(hi * (len(arr) - 1))))
    return (arr[ilo], arr[ihi])
