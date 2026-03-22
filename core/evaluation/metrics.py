# evaluation/metrics.py
from __future__ import annotations

import math

try:
    from sklearn.metrics import (
        accuracy_score as _sk_accuracy_score,
        classification_report as _sk_classification_report,
        confusion_matrix as _sk_confusion_matrix,
        precision_recall_fscore_support as _sk_prfs,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _to_float_list(xs):
    return [float(x) for x in xs]


def _to_int_list(xs):
    return [int(float(x)) for x in xs]


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


def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    yt = _to_int_list(y_true)
    yp = _to_int_list(y_pred)
    if labels is None:
        labels = sorted(set(yt) | set(yp))
    labels = [int(l) for l in labels]

    if _HAS_SKLEARN:
        cm = _sk_confusion_matrix(yt, yp, labels=labels)
        return cm.tolist()

    idx = {label: i for i, label in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for a, b in zip(yt, yp):
        if a in idx and b in idx:
            cm[idx[a]][idx[b]] += 1
    return cm


def accuracy_multiclass(y_true, y_pred) -> float:
    yt = _to_int_list(y_true)
    yp = _to_int_list(y_pred)
    if not yt:
        return 0.0
    if _HAS_SKLEARN:
        return float(_sk_accuracy_score(yt, yp))
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    return float(correct / len(yt))


def precision_recall_f1_multiclass(y_true, y_pred, labels=None):
    yt = _to_int_list(y_true)
    yp = _to_int_list(y_pred)
    if labels is None:
        labels = sorted(set(yt) | set(yp))
    labels = [int(l) for l in labels]

    if _HAS_SKLEARN:
        p, r, f1, s = _sk_prfs(yt, yp, labels=labels, average=None, zero_division=0)
        p_macro, r_macro, f1_macro, _ = _sk_prfs(yt, yp, labels=labels, average="macro", zero_division=0)
        return {
            "labels": labels,
            "precision_per_class": [float(x) for x in p.tolist()],
            "recall_per_class": [float(x) for x in r.tolist()],
            "f1_per_class": [float(x) for x in f1.tolist()],
            "support_per_class": [int(x) for x in s.tolist()],
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
        }

    cm = confusion_matrix_multiclass(yt, yp, labels=labels)
    n = len(labels)
    precision = []
    recall = []
    f1 = []
    support = []
    eps = 1e-9
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        sup = sum(cm[i])
        p_i = tp / (tp + fp + eps)
        r_i = tp / (tp + fn + eps)
        f_i = 2 * p_i * r_i / (p_i + r_i + eps)
        precision.append(p_i)
        recall.append(r_i)
        f1.append(f_i)
        support.append(sup)

    return {
        "labels": labels,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "precision_macro": sum(precision) / max(1, n),
        "recall_macro": sum(recall) / max(1, n),
        "f1_macro": sum(f1) / max(1, n),
    }


def classification_report_dict(y_true, y_pred, labels=None, target_names=None):
    yt = _to_int_list(y_true)
    yp = _to_int_list(y_pred)
    if labels is None:
        labels = sorted(set(yt) | set(yp))
    labels = [int(l) for l in labels]

    if _HAS_SKLEARN:
        rep = _sk_classification_report(
            yt,
            yp,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        return rep

    stats = precision_recall_f1_multiclass(yt, yp, labels=labels)
    out = {}
    names = target_names if target_names is not None else [str(l) for l in labels]
    for i, name in enumerate(names):
        out[name] = {
            "precision": float(stats["precision_per_class"][i]),
            "recall": float(stats["recall_per_class"][i]),
            "f1-score": float(stats["f1_per_class"][i]),
            "support": int(stats["support_per_class"][i]),
        }

    total_support = sum(stats["support_per_class"])
    weighted_f1 = 0.0
    weighted_p = 0.0
    weighted_r = 0.0
    for i in range(len(names)):
        w = stats["support_per_class"][i] / max(1, total_support)
        weighted_p += w * stats["precision_per_class"][i]
        weighted_r += w * stats["recall_per_class"][i]
        weighted_f1 += w * stats["f1_per_class"][i]

    out["accuracy"] = accuracy_multiclass(yt, yp)
    out["macro avg"] = {
        "precision": float(stats["precision_macro"]),
        "recall": float(stats["recall_macro"]),
        "f1-score": float(stats["f1_macro"]),
        "support": int(total_support),
    }
    out["weighted avg"] = {
        "precision": float(weighted_p),
        "recall": float(weighted_r),
        "f1-score": float(weighted_f1),
        "support": int(total_support),
    }
    return out


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
