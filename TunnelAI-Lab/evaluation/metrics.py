# evaluation/metrics.py
import numpy as np

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def precision_recall_f1(y_true_bin, y_pred_bin):
    y_true = np.asarray(y_true_bin).astype(int)
    y_pred = np.asarray(y_pred_bin).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    far = fp / (fp + tn + eps)
    return {"precision": prec, "recall": rec, "f1": f1, "far": far, "tp": tp, "fp": fp, "fn": fn, "tn": tn}