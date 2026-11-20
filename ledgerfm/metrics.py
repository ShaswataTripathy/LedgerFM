"""Shared evaluation metrics (fixed before any test run)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def recall_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])


def f1_at(y, s, thr):
    pred = s >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(y, s, thr=None):
    """PR-AUC is primary. If thr is None, F1 uses the best threshold on this set (only allowed on val)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    out = {
        "n": int(len(y)), "pos": int(y.sum()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "rec@0.1%fpr": recall_at_fpr(y, s, 0.001),
        "rec@1%fpr": recall_at_fpr(y, s, 0.01),
    }
    if thr is None:
        thr, f1 = best_f1_threshold(y, s)
        out.update(f1=f1, thr=thr)
    else:
        out.update(f1=f1_at(y, s, thr), thr=thr)
    return out


def label_keep(txn_id, frac, seed=0):
    """Low-label regime: deterministic subset of labelled training rows, shared by every model.
    splitmix64 hash of txn_id; keeps rows with hash % 1000 < frac*1000."""
    x = (np.asarray(txn_id, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1))
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(1000)) < np.uint64(int(round(frac * 1000)))
