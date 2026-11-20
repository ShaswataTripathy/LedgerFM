"""Blend a fine-tuned LedgerFM with an existing model (e.g. XGBoost); the weight is chosen on validation.

    from ledgerfm import Blend
    b = Blend().fit(y_val, p_fm_val, p_xgb_val)     # probabilities in (0, 1)
    p = b.predict(p_fm_new, p_xgb_new)              # blended score (higher = riskier, not a probability)
    b.w, b.curve                                    # chosen weight on LedgerFM and val PR-AUC per weight

Each model's logit is standardised with its validation mean/std (fixed constants after fit), then
score = w * z_fm + (1 - w) * z_other. If LedgerFM adds nothing on your data, fit() picks w = 0.
"""
import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _pr_auc(y, s):
    """Average precision (same definition as sklearn.metrics.average_precision_score)."""
    y = np.asarray(y).astype(bool); o = np.argsort(-np.asarray(s), kind="mergesort")
    y = y[o]; tp = np.cumsum(y); P = tp[-1]
    if P == 0:
        return float("nan")
    s = np.asarray(s)[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    prec = tp[last] / (last + 1); rec = tp[last] / P
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


class Blend:
    def __init__(self, grid=None):
        self.grid = grid if grid is not None else [round(x, 1) for x in np.linspace(0, 1, 11)]
        self.w = None

    def fit(self, y_val, p_fm, p_other):
        lf, lo = _logit(p_fm), _logit(p_other)
        self.norm_fm, self.norm_other = (lf.mean(), lf.std() or 1.0), (lo.mean(), lo.std() or 1.0)
        zf, zo = self._z(lf, self.norm_fm), self._z(lo, self.norm_other)
        self.curve = {w: _pr_auc(y_val, w * zf + (1 - w) * zo) for w in self.grid}
        self.w = max(self.grid, key=lambda w: self.curve[w])
        return self

    @staticmethod
    def _z(l, n):
        return (l - n[0]) / n[1]

    def predict(self, p_fm, p_other):
        if self.w is None:
            raise RuntimeError("call fit() on validation data first")
        return self.w * self._z(_logit(p_fm), self.norm_fm) + (1 - self.w) * self._z(_logit(p_other), self.norm_other)
