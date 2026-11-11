"""Group-conditional (Mondrian) split conformal for multilabel ECG (stage 6).

Per (group g, label c): binary prediction sets over {0, 1}.
Nonconformity of value v given prob p: s = 1 - p if v == 1 else p.
Threshold q[g, c] = ceil((n+1)(1-alpha))/n empirical quantile of the
calibration scores OF THE TRUE VALUES within group g, label c.
Prediction set includes v iff s(v) <= q[g, c]. Guarantees per-group,
per-label coverage >= 1 - alpha under exchangeability; coverage DEGRADATION
on a shifted held-out cohort is itself a reported finding.

Fallback: a group too small to calibrate (n < min_n) uses the marginal
(all-groups) threshold for that label.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import numpy as np

MIN_CAL_N = 50


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0  # degenerate: include everything
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def _true_value_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 1.0, 1.0 - probs, probs)


class LabelConditionalMondrian:
    """Mondrian conformal conditioned on (group g, label c, TRUE value v).

    Amendment 1 (2026-11-09): the (g, c)-only version pools the abundant
    easy negatives with the rare positives, so its threshold becomes too
    tight and ~14% of cells got EMPTY sets on the PTB-XL test, with positive
    (sensitivity) coverage unguaranteed. Conditioning on the true value gives
    per-group, per-label coverage >= 1 - alpha separately for positives and
    negatives; ambiguous cells get {0, 1} (abstain/refer) instead of {}.

    q[g, c, v] from calibration cells with group g, label c, y == v;
    fallback to (c, v) pooled over groups when that cell count < min_n.
    """

    def __init__(self, n_groups: int = 8, alpha: float = 0.1,
                 min_n: int = 30):
        self.alpha, self.n_groups, self.min_n = alpha, n_groups, min_n
        self.q: np.ndarray | None = None  # (n_groups, C, 2)
        self.n_fallback = 0

    def calibrate(self, probs: np.ndarray, y: np.ndarray,
                  group: np.ndarray) -> "LabelConditionalMondrian":
        N, C = probs.shape
        self.q = np.ones((self.n_groups, C, 2))
        self.n_fallback = 0
        for c in range(C):
            for v in (0, 1):
                s_v = probs[:, c] if v == 0 else 1.0 - probs[:, c]
                is_v = y[:, c] == v  # NaN compares False -> excluded
                q_pooled = _conformal_quantile(s_v[is_v], self.alpha)
                for g in range(self.n_groups):
                    sel = is_v & (group == g)
                    if sel.sum() >= self.min_n:
                        self.q[g, c, v] = _conformal_quantile(s_v[sel], self.alpha)
                    else:
                        self.q[g, c, v] = q_pooled
                        self.n_fallback += 1
        return self

    def predict_sets(self, probs: np.ndarray, group: np.ndarray) -> np.ndarray:
        gi = np.clip(group, 0, self.n_groups - 1)
        q = self.q[gi]                                   # (N, C, 2)
        # group -1 (missing demographics): use the max over groups (conservative)
        if (group < 0).any():
            q[group < 0] = self.q.max(axis=0)
        inc0 = probs <= q[..., 0]
        inc1 = (1.0 - probs) <= q[..., 1]
        return np.stack([inc0, inc1], axis=-1)

    def coverage(self, probs: np.ndarray, y: np.ndarray,
                 group: np.ndarray) -> dict:
        sets = self.predict_sets(probs, group)
        valid = ~np.isnan(y)
        yv = np.nan_to_num(y, nan=0.0).astype(int)
        covered = np.take_along_axis(sets, yv[..., None], axis=-1)[..., 0]
        size = sets.sum(-1).astype(float)
        pos, neg = valid & (yv == 1), valid & (yv == 0)
        out = {"marginal_coverage": float(covered[valid].mean()),
               "positive_coverage": float(covered[pos].mean()) if pos.any() else None,
               "negative_coverage": float(covered[neg].mean()) if neg.any() else None,
               "mean_set_size": float(size[valid].mean()),
               "empty_set_rate": float((size[valid] == 0).mean()),
               "abstain_rate": float((size[valid] == 2).mean()),
               "n_fallback_cells": int(self.n_fallback),
               "per_group": {}}
        for g in sorted(set(group[group >= 0].tolist())):
            gm = (group == g)[:, None]
            sel, selp = gm & valid, gm & pos
            if sel.sum() >= 50:
                out["per_group"][int(g)] = {
                    "coverage": float(covered[sel].mean()),
                    "positive_coverage": (float(covered[selp].mean())
                                          if selp.sum() >= 20 else None),
                    "n_pos_cells": int(selp.sum()),
                    "mean_set_size": float(size[sel].mean())}
        return out


class MondrianConformal:
    def __init__(self, n_groups: int = 8, alpha: float = 0.1,
                 min_n: int = MIN_CAL_N):
        self.alpha, self.n_groups, self.min_n = alpha, n_groups, min_n
        self.q: np.ndarray | None = None          # (n_groups, C)
        self.q_marginal: np.ndarray | None = None  # (C,)

    def calibrate(self, probs: np.ndarray, y: np.ndarray,
                  group: np.ndarray) -> "MondrianConformal":
        N, C = probs.shape
        scores = _true_value_scores(probs, y)
        self.q_marginal = np.array([
            _conformal_quantile(scores[~np.isnan(y[:, c]), c], self.alpha)
            for c in range(C)])
        self.q = np.tile(self.q_marginal, (self.n_groups, 1))
        for g in range(self.n_groups):
            for c in range(C):
                sel = (group == g) & ~np.isnan(y[:, c])
                if sel.sum() >= self.min_n:
                    self.q[g, c] = _conformal_quantile(scores[sel, c], self.alpha)
        return self

    def _thresholds(self, group: np.ndarray) -> np.ndarray:
        q = np.where((group >= 0)[:, None],
                     self.q[np.clip(group, 0, self.n_groups - 1)],
                     self.q_marginal[None, :])
        return q  # (N, C)

    def predict_sets(self, probs: np.ndarray, group: np.ndarray) -> np.ndarray:
        """-> (N, C, 2) bool: membership of value 0 and value 1."""
        q = self._thresholds(group)
        inc1 = (1.0 - probs) <= q
        inc0 = probs <= q
        return np.stack([inc0, inc1], axis=-1)

    def coverage(self, probs: np.ndarray, y: np.ndarray,
                 group: np.ndarray) -> dict:
        """Marginal + per-group empirical coverage and mean set size,
        pooled over defined (sample, label) cells."""
        sets = self.predict_sets(probs, group)
        valid = ~np.isnan(y)
        yv = np.nan_to_num(y, nan=0.0).astype(int)
        covered = np.take_along_axis(sets, yv[..., None], axis=-1)[..., 0]
        size = sets.sum(-1).astype(float)
        out = {"marginal_coverage": float(covered[valid].mean()),
               "mean_set_size": float(size[valid].mean()),
               "empty_set_rate": float((size[valid] == 0).mean()),
               "per_group": {}}
        for g in sorted(set(group[group >= 0].tolist())):
            sel = (group == g)[:, None] & valid
            if sel.sum() >= self.min_n:
                out["per_group"][int(g)] = {
                    "coverage": float(covered[sel].mean()),
                    "mean_set_size": float(size[sel].mean()),
                    "n_cells": int(sel.sum())}
        return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Post-hoc temperature minimizing NLL on DEFINED cells. Continuous
    bounded 1-D search over log T (a coarse grid previously produced
    identical T across seeds — a resolution artifact)."""
    from scipy.optimize import minimize_scalar
    valid = ~np.isnan(y)
    z, t = logits[valid].astype(np.float64), y[valid].astype(np.float64)

    def nll(log_t: float) -> float:
        zz = z / np.exp(log_t)
        # numerically stable BCE-with-logits
        return float(np.mean(np.maximum(zz, 0) - zz * t + np.log1p(np.exp(-np.abs(zz)))))

    res = minimize_scalar(nll, bounds=(np.log(0.25), np.log(8.0)),
                          method="bounded", options={"xatol": 1e-5})
    return float(np.exp(res.x))
