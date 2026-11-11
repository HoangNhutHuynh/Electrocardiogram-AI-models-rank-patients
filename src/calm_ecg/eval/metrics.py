"""Evaluation metrics (stage 5 model selection + stage 6 final tables).

All metrics respect NaN = undefined label. Gate (quy trình, stage 6):
self-written metrics must match sklearn to 1e-6 on random data — the unit
tests enforce this by construction (we call sklearn per label and test the
masking logic separately).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

MIN_GROUP_N = 50  # a group needs this many samples AND both classes to count


def masked_macro_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """y: (N, C) in {0,1,NaN}; scores: (N, C). Macro over labels computable
    on their DEFINED entries; labels with one class present are skipped."""
    aucs = []
    for c in range(y.shape[1]):
        valid = ~np.isnan(y[:, c])
        yc = y[valid, c]
        if valid.sum() == 0 or len(np.unique(yc)) < 2:
            continue
        aucs.append(roc_auc_score(yc, scores[valid, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_group_auroc(y: np.ndarray, scores: np.ndarray,
                    group: np.ndarray) -> dict[int, float]:
    out = {}
    for g in sorted(set(group[group >= 0].tolist())):
        sel = group == g
        if sel.sum() < MIN_GROUP_N:
            continue
        auc = masked_macro_auroc(y[sel], scores[sel])
        if np.isfinite(auc):
            out[int(g)] = auc
    return out


def worst_group_auroc(y: np.ndarray, scores: np.ndarray,
                      group: np.ndarray) -> float:
    pg = per_group_auroc(y, scores, group)
    return min(pg.values()) if pg else float("nan")


def expected_calibration_error(y: np.ndarray, probs: np.ndarray,
                               n_bins: int = 15) -> float:
    """Top-label-free multilabel ECE: pooled over all DEFINED (sample, label)
    cells, equal-width confidence bins."""
    valid = ~np.isnan(y)
    p, t = probs[valid].ravel(), y[valid].ravel()
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece, n = 0.0, len(p)
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            ece += sel.sum() / n * abs(p[sel].mean() - t[sel].mean())
    return float(ece)
