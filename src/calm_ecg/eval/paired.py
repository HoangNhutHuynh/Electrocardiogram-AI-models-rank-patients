"""Paired bootstrap of the pre-registered primary contrast (PREREGISTRATION.md).

Δ_f = mean_seeds(worst-group AUROC of variant V) − mean_seeds(ERM), on the
held-out cohort of fold f. CI: resample TEST RECORDS with replacement, the
same indices for every model (paired), recompute every model's worst-group
AUROC, then Δ. Pooled over folds: each fold is resampled independently and
draw b of the pooled statistic is mean_f Δ_f^b.

Worst-group AUROC definition matches metrics.worst_group_auroc: groups with
>= MIN_GROUP_N records; per group, macro over labels with both classes.
A fast rank-based AUROC is used (equals sklearn's; tested).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .metrics import MIN_GROUP_N


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney AUROC with average ranks for ties (== sklearn)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def worst_group_auroc_fast(y: np.ndarray, p: np.ndarray, g: np.ndarray,
                           idx: np.ndarray | None = None) -> float:
    if idx is not None:
        y, p, g = y[idx], p[idx], g[idx]
    worst = np.inf
    for grp in np.unique(g[g >= 0]):
        sel = g == grp
        if sel.sum() < MIN_GROUP_N:
            continue
        aucs = []
        for c in range(y.shape[1]):
            v = sel & ~np.isnan(y[:, c])
            yc = y[v, c]
            if yc.size and 0 < yc.sum() < yc.size:
                aucs.append(fast_auroc(yc, p[v, c]))
        if aucs:
            worst = min(worst, float(np.mean(aucs)))
    return worst if np.isfinite(worst) else float("nan")


class _WeightedWorstGroup:
    """Worst-group AUROC of one model under per-record bootstrap weights.

    A bootstrap resample only changes how often each record appears, not the
    score order, so scores are ranked ONCE (np.unique) and each resample is a
    weighted Mann–Whitney count via bincount + cumsum. Ties (including
    duplicated records) count 1/2, exactly like average-rank AUROC.
    """

    def __init__(self, y: np.ndarray, p: np.ndarray, g: np.ndarray):
        self.groups = []
        for grp in np.unique(g[g >= 0]):
            rows = np.flatnonzero(g == grp)
            cells = []
            for c in range(y.shape[1]):
                sel = rows[~np.isnan(y[rows, c])]
                _, inv = np.unique(p[sel, c], return_inverse=True)
                cells.append((sel, inv, y[sel, c], inv.max() + 1 if inv.size else 0))
            self.groups.append((rows, cells))

    def __call__(self, w: np.ndarray) -> float:
        worst = np.inf
        for rows, cells in self.groups:
            if w[rows].sum() < MIN_GROUP_N:
                continue
            aucs = []
            for sel, inv, yc, nlev in cells:
                ws = w[sel]
                pos = np.bincount(inv, weights=ws * yc, minlength=nlev)
                neg = np.bincount(inv, weights=ws * (1 - yc), minlength=nlev)
                w1, w0 = pos.sum(), neg.sum()
                if w1 == 0 or w0 == 0:
                    continue
                below = np.cumsum(neg) - neg
                aucs.append(float((pos * (below + 0.5 * neg)).sum() / (w1 * w0)))
            if aucs:
                worst = min(worst, float(np.mean(aucs)))
        return worst if np.isfinite(worst) else float("nan")


def fold_delta_draws(y: np.ndarray, g: np.ndarray, probs_v: list[np.ndarray],
                     probs_erm: list[np.ndarray], n_boot: int = 1000,
                     seed: int = 0, progress: bool = False) -> tuple[float, np.ndarray]:
    """Point Δ_f and its n_boot paired bootstrap draws for one fold.
    Same resamples as the original index-based version (same RNG calls)."""
    mv = [_WeightedWorstGroup(y, p, g) for p in probs_v]
    me = [_WeightedWorstGroup(y, p, g) for p in probs_erm]
    n = len(y)

    def delta(w):
        return np.mean([f(w) for f in mv]) - np.mean([f(w) for f in me])

    point = delta(np.ones(n))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = delta(np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
        if progress and (b + 1) % 100 == 0:
            print(f"    {b + 1}/{n_boot}", flush=True)
    return float(point), draws


def pooled_verdict(fold_results: dict[str, tuple[float, np.ndarray]],
                   macro_drop: dict[str, float], max_drop: float = 0.005) -> dict:
    """Apply the pre-registered rule. fold_results: {fold: (Δ_f, draws)};
    macro_drop: {fold: mean macro-AUROC(ERM) − mean macro-AUROC(V)}."""
    folds = sorted(fold_results)
    points = np.array([fold_results[f][0] for f in folds])
    draws = np.mean([fold_results[f][1] for f in folds], axis=0)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    n_pos = int((points > 0).sum())
    pooled = float(points.mean())
    worst_drop = max(macro_drop.values()) if macro_drop else 0.0
    complete = len(folds) >= 3
    if not complete:
        verdict = "INCOMPLETE (need 3 folds)"
    elif pooled <= 0 or lo <= 0 <= hi:
        verdict = "REJECTED"
    elif n_pos < 2:
        verdict = "NOT SUPPORTED (Δ>0 in fewer than 2 folds)"
    elif worst_drop > max_drop:
        verdict = "NOT SUPPORTED (macro-AUROC drop > 0.5 pt)"
    else:
        verdict = "SUPPORTED"
    fold_ci = {f: [round(float(v), 4) for v in
                   np.nanpercentile(fold_results[f][1], [2.5, 97.5])] for f in folds}
    return {"folds": folds,
            "delta_per_fold": dict(zip(folds, points.round(4).tolist())),
            "ci95_per_fold": fold_ci,
            "pooled_delta": round(pooled, 4),
            "pooled_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "folds_with_positive_delta": n_pos,
            "max_macro_auroc_drop": round(worst_drop, 4),
            "verdict": verdict}
