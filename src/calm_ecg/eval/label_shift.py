"""Prior (label) shift adaptation by EM — Saerens, Latinne & Decaestecker,
Neural Computation 2002. EXPLORATORY analysis (not pre-registered).

Per label independently (multilabel = 6 binary problems). Assumes calibrated
source posteriors and p(x|y) unchanged between cohorts; only p(y) moves.
Uses UNLABELED target predictions only; true labels are used for scoring.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import numpy as np


def adjust(p: np.ndarray, pi_s: float, pi_t: float) -> np.ndarray:
    """Bayes-rule reweighting of a source posterior to a new prior."""
    a = p * (pi_t / pi_s)
    b = (1.0 - p) * ((1.0 - pi_t) / (1.0 - pi_s))
    return a / (a + b)


def em_prior(p: np.ndarray, pi_s: float, tol: float = 1e-6,
             max_iter: int = 1000) -> tuple[float, int]:
    """Estimate the target prior of one binary label. Returns (pi_t, iters)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pi_t = pi_s
    for it in range(1, max_iter + 1):
        new = float(adjust(p, pi_s, pi_t).mean())
        if abs(new - pi_t) < tol:
            return new, it
        pi_t = new
    return pi_t, max_iter


def adapt_multilabel(p: np.ndarray, pi_s: np.ndarray):
    """(N, C) probs + (C,) source priors -> adjusted probs, est. priors, iters."""
    out = np.empty_like(p)
    est, iters = np.empty(p.shape[1]), []
    for c in range(p.shape[1]):
        est[c], k = em_prior(p[:, c], float(pi_s[c]))
        out[:, c] = adjust(np.clip(p[:, c], 1e-7, 1 - 1e-7), float(pi_s[c]), est[c])
        iters.append(k)
    return out, est, iters
