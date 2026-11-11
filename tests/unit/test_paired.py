"""Tests for the pre-registered paired bootstrap (H1).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.eval.metrics import worst_group_auroc  # noqa: E402
from calm_ecg.eval.paired import (fast_auroc, fold_delta_draws,  # noqa: E402
                                  pooled_verdict, worst_group_auroc_fast)


def test_fast_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(float)
    s = np.round(rng.random(500), 2)  # many ties
    assert abs(fast_auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_worst_group_fast_matches_reference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (600, 3)).astype(float)
    y[:100, 2] = np.nan
    p = np.clip(y * 0.6 + rng.random((600, 3)) * 0.5, 0, 1)
    p = np.nan_to_num(p, nan=0.5)
    g = rng.integers(0, 4, 600)
    assert abs(worst_group_auroc_fast(y, p, g)
               - worst_group_auroc(y, p, g)) < 1e-9


def test_delta_sign_and_ci():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, (n, 2)).astype(float)
    g = rng.integers(0, 2, n)
    noise = lambda k: rng.random((n, 2)) * k  # noqa: E731
    good = [np.clip(y * 0.5 + noise(0.6), 0, 1) for _ in range(3)]
    bad = [np.clip(y * 0.2 + noise(0.9), 0, 1) for _ in range(3)]
    point, draws = fold_delta_draws(y, g, good, bad, n_boot=100)
    assert point > 0 and np.percentile(draws, 2.5) > 0


def test_pooled_verdict_rules():
    d = lambda m: (m, np.full(100, m) + np.linspace(-0.001, 0.001, 100))  # noqa: E731
    v = pooled_verdict({"a": d(0.01)}, {"a": 0.0})
    assert v["verdict"].startswith("INCOMPLETE")
    v = pooled_verdict({"a": d(0.01), "b": d(0.01), "c": d(-0.002)},
                       {"a": 0.0, "b": 0.001, "c": 0.0})
    assert v["verdict"] == "SUPPORTED"
    v = pooled_verdict({"a": d(0.01), "b": d(0.01), "c": d(0.01)},
                       {"a": 0.0, "b": 0.01, "c": 0.0})
    assert "macro" in v["verdict"]
    v = pooled_verdict({"a": d(-0.01), "b": d(-0.01), "c": d(0.001)},
                       {"a": 0.0, "b": 0.0, "c": 0.0})
    assert v["verdict"] == "REJECTED"


def _reference_draws(y, g, pv, pe, n_boot, seed=0):
    """Original index-based bootstrap (pre-optimisation) for equivalence."""
    def delta(idx):
        return (np.mean([worst_group_auroc_fast(y, p, g, idx) for p in pv])
                - np.mean([worst_group_auroc_fast(y, p, g, idx) for p in pe]))
    rng = np.random.default_rng(seed)
    n = len(y)
    return delta(None), np.array([delta(rng.integers(0, n, n)) for _ in range(n_boot)])


def test_weighted_bootstrap_equals_index_bootstrap():
    rng = np.random.default_rng(5)
    n = 900
    y = rng.integers(0, 2, (n, 3)).astype(float)
    y[:150, 1] = np.nan
    g = rng.integers(-1, 4, n)                      # includes missing group -1
    mk = lambda k: np.round(np.clip(y * k + rng.random((n, 3)), 0, 2), 2)  # noqa: E731  ties
    pv, pe = [np.nan_to_num(mk(0.6)) for _ in range(2)], [np.nan_to_num(mk(0.3)) for _ in range(2)]
    p_new, d_new = fold_delta_draws(y, g, pv, pe, n_boot=40)
    p_ref, d_ref = _reference_draws(y, g, pv, pe, n_boot=40)
    assert abs(p_new - p_ref) < 1e-12
    assert np.allclose(d_new, d_ref, atol=1e-12, equal_nan=True)
