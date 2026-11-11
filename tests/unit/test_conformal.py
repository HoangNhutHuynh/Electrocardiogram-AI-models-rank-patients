"""Unit tests for Mondrian conformal + temperature scaling (stage 6 gates).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.eval.conformal import (MondrianConformal,  # noqa: E402
                                     fit_temperature)


def _synthetic(n, seed, miscal=1.0):
    """Exchangeable binary data: p true prob, y ~ Bernoulli(p); returned
    'probs' are miscalibrated by temperature `miscal` on the logit scale."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=(n, 3))
    y = (rng.random((n, 3)) < p).astype(float)
    logits = np.log(p / (1 - p)) * miscal
    probs = 1 / (1 + np.exp(-logits))
    g = rng.integers(0, 4, n)
    return probs, logits, y, g


def test_conformal_coverage_on_exchangeable_data():
    """Gate: empirical coverage converges to 1 - alpha (within sampling err)."""
    probs_cal, _, y_cal, g_cal = _synthetic(4000, seed=0)
    probs_te, _, y_te, g_te = _synthetic(4000, seed=1)
    conf = MondrianConformal(n_groups=4, alpha=0.1).calibrate(
        probs_cal, y_cal, g_cal)
    cov = conf.coverage(probs_te, y_te, g_te)
    assert 0.88 <= cov["marginal_coverage"] <= 0.93, cov["marginal_coverage"]
    for g, d in cov["per_group"].items():
        assert 0.86 <= d["coverage"] <= 0.95, (g, d)


def test_conformal_small_group_falls_back_to_marginal():
    probs, _, y, g = _synthetic(500, seed=2)
    g[:] = 0
    g[:10] = 1  # group 1 below min_n
    conf = MondrianConformal(n_groups=4, alpha=0.1, min_n=50).calibrate(
        probs, y, g)
    assert np.allclose(conf.q[1], conf.q_marginal)
    assert not np.allclose(conf.q[0], conf.q_marginal)


def test_nan_labels_excluded_from_calibration_and_coverage():
    probs, _, y, g = _synthetic(1000, seed=3)
    y[:, 2] = np.nan  # label undefined everywhere
    conf = MondrianConformal(n_groups=4, alpha=0.1).calibrate(probs, y, g)
    cov = conf.coverage(probs, y, g)
    assert np.isfinite(cov["marginal_coverage"])


def test_temperature_recovers_miscalibration():
    """Probs sharpened by T=3 on the logit scale -> fitted T ~ 3."""
    _, logits, y, _ = _synthetic(20000, seed=4, miscal=3.0)
    T = fit_temperature(logits, y)
    assert 2.4 < T < 3.7, T
    # already-calibrated logits -> T ~ 1
    _, logits1, y1, _ = _synthetic(20000, seed=5, miscal=1.0)
    assert 0.8 < fit_temperature(logits1, y1) < 1.25


def _imbalanced(n, seed, prev=0.04):
    """Rare positives, mostly easy negatives — the regime of the real ECG
    labels where the (g, c)-only conformal failed (Amendment 1)."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 2)) < prev).astype(float)
    logit = np.where(y == 1, rng.normal(1.5, 1.5, y.shape),
                     rng.normal(-4.0, 1.5, y.shape))
    p = 1 / (1 + np.exp(-logit))
    g = rng.integers(0, 2, n)
    return p, y, g


def test_label_conditional_guarantees_positive_coverage():
    from calm_ecg.eval.conformal import LabelConditionalMondrian
    p_cal, y_cal, g_cal = _imbalanced(20000, seed=10)
    p_te, y_te, g_te = _imbalanced(20000, seed=11)
    old = MondrianConformal(n_groups=2, alpha=0.1).calibrate(p_cal, y_cal, g_cal)
    new = LabelConditionalMondrian(n_groups=2, alpha=0.1).calibrate(
        p_cal, y_cal, g_cal)
    pos = y_te == 1
    sets_old = old.predict_sets(p_te, g_te)
    cov_old_pos = sets_old[..., 1][pos].mean()
    c_new = new.coverage(p_te, y_te, g_te)
    assert c_new["positive_coverage"] >= 0.87, c_new["positive_coverage"]
    assert c_new["negative_coverage"] >= 0.87
    assert cov_old_pos < c_new["positive_coverage"], \
        "old (g,c) conformal should under-cover rare positives"


def test_label_conditional_fallback_small_cells():
    from calm_ecg.eval.conformal import LabelConditionalMondrian
    p, y, g = _imbalanced(400, seed=12, prev=0.02)  # ~8 positives per label
    m = LabelConditionalMondrian(n_groups=2, alpha=0.1, min_n=30).calibrate(p, y, g)
    assert m.n_fallback > 0  # positive cells fell back to pooled thresholds
    assert np.isfinite(m.q).all()


def test_temperature_varies_continuously():
    """Regression: grid search gave identical T for slightly different models."""
    _, logits, y, _ = _synthetic(20000, seed=6, miscal=2.0)
    t1 = fit_temperature(logits, y)
    t2 = fit_temperature(logits * 1.01, y)
    assert t1 != t2 and abs(t2 / t1 - 1.01) < 0.02
