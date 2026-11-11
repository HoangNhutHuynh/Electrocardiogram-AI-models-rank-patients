"""Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.eval.label_shift import adjust, em_prior  # noqa: E402


def _calibrated_scores(n, prior, pi_s, seed):
    """Gaussian class-conditionals (±2, sd 1) -> exact posterior under pi_s."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prior
    x = np.where(y, rng.normal(2, 1, n), rng.normal(-2, 1, n))
    p = 1 / (1 + np.exp(-(4 * x + np.log(pi_s / (1 - pi_s)))))
    return p, y


def test_em_recovers_shifted_prior():
    p, y = _calibrated_scores(50000, prior=0.35, pi_s=0.05, seed=0)
    est, _ = em_prior(p, pi_s=0.05)
    assert abs(est - 0.35) < 0.01, est


def test_em_no_shift_is_fixed_point():
    p, _ = _calibrated_scores(50000, prior=0.05, pi_s=0.05, seed=1)
    est, _ = em_prior(p, pi_s=0.05)
    assert abs(est - 0.05) < 0.005


def test_adjust_identity_and_monotone():
    p = np.linspace(0.01, 0.99, 50)
    assert np.allclose(adjust(p, 0.1, 0.1), p)
    assert np.all(np.diff(adjust(p, 0.05, 0.4)) > 0)  # ranks (AUROC) unchanged
