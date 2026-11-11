"""Unit tests for dataset utilities (no real data needed).
Run: pytest tests/unit/test_dataset.py -q

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.data.dataset import (EcgDataset, demographic_group,  # noqa: E402
                                   masked_bce_loss)


def test_demographic_groups():
    assert demographic_group("M", 30) == 0
    assert demographic_group("M", 45) == 1
    assert demographic_group("M", 65) == 2
    assert demographic_group("M", 80) == 3
    assert demographic_group("F", 30) == 4
    assert demographic_group("F", 80) == 7
    assert demographic_group("", 30) == -1
    assert demographic_group("M", float("nan")) == -1


def test_masked_bce_ignores_nan():
    """THE partial-label gate: NaN labels contribute nothing to the loss."""
    logits = torch.tensor([[0.5, -1.0, 2.0]])
    y_full = torch.tensor([[1.0, 0.0, 1.0]])
    y_nan = torch.tensor([[1.0, 0.0, float("nan")]])
    full = masked_bce_loss(logits, y_full)
    part = masked_bce_loss(logits[:, :2], y_full[:, :2])
    with_nan = masked_bce_loss(logits, y_nan)
    assert torch.isclose(with_nan, part), "NaN label leaked into the loss"
    assert not torch.isclose(with_nan, full)
    # all-NaN batch: zero loss, still differentiable, no crash
    y_all_nan = torch.full((1, 3), float("nan"))
    z = masked_bce_loss(logits.requires_grad_(True), y_all_nan)
    assert z.item() == 0.0
    z.backward()  # must not raise


def test_crop_and_zscore_semantics():
    ds = object.__new__(EcgDataset)  # no real files needed
    ds.crop_len, ds.train = 100, True
    ds.rng = np.random.default_rng(0)
    sig = np.zeros((12, 300), dtype=np.float32)
    sig[:, 50:250] = np.random.default_rng(1).standard_normal((12, 200)).astype(np.float32)
    mask = np.zeros(300, dtype=bool)
    mask[50:250] = True
    for _ in range(20):  # crop must stay inside the real-signal region
        out, m = ds._crop(sig, mask)
        assert out.shape == (12, 100) and m.shape == (100,)
        assert m.all(), "train crop escaped the padding mask"
    z = EcgDataset._zscore(out, m)
    assert np.allclose(z[:, m].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(z[:, m].std(axis=1), 1, atol=1e-3)


def test_crop_shorter_record_padded():
    ds = object.__new__(EcgDataset)
    ds.crop_len, ds.train = 100, False
    ds.rng = np.random.default_rng(0)
    sig = np.ones((12, 60), dtype=np.float32)
    mask = np.ones(60, dtype=bool)
    out, m = ds._crop(sig, mask)
    assert out.shape == (12, 100)
    assert m.sum() == 60 and not m[60:].any()


def test_zscore_dtype_stable_on_empty_mask():
    """Regression: fully-padded crop must stay float32 (double crashed AMP)."""
    sig = np.zeros((12, 100), dtype=np.float32)
    mask = np.zeros(100, dtype=bool)
    out = EcgDataset._zscore(sig, mask)
    assert out.dtype == np.float32
    out2 = EcgDataset._zscore(np.random.randn(12, 100).astype(np.float32),
                              np.ones(100, dtype=bool))
    assert out2.dtype == np.float32


def test_sanitize_nonfinite():
    """Regression: NaN/Inf samples are zeroed AND removed from the mask."""
    sig = np.ones((12, 100), dtype=np.float32)
    sig[3, 10] = np.nan
    sig[7, 90] = np.inf
    mask = np.ones(100, dtype=bool)
    out, m = EcgDataset._sanitize(sig, mask)
    assert np.isfinite(out).all()
    assert not m[10] and not m[90] and m.sum() == 98
    # clean input passes through untouched
    sig2 = np.ones((12, 100), dtype=np.float32)
    out2, m2 = EcgDataset._sanitize(sig2, np.ones(100, dtype=bool))
    assert m2.all() and (out2 == sig2).all()
