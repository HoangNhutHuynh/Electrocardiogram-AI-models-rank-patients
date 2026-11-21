"""Tests for the ResNet-1D architecture control.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from calm_ecg.models import build_encoder  # noqa: E402
from calm_ecg.models.lora import apply_lora, trainable_fraction  # noqa: E402
from calm_ecg.models.resnet import ResNet1dEncoder  # noqa: E402
from calm_ecg.models.s4d import (ClsModel, MaskedReconModel,  # noqa: E402
                                 S4DEncoder, count_params)

TINY = dict(d_model=32, n_layers=3, n_state=16)


def test_shape_matches_s4d():
    x = torch.randn(2, 12, 2500)
    pad = torch.ones(2, 2500, dtype=torch.bool)
    res = ResNet1dEncoder(**{k: v for k, v in TINY.items() if k != "n_state"},
                          dropout=0.0).eval()
    s4d = S4DEncoder(**TINY, dropout=0.0).eval()
    assert res(x).shape == s4d(x).shape == (2, 32, 500)
    assert ClsModel(res)(x, pad).shape == (2, 6)


def test_parameter_budget_within_1_percent_of_s4d():
    """A difference between the backbones must not be a difference in size."""
    res = count_params(ResNet1dEncoder())
    s4d = count_params(S4DEncoder())
    assert abs(res - s4d) / s4d < 0.01, (res, s4d)


def test_receptive_field_covers_the_crop():
    """Dilations 1..2^(n-1) with kernel k reach past the 500-step latent."""
    enc = ResNet1dEncoder()
    k = enc.layers[0].conv.kernel_size[0]
    rf = 1 + sum((k - 1) * b.conv.dilation[0] for b in enc.layers)
    assert rf >= 500, rf


def test_lora_is_identity_at_init_and_trains_a_small_fraction():
    enc = ResNet1dEncoder(d_model=32, n_layers=3, dropout=0.0).eval()
    x = torch.randn(2, 12, 500)
    with torch.no_grad():
        before = enc(x)
    apply_lora(enc, top_blocks=2, r=4, alpha=8)
    enc.eval()
    with torch.no_grad():
        after = enc(x)
    assert torch.allclose(before, after, atol=1e-5)


def test_lora_trains_about_the_same_fraction_as_s4d():
    """At the reported size: 1.18% here against 1.47% for the state-space
    encoder. The gap is the proj 1x1 convolution, which only S4D blocks have."""
    enc = ResNet1dEncoder(dropout=0.0)
    apply_lora(enc, top_blocks=2, r=8, alpha=16)
    assert 0.010 < trainable_fraction(ClsModel(enc)) < 0.015


def test_masked_recon_runs():
    model = MaskedReconModel(ResNet1dEncoder(d_model=32, n_layers=2, dropout=0.0))
    x = torch.randn(2, 12, 500)
    pad = torch.ones(2, 500, dtype=torch.bool)
    loss, rec, m = model(x, pad, mask_ratio=0.4, span=100)
    assert rec.shape == x.shape and torch.isfinite(loss) and m.any()


def test_build_encoder_defaults_to_s4d_for_old_checkpoints():
    """Configs written before this ablation have no 'arch' key."""
    assert isinstance(build_encoder(dict(TINY), 0.0), S4DEncoder)
    assert isinstance(build_encoder({**TINY, "arch": "resnet"}, 0.0),
                      ResNet1dEncoder)
    with pytest.raises(ValueError):
        build_encoder({**TINY, "arch": "lstm"}, 0.0)
