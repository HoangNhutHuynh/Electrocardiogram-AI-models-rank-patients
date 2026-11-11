"""Unit tests for stage 1 signal standardization. Run: pytest tests/unit -q

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.data.signals import (LEADS, padding_mask,  # noqa: E402
                                   quality_flags, reorder_leads,
                                   resample_to_target)


def _synthetic_ecg(fs: int, seconds: float = 10.0, seed: int = 0) -> np.ndarray:
    """Chirp + spike train standing in for QRS; deterministic."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    base = np.sin(2 * np.pi * (1 + 2 * t) * t)          # chirp 1->~21 Hz
    spikes = np.zeros_like(t)
    spikes[::fs] = 1.0                                   # 1 Hz "QRS"
    sig = 0.8 * base + spikes + 0.01 * rng.standard_normal(t.size)
    return np.tile(sig, (12, 1)).astype(np.float32)


def test_resample_roundtrip_nrmse_below_1pct():
    """Gate: 500 -> 400 -> 500 Hz round trip, NRMSE < 1% (edges trimmed —
    polyphase filters have transient at boundaries)."""
    from scipy.signal import resample_poly
    x = _synthetic_ecg(500)
    assert resample_to_target(x, 500) is x        # no-op at target fs
    x_400 = resample_poly(x, 4, 5, axis=-1)        # down to 400 Hz
    x_back = resample_to_target(x_400, 400)        # back up to 500 Hz
    n = min(x.shape[-1], x_back.shape[-1])
    a, b = x[..., 50:n - 50], x_back[..., 50:n - 50]
    nrmse = np.sqrt(np.mean((a - b) ** 2)) / (a.max() - a.min())
    assert nrmse < 0.01, f"NRMSE {nrmse:.4f} >= 1%"


def test_resample_output_length():
    x = _synthetic_ecg(400)
    y = resample_to_target(x, 400)
    assert y.shape == (12, x.shape[-1] * 5 // 4)
    assert not np.isnan(y).any()


def test_padding_mask_leading_trailing_only():
    sig = np.zeros((12, 100), dtype=np.float32)
    sig[:, 20:80] = 1.0
    sig[:, 50] = 0.0  # interior zero sample is NOT padding
    m = padding_mask(sig)
    assert not m[:20].any() and not m[80:].any()
    assert m[20:80].all()


def test_padding_mask_all_zero():
    assert not padding_mask(np.zeros((12, 100), dtype=np.float32)).any()


def test_reorder_leads_aliases_and_shapes():
    sig = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
    names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    out = reorder_leads(sig.T.copy(), names)      # (L, 12) input accepted
    assert out.shape == (12, 5)
    assert np.allclose(out, sig)                   # aliases map to same order
    out2 = reorder_leads(sig[::-1].copy(), names[::-1])  # scrambled order
    assert np.allclose(out2, sig)


def test_quality_flags():
    ok = _synthetic_ecg(500)
    assert quality_flags(ok) == {"has_nan": False, "flat": False}
    bad = ok.copy()
    bad[0, 0] = np.nan
    assert quality_flags(bad)["has_nan"]
    assert quality_flags(np.zeros((12, 100), dtype=np.float32))["flat"]


def test_resolve_code_path_layouts(tmp_path):
    from calm_ecg.data.signals import resolve_code_path
    flat = tmp_path / "flat"
    (flat).mkdir()
    (flat / "exams_part0.hdf5").touch()
    assert resolve_code_path(flat, "exams_part0.hdf5").name == "exams_part0.hdf5"
    nested = tmp_path / "nested"
    (nested / "exams_part1").mkdir(parents=True)
    (nested / "exams_part1" / "exams_part1.hdf5").touch()
    p = resolve_code_path(nested, "exams_part1.hdf5")
    assert p.parent.name == "exams_part1"
    import pytest
    with pytest.raises(FileNotFoundError):
        resolve_code_path(tmp_path, "exams_part9.hdf5")
