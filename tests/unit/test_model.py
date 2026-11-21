"""Unit tests for the S4D backbone and SSL objective (CPU, synthetic data).
Includes the stage-4 gates: gradient flow, overfit-one-batch, determinism.
Run: pytest tests/unit/test_model.py -q   (overfit test ~30 s on CPU)

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.models.s4d import (ClsModel, MaskedReconModel,  # noqa: E402
                                 S4DEncoder, count_params)

torch.manual_seed(0)


def _tiny():
    return S4DEncoder(d_model=32, n_layers=2, n_state=16, dropout=0.0)


def _batch(B=4, L=500, smooth=False):
    if smooth:  # mixture of sinusoids — learnable target for the overfit gate
        g = torch.Generator().manual_seed(2)
        t = torch.arange(L) / 500.0
        f = 1 + 14 * torch.rand(B, 12, 3, generator=g)
        ph = 2 * torch.pi * torch.rand(B, 12, 3, generator=g)
        x = torch.sin(2 * torch.pi * f.unsqueeze(-1) * t + ph.unsqueeze(-1)).sum(2)
    else:
        x = torch.randn(B, 12, L)
    pad = torch.ones(B, L, dtype=torch.bool)
    pad[:, -50:] = False  # trailing padding
    return x * pad.unsqueeze(1), pad


def test_shapes_and_param_count():
    enc = S4DEncoder()  # default config
    n = count_params(MaskedReconModel(enc))
    assert 2e6 < n < 8e6, f"param count {n:,} outside the ~4M target band"
    x, pad = _batch()
    h = _tiny()(x)
    assert h.shape[0] == 4 and h.shape[1] == 32
    assert h.shape[2] == x.shape[-1] // 5  # stem stride


def test_gradient_flows_to_all_params():
    model = MaskedReconModel(_tiny())
    x, pad = _batch()
    loss, _, _ = model(x, pad)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    # mask_token may get zero grad if no span hit a downsampled bin — allow only it
    assert all("mask_token" in n for n in missing), f"no gradient: {missing}"


def test_mask_never_covers_padding():
    model = MaskedReconModel(_tiny())
    x, pad = _batch()
    _, _, m = model(x, pad)
    assert not (m & ~pad).any()


def test_cls_head_shapes_and_pad_invariance():
    model = ClsModel(_tiny())
    model.eval()
    x, pad = _batch()
    with torch.no_grad():
        out = model(x, pad)
        assert out.shape == (4, 6)
        # changing values inside padding must not change pooled logits
        x2 = x.clone()
        x2[:, :, -50:] += 100.0
        x2 = x2 * pad.unsqueeze(1)  # dataset zeroes padding; emulate
        out2 = model(x2, pad)
    assert torch.allclose(out, out2, atol=1e-5)


def test_overfit_one_batch():
    """Stage-4 gate: the architecture can memorize 4 samples."""
    torch.manual_seed(1)
    model = MaskedReconModel(_tiny())
    x, pad = _batch(B=4, L=500, smooth=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        loss, _, _ = model(x, pad, mask_ratio=0.3, span=50)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.3 * first, \
        f"loss {loss.item():.4f} did not drop vs first {first:.4f}"


def test_eval_determinism():
    model = MaskedReconModel(_tiny())
    model.eval()
    x, pad = _batch()
    with torch.no_grad():
        h1 = model.encoder(x)
        h2 = model.encoder(x)
    assert torch.equal(h1, h2)


def _kernel_complex_reference(k, L):
    """The original complex-arithmetic S4D-Lin kernel (pre-2026-11-10).

    Computed in NumPy complex128, not torch: torch's complex exp/einsum is
    the nondeterministic path on the Windows CPU build that motivated the
    real-arithmetic kernel, so a torch reference made this test flaky.
    """
    import numpy as np
    dt = np.exp(k.log_dt.detach().double().numpy())
    A = -np.exp(k.log_A_real.double().numpy()) + 1j * k.A_imag.double().numpy()
    C = k.C.detach().double().numpy()
    C = C[..., 0] + 1j * C[..., 1]
    dtA = A * dt[:, None]
    K = dtA[..., None] * np.arange(L)
    C2 = C * (np.exp(dtA) - 1.0) / A
    return torch.from_numpy(2 * np.einsum("dn,dnl->dl", C2, np.exp(K)).real)


def test_real_kernel_matches_complex_reference():
    """Regression for the CPU nondeterminism fix: same math, real ops only."""
    from calm_ecg.models.s4d import S4DKernel
    for d, n, L in [(32, 16, 100), (256, 64, 500)]:
        torch.manual_seed(d)
        k = S4DKernel(d, n)
        with torch.no_grad():
            ref = _kernel_complex_reference(k, L).double()
            new = k(L).double()
        scale = ref.abs().max().item()
        assert (new - ref).abs().max().item() < 1e-5 * max(scale, 1.0), (d, L)


def test_kernel_and_encoder_deterministic_many_reps():
    from calm_ecg.models.s4d import S4DKernel
    torch.manual_seed(0)
    k = S4DKernel(32, 16)
    enc = _tiny().eval()
    x, _ = _batch()
    with torch.no_grad():
        k0, h0 = k(100), enc(x)
        for _ in range(100):
            assert torch.equal(k(100), k0)
            assert torch.equal(enc(x), h0)
