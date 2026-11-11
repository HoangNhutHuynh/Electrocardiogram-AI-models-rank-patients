"""Second pass: isolate the nondeterministic op inside one S4D layer.
Run: python scripts/diag_determinism2.py   and paste the output.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.models.s4d import S4DEncoder  # noqa: E402

REPS = 300
torch.manual_seed(0)
enc = S4DEncoder(d_model=32, n_layers=3, n_state=16, dropout=0.0).eval()
layer = enc.layers[0]
with torch.no_grad():
    h = layer._ln(layer.norm1, enc.stem(torch.randn(2, 12, 500)))
L = h.shape[-1]


def check(name, fn):
    with torch.no_grad():
        ref = fn().clone()
        bad = sum(int((fn() != ref).any()) for _ in range(REPS))
    print(f"{name:40s} differing runs {bad:3d}/{REPS}")


with torch.no_grad():
    k = layer.kernel(L)
    kf = torch.fft.rfft(k.float(), n=2 * L)
    hf = torch.fft.rfft(h.float(), n=2 * L)

check("kernel(L)  [complex exp + einsum]", lambda: layer.kernel(L))
check("rfft(h, n=2L)", lambda: torch.fft.rfft(h.float(), n=2 * L))
check("irfft(hf*kf, n=2L)", lambda: torch.fft.irfft(hf * kf, n=2 * L))
check("proj 1x1 conv", lambda: layer.proj(h))
check("ffn (1x1 convs + GELU)", lambda: layer.ffn(h))
check("LayerNorm", lambda: layer._ln(layer.norm1, h))
# candidate fix: causal conv computed directly (no FFT)
check("direct causal conv (no FFT)", lambda: F.conv1d(
    F.pad(h, (L - 1, 0)), k.flip(-1).unsqueeze(1), groups=h.shape[1]))
with torch.no_grad():
    y_fft = torch.fft.irfft(hf * kf, n=2 * L)[..., :L]
    y_dir = F.conv1d(F.pad(h, (L - 1, 0)), k.flip(-1).unsqueeze(1), groups=h.shape[1])
print(f"max |FFT - direct| = {(y_fft - y_dir).abs().max().item():.2e}  (should be ~1e-6)")
print("threads", torch.get_num_threads(), "| torch", torch.__version__)
torch.set_num_threads(1)
check("irfft with 1 thread", lambda: torch.fft.irfft(hf * kf, n=2 * L))
check("rfft with 1 thread", lambda: torch.fft.rfft(h.float(), n=2 * L))
