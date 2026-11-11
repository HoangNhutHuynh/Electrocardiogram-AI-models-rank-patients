"""Locate CPU forward-pass nondeterminism (seen on Windows / AMD Zen 4).

Each component is run REPS times on identical input; prints how many runs
differ from the first and where in the sequence the difference starts.
Run: python scripts/diag_determinism.py   and paste the output.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.models.s4d import S4DEncoder  # noqa: E402

REPS = 300
torch.manual_seed(0)
enc = S4DEncoder(d_model=32, n_layers=3, n_state=16, dropout=0.0).eval()
x = torch.randn(2, 12, 500)


def check(name, fn):
    with torch.no_grad():
        ref = fn().clone()
        bad, first_t = 0, None
        for _ in range(REPS):
            d = (fn() - ref).abs()
            if d.max() > 0:
                bad += 1
                t = int(torch.nonzero(d.amax(dim=tuple(range(d.dim() - 1))))[0])
                first_t = t if first_t is None else min(first_t, t)
    print(f"{name:34s} differing runs {bad:3d}/{REPS}  "
          f"first differing time index {first_t} (of {ref.shape[-1]})")


h_stem = enc.stem(x).detach()
for mk in (True, False):
    torch.backends.mkldnn.enabled = mk
    tag = "mkldnn ON " if mk else "mkldnn OFF"
    check(f"[{tag}] stem conv (k11 s5 p5)", lambda: enc.stem(x))
    check(f"[{tag}] stem conv, explicit F.pad", lambda: torch.nn.functional.gelu(
        torch.nn.functional.conv1d(torch.nn.functional.pad(x, (5, 5)),
                                   enc.stem[0].weight, enc.stem[0].bias, stride=5)))
    check(f"[{tag}] S4D layer 0 only", lambda: enc.layers[0](h_stem))
    check(f"[{tag}] full encoder", lambda: enc(x))
torch.backends.mkldnn.enabled = True
print("torch", torch.__version__)
