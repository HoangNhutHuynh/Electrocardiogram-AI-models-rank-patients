"""Diagnose why LoRA is not an exact identity at init on this machine.

Expected: every 'max|after-before|' below is 0 (B is zero-initialised).
Run: python scripts/diag_lora.py   and paste the full output.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.models.lora import LoRAConv1d, apply_lora  # noqa: E402
from calm_ecg.models.s4d import S4DEncoder  # noqa: E402

print("python", platform.python_version(), "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| cpu", platform.processor())
print("mkldnn available", torch.backends.mkldnn.is_available(),
      "| threads", torch.get_num_threads())


def run(device="cpu", mkldnn=True, threads=None, tag=""):
    if threads:
        torch.set_num_threads(threads)
    torch.backends.mkldnn.enabled = mkldnn
    torch.manual_seed(0)
    enc = S4DEncoder(d_model=32, n_layers=3, n_state=16, dropout=0.0).to(device).eval()
    x = torch.randn(2, 12, 500, device=device)
    with torch.no_grad():
        b1 = enc(x)
        b2 = enc(x)
    # per-layer capture before
    acts_b = []
    hooks = [l.register_forward_hook(lambda m, i, o: acts_b.append(o.clone()))
             for l in enc.layers]
    with torch.no_grad():
        enc(x)
    for h in hooks:
        h.remove()
    apply_lora(enc, top_blocks=2, r=4, alpha=8)
    enc.to(device).eval()
    b_max = max(m.B.weight.abs().max().item()
                for m in enc.modules() if isinstance(m, LoRAConv1d))
    acts_a = []
    hooks = [l.register_forward_hook(lambda m, i, o: acts_a.append(o.clone()))
             for l in enc.layers]
    with torch.no_grad():
        a = enc(x)
    for h in hooks:
        h.remove()
    per_layer = [round((u - v).abs().max().item(), 6) for u, v in zip(acts_a, acts_b)]
    print(f"[{tag:22s}] repeat-diff {(b1 - b2).abs().max().item():.2e} | "
          f"max|B| {b_max:.1e} | max|after-before| {(a - b1).abs().max().item():.2e} "
          f"| per-layer {per_layer}")
    torch.backends.mkldnn.enabled = True


run(tag="cpu default")
run(mkldnn=False, tag="cpu mkldnn OFF")
run(threads=1, tag="cpu 1 thread")
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run(device="cuda", tag="cuda fp32 (no tf32)")
