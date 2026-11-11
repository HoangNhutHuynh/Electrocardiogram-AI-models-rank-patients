"""LoRA for the S4D encoder's pointwise convolutions (stage 5).

Hybrid placement per the proposal: LoRA on the TOP-k S4D blocks' proj/ffn
1x1 convolutions; everything else frozen except LayerNorms and the task
head. Rationale: naive full-encoder LoRA underperformed partial fine-tuning
in prior ECG-FM work, so adaptation capacity is concentrated where features
are most task-specific.

y = base(x) + (alpha/r) * B(A(x)),  A: in->r (kaiming), B: r->out (zeros)
so the wrapped module is EXACTLY the base at init (tested).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, r: int = 8, alpha: int = 16):
        super().__init__()
        assert base.kernel_size == (1,), "LoRA here wraps pointwise convs only"
        self.base = base
        self.A = nn.Conv1d(base.in_channels, r, 1, bias=False)
        self.B = nn.Conv1d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))


def _wrap_pointwise(module: nn.Module, r: int, alpha: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv1d) and child.kernel_size == (1,):
            setattr(module, name, LoRAConv1d(child, r, alpha))
        else:
            _wrap_pointwise(child, r, alpha)


def apply_lora(encoder: nn.Module, top_blocks: int = 2,
               r: int = 8, alpha: int = 16) -> nn.Module:
    """Freeze the whole encoder, then unfreeze: LoRA adapters on the top
    `top_blocks` S4D layers + all LayerNorms. Returns the encoder."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    for layer in encoder.layers[-top_blocks:]:
        _wrap_pointwise(layer, r, alpha)
    for m in encoder.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    return encoder


def trainable_fraction(model: nn.Module) -> float:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = sum(p.numel() for p in model.parameters())
    return t / n
