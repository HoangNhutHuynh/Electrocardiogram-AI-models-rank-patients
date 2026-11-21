"""Dilated residual convolutional backbone, the architecture control for the
state-space encoder in `s4d.py`.

Same interface as S4DEncoder (d_model, stem_stride, layers, (B,12,L) ->
(B,d,L/5)), so MaskedReconModel, ClsModel, PooledClsModel and apply_lora
work on it unchanged. Same stem and the same parameter budget within 1%,
so a difference between the two backbones is not a difference in capacity.

Mixing is local and convolutional instead of a global state-space kernel:
6 blocks of a grouped k=9 convolution with dilations 1..32, giving a
receptive field of 505 latent steps, slightly wider than the 500-step
latent sequence of a 5-s crop.

Normalization is LayerNorm, not BatchNorm, so that adaptation trains the
same kinds of parameters as in the state-space encoder (apply_lora unfreezes
the LayerNorms) and no batch statistics estimated on the source mix enter the
comparison. (In eval mode BatchNorm would not update its statistics on the
held-out cohort; an earlier version of this note and PREREGISTRATION.md E4
said otherwise.)

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Prenorm dilated convolution + position-wise FFN, both residual."""

    def __init__(self, d_model: int, kernel_size: int = 9, dilation: int = 1,
                 groups: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=groups,
                              dilation=dilation,
                              padding=dilation * (kernel_size - 1) // 2)
        self.act = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.ffn = nn.Sequential(nn.Conv1d(d_model, 4 * d_model, 1),
                                 nn.GELU(),
                                 nn.Conv1d(4 * d_model, d_model, 1),
                                 nn.Dropout(dropout))

    @staticmethod
    def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, d, L)
        x = x + self.act(self.conv(self._ln(self.norm1, x)))
        return x + self.ffn(self._ln(self.norm2, x))


class ResNet1dEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 6,
                 kernel_size: int = 9, groups: int = 4, stem_stride: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.stem_stride = stem_stride
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(12, d_model, kernel_size=stem_stride * 2 + 1,
                      stride=stem_stride, padding=stem_stride),
            nn.GELU())
        self.layers = nn.ModuleList(
            [ResBlock(d_model, kernel_size, 2 ** i, groups, dropout)
             for i in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,12,L)->(B,d,L')
        h = self.stem(x)
        for layer in self.layers:
            h = layer(h)
        return h
