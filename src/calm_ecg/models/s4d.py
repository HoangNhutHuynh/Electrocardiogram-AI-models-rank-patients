"""S4D backbone (~4M params at default config) for 12-lead ECG.

S4D-Lin diagonal state-space layer (Gu et al., 2022): kernel computed from
A = -1/2 + i*pi*n, ZOH discretization with learnable per-channel log(dt),
convolution via FFT. Compact and VRAM-friendly (fits pretraining at
batch 64, crop 2500, bf16 on an 8 GB card).

Encoder: conv stem (stride 5: 500 Hz -> 100 Hz latent) -> n_layers S4D
blocks (prenorm, GLU, residual) -> features (B, d_model, L/5).
Heads: MaskedReconHead for SSL pretraining, ClsHead for the 6 labels.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, n_state: int = 64,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        log_dt = (math.log(dt_min) + torch.rand(d_model)
                  * (math.log(dt_max) - math.log(dt_min)))
        self.log_dt = nn.Parameter(log_dt)
        c = torch.randn(d_model, n_state // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))
        self.register_buffer("log_A_real",
                             torch.log(0.5 * torch.ones(d_model, n_state // 2)))
        self.register_buffer("A_imag",
                             math.pi * torch.arange(n_state // 2).repeat(d_model, 1).float())

    def forward(self, L: int) -> torch.Tensor:
        # Real arithmetic only: complex exp/einsum was nondeterministic on a
        # Windows CPU build (torch 2.11, AMD Zen 4). Same math as
        # 2*Re(sum_n C2_n * exp(dtA_n * l)), C2 = C (exp(dtA) - 1) / A.
        dt = torch.exp(self.log_dt).unsqueeze(-1)                     # (d, 1)
        ar, ai = -torch.exp(self.log_A_real), self.A_imag             # A = ar + i ai
        cr, ci = self.C[..., 0], self.C[..., 1]                       # (d, n)
        a, b = ar * dt, ai * dt                                       # dtA = a + i b
        er, ei = torch.exp(a) * torch.cos(b) - 1.0, torch.exp(a) * torch.sin(b)
        nr, ni = cr * er - ci * ei, cr * ei + ci * er                 # C * (exp(dtA) - 1)
        den = ar * ar + ai * ai                                       # |A|^2 > 0 (ar < 0)
        c2r, c2i = (nr * ar + ni * ai) / den, (ni * ar - nr * ai) / den
        l = torch.arange(L, device=a.device, dtype=a.dtype)
        al, bl = a.unsqueeze(-1) * l, b.unsqueeze(-1) * l             # (d, n, L)
        K = torch.exp(al) * (c2r.unsqueeze(-1) * torch.cos(bl)
                             - c2i.unsqueeze(-1) * torch.sin(bl))
        return 2 * K.sum(1)                                           # (d, L)


class S4DLayer(nn.Module):
    """Prenorm S4D mixing + position-wise FFN, both residual."""

    def __init__(self, d_model: int, n_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.kernel = S4DKernel(d_model, n_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.GLU(dim=1), nn.Dropout(dropout))
        self.proj = nn.Conv1d(d_model, 2 * d_model, 1)
        self.ffn = nn.Sequential(nn.Conv1d(d_model, 4 * d_model, 1),
                                 nn.GELU(),
                                 nn.Conv1d(4 * d_model, d_model, 1),
                                 nn.Dropout(dropout))

    @staticmethod
    def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, d, L)
        h = self._ln(self.norm1, x)
        L = h.shape[-1]
        k = self.kernel(L)                                   # (d, L)
        k_f = torch.fft.rfft(k.float(), n=2 * L)             # fp32 FFT under AMP
        h_f = torch.fft.rfft(h.float(), n=2 * L)
        y = torch.fft.irfft(h_f * k_f, n=2 * L)[..., :L].to(h.dtype)
        y = y + h * self.D.unsqueeze(-1)
        x = x + self.out(self.proj(y))
        return x + self.ffn(self._ln(self.norm2, x))


class S4DEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 6,
                 n_state: int = 64, stem_stride: int = 5, dropout: float = 0.1):
        super().__init__()
        self.stem_stride = stem_stride
        self.stem = nn.Sequential(
            nn.Conv1d(12, d_model, kernel_size=stem_stride * 2 + 1,
                      stride=stem_stride, padding=stem_stride),
            nn.GELU())
        self.layers = nn.ModuleList(
            [S4DLayer(d_model, n_state, dropout) for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 12, L) -> (B, d, L')
        h = self.stem(x)
        for layer in self.layers:
            h = layer(h)
        return h


class MaskedReconModel(nn.Module):
    """SSL pretraining: contiguous time segments are zeroed at the input and
    marked by a learned token added after the stem; the head reconstructs the
    original 12-lead signal; loss = MSE on masked AND real (non-padding)
    positions only."""

    def __init__(self, encoder: S4DEncoder):
        super().__init__()
        self.encoder = encoder
        s = encoder.stem_stride
        self.mask_token = nn.Parameter(torch.zeros(encoder.d_model))
        self.head = nn.ConvTranspose1d(encoder.d_model, 12,
                                       kernel_size=2 * s, stride=s, padding=s // 2)

    @staticmethod
    def make_mask(B: int, L: int, mask_ratio: float, span: int,
                  device, generator=None) -> torch.Tensor:
        """(B, L) bool, True = masked; contiguous spans, ~mask_ratio total."""
        n_spans = max(1, int(L * mask_ratio / span))
        m = torch.zeros(B, L, dtype=torch.bool, device=device)
        starts = torch.randint(0, max(1, L - span), (B, n_spans),
                               generator=generator, device=device)
        for j in range(n_spans):
            ar = torch.arange(span, device=device)
            idx = (starts[:, j:j + 1] + ar).clamp(max=L - 1)
            m.scatter_(1, idx, True)
        return m

    def forward(self, x: torch.Tensor, pad: torch.Tensor,
                mask_ratio: float = 0.4, span: int = 100):
        B, _, L = x.shape
        m = self.make_mask(B, L, mask_ratio, span, x.device)
        m = m & pad                       # never "reconstruct" padding
        x_in = x * (~m).unsqueeze(1)
        h = self.encoder(x_in)
        # add mask token at downsampled positions that contain masked samples
        s = self.encoder.stem_stride
        m_ds = m[:, :h.shape[-1] * s].reshape(B, -1, s).any(-1)      # (B, L')
        h = h + self.mask_token.view(1, -1, 1) * m_ds.unsqueeze(1)
        rec = self.head(h)[..., :L]
        target_mask = m.unsqueeze(1)                                  # (B,1,L)
        loss = ((rec - x) ** 2 * target_mask).sum() / target_mask.sum().clamp(min=1)
        return loss, rec, m


class ClsModel(nn.Module):
    """Masked mean pool over real positions -> 6-label logits."""

    def __init__(self, encoder: S4DEncoder, n_labels: int = 6):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.d_model, n_labels)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                                           # (B,d,L')
        s = self.encoder.stem_stride
        p_ds = pad[:, :h.shape[-1] * s].reshape(x.shape[0], -1, s).any(-1)
        w = p_ds.unsqueeze(1).float()
        pooled = (h * w).sum(-1) / w.sum(-1).clamp(min=1)
        return self.fc(pooled)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
