"""Stage 5 — fairness-aware objectives: GroupDRO + fair supervised contrastive.

GroupDRO (Sagawa et al. 2020): exponential-weights ascent over per-group
losses; groups = sex x age-bin (8), id -1 = missing demographics (kept in
the plain BCE term, excluded from DRO re-weighting).

Fair-SupCon: multi-label supervised contrastive where positives share >=1
positive label; CROSS-group positive pairs get weight (1 + group_weight),
pulling same-diagnosis embeddings together across demographic groups.
With group_weight=0 it reduces exactly to multi-label SupCon (tested).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GroupDRO:
    def __init__(self, n_groups: int = 8, eta: float = 0.01,
                 device: str | torch.device = "cpu"):
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.eta = eta
        self.n_groups = n_groups

    def to(self, device) -> "GroupDRO":
        self.q = self.q.to(device)
        return self

    def __call__(self, per_sample_loss: torch.Tensor,
                 group: torch.Tensor) -> torch.Tensor:
        """per_sample_loss: (B,), group: (B,) long in [-1, n_groups).
        Returns the DRO loss over groups present in the batch; updates q."""
        losses, present = [], []
        for g in range(self.n_groups):
            sel = group == g
            if sel.any():
                losses.append(per_sample_loss[sel].mean())
                present.append(g)
        if not present:
            return per_sample_loss.mean()
        gl = torch.stack(losses)
        with torch.no_grad():  # exponential-weights update on the q simplex
            self.q[present] = self.q[present] * torch.exp(self.eta * gl.detach())
            self.q = self.q / self.q.sum()
        w = self.q[present] / self.q[present].sum()
        return (w * gl).sum()


def per_sample_masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, C) -> (B,): mean BCE over DEFINED labels of each sample;
    samples with no defined label contribute 0."""
    valid = ~torch.isnan(y)
    y0 = torch.nan_to_num(y, nan=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, y0, reduction="none")
    bce = bce * valid
    return bce.sum(1) / valid.sum(1).clamp(min=1)


def fair_supcon(z: torch.Tensor, y: torch.Tensor, group: torch.Tensor,
                temperature: float = 0.1, group_weight: float = 1.0
                ) -> torch.Tensor:
    """z: (B, D) embeddings; y: (B, C) labels with NaN; group: (B,) long.
    Positives: pairs sharing >=1 positive (==1) label. Cross-group positive
    pairs weighted (1 + group_weight); pairs with a missing group get 1."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    yp = (torch.nan_to_num(y, nan=0.0) == 1.0).float()
    pos = (yp @ yp.t() > 0) & ~eye                       # share >=1 positive
    both_valid = (group.unsqueeze(0) >= 0) & (group.unsqueeze(1) >= 0)
    cross = (group.unsqueeze(0) != group.unsqueeze(1)) & both_valid
    w = torch.ones_like(sim) + group_weight * (cross & pos).float()

    logits = sim.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    anchors = pos.any(dim=1)
    if not anchors.any():
        return z.sum() * 0.0
    wp = w * pos
    # 0 * (-inf) = NaN: zero out log_prob where the pair carries no weight
    lp = torch.where(wp > 0, log_prob, torch.zeros_like(log_prob))
    per_anchor = -(wp * lp).sum(1) / wp.sum(1).clamp(min=1e-8)
    return per_anchor[anchors].mean()
