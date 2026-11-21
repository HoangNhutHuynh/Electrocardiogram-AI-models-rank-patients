"""Encoder factory.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import torch.nn as nn

from calm_ecg.models.resnet import ResNet1dEncoder
from calm_ecg.models.s4d import S4DEncoder


def build_encoder(cfg: dict, dropout: float) -> nn.Module:
    """Encoder named by cfg['arch'].

    Defaults to 's4d': the checkpoints written before the architecture
    control was added carry no 'arch' key, and must keep loading.
    """
    arch = cfg.get("arch", "s4d")
    if arch == "s4d":
        return S4DEncoder(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                          n_state=cfg["n_state"], dropout=dropout)
    if arch == "resnet":
        return ResNet1dEncoder(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                               kernel_size=cfg.get("kernel_size", 9),
                               groups=cfg.get("conv_groups", 4), dropout=dropout)
    raise ValueError(f"unknown arch {arch!r} (expected 's4d' or 'resnet')")
