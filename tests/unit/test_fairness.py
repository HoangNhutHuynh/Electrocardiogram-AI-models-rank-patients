"""Unit tests for stage 5: LoRA, GroupDRO, fair-SupCon, metrics.
Run: pytest tests/unit/test_fairness.py -q

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.eval.metrics import (expected_calibration_error,  # noqa: E402
                                   masked_macro_auroc, worst_group_auroc)
from calm_ecg.models.lora import apply_lora, trainable_fraction  # noqa: E402
from calm_ecg.models.s4d import S4DEncoder  # noqa: E402
from calm_ecg.train.fairness import (GroupDRO, fair_supcon,  # noqa: E402
                                     per_sample_masked_bce)

torch.manual_seed(0)


# ------------------------------------------------------------------- LoRA ---
def test_lora_identity_at_init_and_trainable_set():
    # Self-seeded and repeated: an unreproduced flake (diff ~0.1) was seen
    # once on Windows CPU; on failure, report which layer diverged.
    for trial in range(5):
        torch.manual_seed(100 + trial)
        enc = S4DEncoder(d_model=32, n_layers=3, n_state=16, dropout=0.0)
        x = torch.randn(2, 12, 500)
        enc.eval()
        acts_b, acts_a = [], []
        hs = [l.register_forward_hook(lambda m, i, o: acts_b.append(o.clone()))
              for l in enc.layers]
        with torch.no_grad():
            before = enc(x)
        for h in hs:
            h.remove()
        apply_lora(enc, top_blocks=2, r=4, alpha=8)
        enc.eval()
        hs = [l.register_forward_hook(lambda m, i, o: acts_a.append(o.clone()))
              for l in enc.layers]
        with torch.no_grad():
            after = enc(x)
        for h in hs:
            h.remove()
        per_layer = [(a - b).abs().max().item() for a, b in zip(acts_a, acts_b)]
        assert torch.allclose(before, after, atol=1e-6), \
            f"LoRA must start as identity (trial {trial}, per-layer max diff {per_layer})"
    frac = trainable_fraction(enc)
    assert 0 < frac < 0.10, f"trainable fraction {frac:.3f} outside (0, 10%)"
    # base conv weights frozen; A/B trainable
    grads_ok = any("lora" in n.lower() or ".A." in n or ".B." in n
                   for n, p in enc.named_parameters() if p.requires_grad)
    assert grads_ok


# --------------------------------------------------------------- GroupDRO ---
def test_groupdro_upweights_worse_group():
    dro = GroupDRO(n_groups=2, eta=0.5)
    # group 1 consistently worse
    for _ in range(20):
        loss = torch.tensor([0.1, 0.1, 1.0, 1.0])
        g = torch.tensor([0, 0, 1, 1])
        _ = dro(loss, g)
    assert dro.q[1] > dro.q[0], "DRO must upweight the worse group"
    # weighted loss exceeds plain mean (worst-case focus)
    loss = torch.tensor([0.1, 0.1, 1.0, 1.0])
    assert dro(loss, torch.tensor([0, 0, 1, 1])) > loss.mean()


def test_groupdro_handles_missing_groups():
    dro = GroupDRO(n_groups=8, eta=0.1)
    loss = torch.tensor([0.5, 0.7])
    out = dro(loss, torch.tensor([-1, -1]))  # no valid group -> plain mean
    assert torch.isclose(out, loss.mean())


def test_per_sample_masked_bce_nan():
    logits = torch.zeros(2, 3)
    y = torch.tensor([[1.0, float("nan"), 0.0],
                      [float("nan")] * 3])
    ps = per_sample_masked_bce(logits, y)
    assert torch.isclose(ps[0], torch.tensor(float(np.log(2))), atol=1e-5)
    assert ps[1].item() == 0.0  # all-undefined sample contributes nothing


# ------------------------------------------------------------- fair-SupCon --
def test_fair_supcon_degenerates_to_supcon():
    z = torch.randn(8, 16)
    y = (torch.rand(8, 6) > 0.6).float()
    g = torch.randint(0, 8, (8,))
    a = fair_supcon(z, y, g, group_weight=0.0)
    b = fair_supcon(z, y, torch.full((8,), -1), group_weight=5.0)
    assert torch.isclose(a, b, atol=1e-6), \
        "group_weight=0 and all-missing groups must match plain SupCon"


def test_fair_supcon_no_positives_safe():
    z = torch.randn(4, 16, requires_grad=True)
    y = torch.zeros(4, 6)  # no positive labels -> no anchors
    out = fair_supcon(z, y, torch.zeros(4, dtype=torch.long))
    assert out.item() == 0.0
    out.backward()  # differentiable no-op


# ---------------------------------------------------------------- metrics ---
def test_masked_macro_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 4)).astype(float)
    s = rng.random((200, 4))
    ours = masked_macro_auroc(y, s)
    ref = np.mean([roc_auc_score(y[:, c], s[:, c]) for c in range(4)])
    assert abs(ours - ref) < 1e-9
    # NaN rows for one label are excluded, not treated as negatives
    y2 = y.copy()
    y2[:100, 0] = np.nan
    ref0 = roc_auc_score(y2[100:, 0], s[100:, 0])
    ours2 = masked_macro_auroc(y2[:, :1], s[:, :1])
    assert abs(ours2 - ref0) < 1e-9


def test_worst_group_and_ece():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, (n, 2)).astype(float)
    g = np.array([0] * 200 + [1] * 200)
    s = y * 0.8 + 0.1  # near-perfect scores
    s[200:] = rng.random((200, 2))  # group 1 random -> worse
    wg = worst_group_auroc(y, s, g)
    assert wg < 0.65
    ece = expected_calibration_error(y, s)
    assert 0.0 <= ece <= 1.0


def test_fair_supcon_finite():
    """Regression: diagonal -inf * zero weight must not produce NaN."""
    for seed in range(5):
        torch.manual_seed(seed)
        z = torch.randn(8, 16)
        y = (torch.rand(8, 6) > 0.6).float()
        g = torch.randint(0, 8, (8,))
        out = fair_supcon(z, y, g, group_weight=1.0)
        assert torch.isfinite(out), f"NaN at seed {seed}"
