"""Amendment 3 helpers: model rebuild from checkpoint, reader-CSV parsing.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from calm_ecg.models.lora import apply_lora  # noqa: E402
from calm_ecg.models.s4d import S4DEncoder  # noqa: E402
from calm_ecg.train.adapt import PooledClsModel, load_adapted_model  # noqa: E402

spec = importlib.util.spec_from_file_location("ect", REPO / "scripts/eval_code_test.py")
ect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ect)


def test_load_adapted_model_roundtrip(tmp_path):
    torch.manual_seed(0)
    pre_cfg = {"d_model": 32, "n_layers": 3, "n_state": 16}
    torch.save({"config": pre_cfg}, tmp_path / "pre.pt")
    enc = S4DEncoder(**pre_cfg, dropout=0.0)
    apply_lora(enc, 2, 4, 8)
    m = PooledClsModel(enc).eval()
    for p in m.parameters():  # make LoRA B non-zero so the test is meaningful
        p.data += 0.01 * torch.randn_like(p)
    cfg = {"pretrain_ckpt": str(tmp_path / "pre.pt"), "lora_top_blocks": 2,
           "lora_r": 4, "lora_alpha": 8}
    m2 = load_adapted_model({"config": cfg, "model": m.state_dict()})
    x, pad = torch.randn(2, 12, 500), torch.ones(2, 500, dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(m(x, pad), m2(x, pad), atol=1e-6)


def test_read_annot_order_and_index_column(tmp_path):
    hdr = "1dAVb,RBBB,LBBB,SB,AF,ST"
    (tmp_path / "a.csv").write_text(f"{hdr}\n1,0,0,0,0,1\n", encoding="utf-8")
    (tmp_path / "dnn.csv").write_text(f",{hdr}\n0,1,0,0,0,0,1\n", encoding="utf-8")
    # LABELS order: AF, 1dAVb, RBBB, LBBB, SB, STach
    assert ect.read_annot(tmp_path / "a.csv").tolist() == [[0, 1, 0, 0, 0, 1]]
    assert ect.read_annot(tmp_path / "dnn.csv").tolist() == [[0, 1, 0, 0, 0, 1]]


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = ect.best_f1_threshold(y, p)
    assert 0.3 < t <= 0.8
