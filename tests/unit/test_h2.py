"""Amendment 2 (H2) per-fold rule.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "s7", Path(__file__).resolve().parents[2] / "scripts/stage7_tables.py")
s7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s7)


def _m(g4, male, n_pos_g4=100):
    pg = {str(k): {"positive_coverage": male, "n_pos_cells": 200} for k in range(4)}
    pg["4"] = {"positive_coverage": g4 if n_pos_g4 >= 20 else None,
               "n_pos_cells": n_pos_g4}
    return {"test": {"conformal_label_conditional":
                     {"per_group": pg, "n_fallback_cells": 3}}}


def test_h2_fold_rule():
    assert s7.h2_fold([_m(0.78, 0.91)] * 3)["status"] == "CONFIRMED"
    assert s7.h2_fold([_m(0.92, 0.95)] * 3)["status"] == "NOT CONFIRMED"  # >= 0.90
    assert s7.h2_fold([_m(0.85, 0.84)] * 3)["status"] == "NOT CONFIRMED"  # not < male
    assert s7.h2_fold([_m(0.7, 0.9, n_pos_g4=5)])["status"].startswith("NOT EVALUABLE")
