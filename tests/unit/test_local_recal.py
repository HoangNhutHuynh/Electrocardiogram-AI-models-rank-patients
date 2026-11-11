"""Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "elr", Path(__file__).resolve().parents[2] / "scripts/explore_local_recalibration.py")
elr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elr)


def test_local_thresholds_fallback_and_update():
    y = np.array([[1, 0], [1, 0], [1, 1], [0, 0], [0, 0]], float)  # label 1: 1 positive
    p = np.array([[.9, .2], [.8, .1], [.7, .9], [.2, .3], [.1, .4]])
    tau_src = np.array([.5, .5])
    tau = elr.local_thresholds(y, p, tau_src, min_pos=3)
    assert tau[1] == 0.5                 # < 3 positives -> keep source threshold
    assert 0.2 < tau[0] <= 0.7           # re-selected between negatives and positives
