"""Unit tests for stage 3 LODO splits. Run: pytest tests/unit -q

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.data.splits import (assert_no_leakage,  # noqa: E402
                                  make_lodo_splits)


def _synthetic_labels(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, n_pat in [("a", 50), ("b", 40), ("c", 30)]:
        for p in range(n_pat):
            for r in range(rng.integers(1, 4)):  # 1–3 records per patient
                rows.append({"record_id": f"{cohort}_{p}_{r}",
                             "patient_id": f"{cohort}_pat{p}",
                             "cohort": cohort})
    return pd.DataFrame(rows)


def test_lodo_structure_and_no_leakage():
    labels = _synthetic_labels()
    splits = make_lodo_splits(labels, ["a", "b", "c"], 0.1, seed=7)
    assert set(splits["folds"]) == {"a", "b", "c"}
    assert_no_leakage(splits, labels)  # must not raise
    # held-out cohort fully in test
    n_a = (labels["cohort"] == "a").sum()
    assert len(splits["folds"]["a"]["test"]) == n_a


def test_deterministic_given_seed():
    labels = _synthetic_labels()
    s1 = make_lodo_splits(labels, ["a", "b"], 0.1, seed=7)
    s2 = make_lodo_splits(labels, ["a", "b"], 0.1, seed=7)
    assert s1 == s2
    s3 = make_lodo_splits(labels, ["a", "b"], 0.1, seed=8)
    assert s3 != s1


def test_leakage_is_caught():
    labels = _synthetic_labels()
    splits = make_lodo_splits(labels, ["a", "b", "c"], 0.1, seed=7)
    # inject a patient-level leak: move one val record's sibling into train
    fold = splits["folds"]["a"]
    leaked = fold["val"][0]
    fold["train"].append(leaked.rsplit("_", 1)[0] + "_sib")
    labels = pd.concat([labels, pd.DataFrame([{
        "record_id": leaked.rsplit("_", 1)[0] + "_sib",
        "patient_id": labels.set_index("record_id").loc[leaked, "patient_id"],
        "cohort": labels.set_index("record_id").loc[leaked, "cohort"]}])],
        ignore_index=True)
    with pytest.raises(AssertionError):
        assert_no_leakage(splits, labels)


def test_duplicate_record_id_rejected():
    labels = _synthetic_labels()
    dup = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        make_lodo_splits(dup, ["a", "b"], 0.1, seed=7)
