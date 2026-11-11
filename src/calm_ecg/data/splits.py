"""Stage 3 — LODO splits with zero patient-level leakage.

For each held-out cohort C in lodo.source_cohorts:
  test  = ALL records of C (untouched; evaluated exactly once at the end)
  train/val = records of the remaining source cohorts, split by patient_id
              (GroupShuffleSplit, fixed seed) — val never shares a patient
              with train.
Splits are deterministic for a given (seed, label table) and are committed
to the repo as immutable JSON. assert_no_leakage() is the pre-flight check
run before EVERY training job.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_lodo_splits(labels: pd.DataFrame, source_cohorts: list[str],
                     val_fraction: float, seed: int) -> dict:
    """labels: harmonized table with columns record_id, patient_id, cohort."""
    required = {"record_id", "patient_id", "cohort"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels table missing columns: {missing}")
    if labels["record_id"].duplicated().any():
        raise ValueError("duplicate record_id in labels table")

    splits = {"seed": seed, "val_fraction": val_fraction, "folds": {}}
    for held_out in source_cohorts:
        pool = labels[labels["cohort"].isin(set(source_cohorts) - {held_out})]
        test = labels[labels["cohort"] == held_out]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction,
                                random_state=seed)
        tr_idx, va_idx = next(gss.split(pool, groups=pool["patient_id"]))
        splits["folds"][held_out] = {
            "train": pool.iloc[tr_idx]["record_id"].tolist(),
            "val": pool.iloc[va_idx]["record_id"].tolist(),
            "test": test["record_id"].tolist(),
        }
    return splits


def assert_no_leakage(splits: dict, labels: pd.DataFrame) -> None:
    """Hard pre-flight gate. Raises AssertionError on any violation."""
    rec2pat = dict(zip(labels["record_id"], labels["patient_id"]))
    rec2coh = dict(zip(labels["record_id"], labels["cohort"]))
    for held_out, fold in splits["folds"].items():
        tr, va, te = (set(fold["train"]), set(fold["val"]), set(fold["test"]))
        assert not (tr & va) and not (tr & te) and not (va & te), \
            f"[{held_out}] record overlap between splits"
        tr_pat = {rec2pat[r] for r in tr}
        va_pat = {rec2pat[r] for r in va}
        assert not (tr_pat & va_pat), \
            f"[{held_out}] patient overlap train/val: {len(tr_pat & va_pat)}"
        assert all(rec2coh[r] == held_out for r in te), \
            f"[{held_out}] test contains records from other cohorts"
        assert all(rec2coh[r] != held_out for r in tr | va), \
            f"[{held_out}] held-out cohort leaked into train/val"


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1), encoding="utf-8")


def check_codetest_disjoint(code15: pd.DataFrame, code_test_ids: set) -> int:
    """Flagged open question: CODE-15% vs CODE-test patient disjointness.

    Returns the number of overlapping exam ids found (0 expected). If > 0,
    those exams must be dropped from training — report, don't hide.
    NOTE: CODE-test ships without original exam_ids in some distributions;
    in that case this check must be done on waveform hashes in stage 1.
    """
    return len({r.split("_")[-1] for r in code15["record_id"]} & code_test_ids)
