"""Run stage 3: generate LODO splits from results/labels_all.csv.

Writes results/splits_lodo.json after assert_no_leakage passes.
MIMIC is dropped from the rotation automatically while its labels are absent
(3-cohort LODO until then — the code path is identical).

Usage: python scripts/stage3_splits.py [configs/data.yaml]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.splits import (assert_no_leakage,  # noqa: E402
                                  make_lodo_splits, save_splits)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(repo / "results/labels_all.csv",
                         dtype={"record_id": str, "patient_id": str})

    wanted = config["lodo"]["source_cohorts"]
    present = [c for c in wanted if (labels["cohort"] == c).any()]
    absent = set(wanted) - set(present)
    if absent:
        print(f"NOTE: cohorts absent from labels table, dropped from LODO: {absent}")

    splits = make_lodo_splits(labels, present,
                              config["lodo"]["val_fraction"],
                              config["lodo"]["seed"])
    assert_no_leakage(splits, labels)
    out = repo / "results/splits_lodo.json"
    save_splits(splits, out)
    for held_out, fold in splits["folds"].items():
        print(f"held-out={held_out:8s} train={len(fold['train']):7d} "
              f"val={len(fold['val']):6d} test={len(fold['test']):7d}")
    print(f"\nLeakage check PASSED. Splits written to {out} — commit this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
