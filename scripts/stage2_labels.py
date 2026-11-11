"""Run stage 2: harmonize labels for the cohorts present on disk.

Writes results/labels_<cohort>.csv and results/labels_all.csv, then prints
per-cohort prevalence of the 6 harmonized labels (gate: compare against the
published prevalences, tolerance ±0.5 percentage points — manual check).

Usage: python scripts/stage2_labels.py [configs/data.yaml]
MIMIC is skipped automatically while not downloaded.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.harmonize import (LABELS, harmonize_chapman,  # noqa: E402
                                     harmonize_code15, harmonize_code_test,
                                     harmonize_ptbxl)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    co = config["cohorts"]
    out_dir = repo / "results"
    out_dir.mkdir(exist_ok=True)

    tables = {}
    jobs = {
        "ptbxl": lambda: harmonize_ptbxl(
            root / co["ptbxl"]["dir"] / co["ptbxl"]["database_csv"],
            config["ptbxl_min_likelihood"]),
        "chapman": lambda: harmonize_chapman(
            root / co["chapman"]["dir"] / "WFDBRecords"),
        "code15": lambda: harmonize_code15(
            root / co["code15"]["dir"] / co["code15"]["exams_csv"]),
        "code_test": lambda: harmonize_code_test(
            root / co["code_test"]["dir"] / co["code_test"]["gold_csv"],
            root / co["code_test"]["dir"] / co["code_test"]["attributes_csv"]),
    }
    for name, job in jobs.items():
        try:
            df = job()
        except FileNotFoundError as e:
            print(f"{name}: SKIPPED (missing file: {e})")
            continue
        df.to_csv(out_dir / f"labels_{name}.csv", index=False)
        tables[name] = df
        prev = (df[LABELS].mean() * 100).round(2)
        print(f"\n{name}: n={len(df)}  prevalence (%):")
        print(prev.to_string())

    if tables:
        all_df = pd.concat(tables.values(), ignore_index=True)
        all_df.to_csv(out_dir / "labels_all.csv", index=False)
        print(f"\nWrote {out_dir / 'labels_all.csv'} ({len(all_df)} records). "
              "Compare prevalences with descriptor papers before stage 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
