"""Stage 1 sanity check on REAL data (run once, log output to results/stage1).

Per cohort: sample N records, standardize via calm_ecg.data.signals, then
  - median lead-II peak-to-peak amplitude  (gate: 0.5–2.5 mV, same scale
    across cohorts — catches the 4.88 uV/bit and CODE-scaling pitfalls)
  - % records with NaN/flat lines (reported, not dropped)
  - SaMi-Trop record count inside HDF5 (gate: 1631)
  - overlay plots of 5 records/cohort -> results/stage1/overlays_<cohort>.png

Usage: python scripts/stage1_sanity.py [configs/data.yaml] [N_per_cohort=100]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.signals import (CodeHdf5, load_wfdb,  # noqa: E402
                                   quality_flags, resolve_code_path)

RNG = np.random.default_rng(20260101)


def _stats(sigs: list, masks: list) -> dict:
    pp, bad = [], 0
    for s, m in zip(sigs, masks):
        q = quality_flags(s)
        if q["has_nan"] or q["flat"]:
            bad += 1
            continue
        lead2 = s[1][m] if m.any() else s[1]
        pp.append(float(np.ptp(lead2)))
    return {"n": len(sigs), "n_bad": bad,
            "lead2_pp_median_mV": round(float(np.median(pp)), 3) if pp else None,
            "lead2_pp_iqr_mV": [round(float(np.percentile(pp, q)), 3)
                                for q in (25, 75)] if pp else None,
            "pass_gate": bool(pp) and 0.5 <= float(np.median(pp)) <= 2.5}


def _overlay(sigs: list, title: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sigs), 1, figsize=(10, 2 * len(sigs)),
                             sharex=True)
    for ax, s in zip(np.atleast_1d(axes), sigs):
        ax.plot(np.arange(s.shape[-1]) / 500.0, s[1], lw=0.5)
        ax.set_ylabel("II (mV)")
    axes = np.atleast_1d(axes)
    axes[0].set_title(title)
    axes[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    co = config["cohorts"]
    out_dir = repo / "results/stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # --- PTB-XL (records500) and Chapman via wfdb ---------------------------
    for name, base, path_col in [
        ("ptbxl", root / co["ptbxl"]["dir"], "path"),
        ("chapman", root / co["chapman"]["dir"] / "WFDBRecords", "path"),
    ]:
        labels = pd.read_csv(repo / f"results/labels_{name}.csv")
        rows = labels.sample(min(n_sample, len(labels)), random_state=1)
        sigs, masks = [], []
        for p in rows[path_col]:
            s, m = load_wfdb(base / p)
            sigs.append(s)
            masks.append(m)
        report[name] = _stats(sigs, masks)
        _overlay(sigs[:5], name, out_dir / f"overlays_{name}.png")
        print(name, report[name], flush=True)

    # --- CODE-15% (sample within one random part per record) ----------------
    labels = pd.read_csv(repo / "results/labels_code15.csv")
    rows = labels.sample(min(n_sample, len(labels)), random_state=1)
    sigs, masks = [], []
    open_files: dict = {}
    for p in rows["path"]:
        fname, exam_id = p.split("#")
        if fname not in open_files:
            open_files[fname] = CodeHdf5(
                resolve_code_path(root / co["code15"]["dir"], fname))
        try:
            s, m = open_files[fname].by_exam_id(int(exam_id))
        except KeyError:
            continue
        sigs.append(s)
        masks.append(m)
    for f in open_files.values():
        f.close()
    report["code15"] = _stats(sigs, masks)
    _overlay(sigs[:5], "code15", out_dir / "overlays_code15.png")
    print("code15", report["code15"], flush=True)

    # --- SaMi-Trop count gate ------------------------------------------------
    st = CodeHdf5(root / co["samitrop"]["dir"] / co["samitrop"]["hdf5"])
    report["samitrop"] = {"n_records_in_hdf5": len(st),
                          "pass_gate": len(st) == co["samitrop"]["expected_records"]}
    st.close()
    print("samitrop", report["samitrop"], flush=True)

    (out_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport + overlay PNGs in {out_dir}. "
          "Check amplitude gate (0.5–2.5 mV median, same scale everywhere) "
          "and eyeball the overlays before stage 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
