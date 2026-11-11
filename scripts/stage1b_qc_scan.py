"""Stage 1b — full QC scan of EVERY record, writing results/qc_exclusions.csv.

Why: the 100-sample stage-1 sanity missed (a) at least one unreadable Chapman
WFDB file (wfdb IndexError) and (b) at least one record whose values are
non-finite or flat (NaN-poisoned pretraining at step ~300). This scan checks
all 413k records once and produces an explicit, versioned exclusion list:
    record_id, cohort, reason   (reason: unreadable:<Exc> | nonfinite | flat)
EcgDataset auto-applies it. Excluded counts go in the paper's preprocessing
section — records are excluded by listed reason, never silently.

Speed: wfdb records are read one by one (PTB-XL 21.8k + Chapman 45.2k);
CODE-15%/CODE-test HDF5 are read sequentially in chunks (fast path).
Expect roughly 30–90 min depending on disk. Re-runnable; overwrites the csv.

Usage: python scripts/stage1b_qc_scan.py [configs/data.yaml]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.signals import load_wfdb, resolve_code_path  # noqa: E402

FLAT_EPS = 1e-6


def _flags(sig: np.ndarray) -> str | None:
    if not np.isfinite(sig).all():
        return "nonfinite"
    if np.ptp(sig, axis=-1).max() < FLAT_EPS:
        return "flat"
    return None


def scan_wfdb(labels: pd.DataFrame, base: Path, cohort: str, out: list) -> None:
    rows = labels[labels["cohort"] == cohort]
    t0 = time.time()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        if i % 2000 == 0:
            print(f"{cohort}: {i}/{len(rows)} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        try:
            sig, _ = load_wfdb(base / r["path"])
        except Exception as e:  # any read failure -> excluded with reason
            out.append((r["record_id"], cohort, f"unreadable:{type(e).__name__}"))
            continue
        reason = _flags(sig)
        if reason:
            out.append((r["record_id"], cohort, reason))


def scan_code_hdf5(labels: pd.DataFrame, code_dir: Path, out: list,
                   chunk: int = 256) -> None:
    """Sequential chunked pass over every part file (no resampling needed —
    QC flags are invariant to it)."""
    import h5py
    rows = labels[labels["cohort"] == "code15"]
    by_part: dict[str, dict[int, str]] = {}
    for _, r in rows.iterrows():
        fname, exam_id = r["path"].split("#")
        by_part.setdefault(fname, {})[int(exam_id)] = r["record_id"]
    for fname, exams in sorted(by_part.items()):
        t0 = time.time()
        with h5py.File(resolve_code_path(code_dir, fname), "r") as f:
            ids = f["exam_id"][:]
            tr = f["tracings"]
            keep = [(i, int(e)) for i, e in enumerate(ids) if int(e) in exams]
            for lo in range(0, len(keep), chunk):
                batch = keep[lo:lo + chunk]
                idx = [i for i, _ in batch]
                data = tr[idx[0]:idx[-1] + 1]  # contiguous superset read
                for i, exam in batch:
                    sig = data[i - idx[0]].T
                    # interior of padding: QC on the whole row is fine
                    reason = _flags(sig)
                    if reason:
                        out.append((exams[exam], "code15", reason))
        print(f"code15 {fname}: done in {(time.time() - t0) / 60:.1f} min",
              flush=True)


def scan_code_test(labels: pd.DataFrame, d: Path, out: list) -> None:
    import h5py
    rows = labels[labels["cohort"] == "code_test"]
    with h5py.File(d / "ecg_tracings.hdf5", "r") as f:
        tr = f["tracings"][:]
    for _, r in rows.iterrows():
        i = int(r["path"].split("#")[1])
        reason = _flags(tr[i].T)
        if reason:
            out.append((r["record_id"], "code_test", reason))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    co = config["cohorts"]
    labels = pd.read_csv(repo / "results/labels_all.csv",
                         dtype={"record_id": str, "patient_id": str})

    out: list[tuple[str, str, str]] = []
    scan_wfdb(labels, root / co["ptbxl"]["dir"], "ptbxl", out)
    scan_wfdb(labels, root / co["chapman"]["dir"] / "WFDBRecords", "chapman", out)
    scan_code_hdf5(labels, root / co["code15"]["dir"], out)
    scan_code_test(labels, root / co["code_test"]["dir"], out)

    dst = repo / "results/qc_exclusions.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "cohort", "reason"])
        w.writerows(out)

    summary = pd.DataFrame(out, columns=["record_id", "cohort", "reason"])
    print(f"\n{len(out)} records excluded -> {dst}")
    if len(out):
        print(summary.groupby(["cohort", "reason"]).size().to_string())
        unreadable = summary[summary["reason"].str.startswith("unreadable")]
        if len(unreadable):
            print("\nUnreadable files are likely corrupt downloads — verify "
                  "against SHA256SUMS.txt and re-download those files, then "
                  "re-run this scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
