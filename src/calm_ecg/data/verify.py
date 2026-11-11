"""Stage 0 — verify raw datasets and write MANIFEST.json.

Gate (see 'CALM-ECG quy trình code kiểm thử.md'):
  PTB-XL records == 21,799 (18,869 patients); Chapman+Ningbo == 45,152;
  CODE-15% == 345,779 exams (233,770 patients); CODE-test == 827.
Counts are taken from metadata files AND cross-checked against an
independent second source (file listing / RECORDS) where available.
Any mismatch is reported, never silently 'fixed'.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _count_csv_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def verify_ptbxl(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    db = d / cfg["database_csv"]
    out = {"present": db.exists()}
    if not out["present"]:
        return out
    n_rows = _count_csv_rows(db)
    with open(db, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        patients = {row["patient_id"] for row in rd}
    records_file = d / "RECORDS"
    n_records_file = (
        sum(1 for line in open(records_file, encoding="utf-8") if line.strip())
        if records_file.exists() else None
    )
    out.update(
        n_records=n_rows,
        n_patients=len(patients),
        n_records_independent=(n_records_file // 2 if n_records_file else None),
        # RECORDS lists both records100 and records500 → /2
        pass_gate=(n_rows == cfg["expected_records"]
                   and len(patients) == cfg["expected_patients"]),
    )
    return out


def verify_chapman(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    rec = d / cfg["records_file"]
    out = {"present": rec.exists()}
    if not out["present"]:
        return out
    # RECORDS lists subfolders; count .hea files as the independent source.
    n_hea = sum(1 for _ in (d / "WFDBRecords").rglob("*.hea"))
    out.update(n_records=n_hea, pass_gate=(n_hea == cfg["expected_records"]))
    return out


def verify_code15(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    ex = d / cfg["exams_csv"]
    out = {"present": ex.exists()}
    if not out["present"]:
        return out
    n_rows = _count_csv_rows(ex)
    with open(ex, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        patients = {row["patient_id"] for row in rd}
    n_hdf5 = len(list(d.glob("exams_part*/*.hdf5"))) + len(list(d.glob("exams_part*.hdf5")))
    out.update(
        n_records=n_rows,
        n_patients=len(patients),
        n_hdf5_parts=n_hdf5,
        pass_gate=(n_rows == cfg["expected_records"]
                   and len(patients) == cfg["expected_patients"]),
    )
    return out


def verify_code_test(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    gold = d / cfg["gold_csv"]
    out = {"present": gold.exists() and (d / cfg["tracings"]).exists()}
    if not out["present"]:
        return out
    n = _count_csv_rows(gold)
    out.update(n_records=n, pass_gate=(n == cfg["expected_records"]))
    return out


def verify_samitrop(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    out = {"present": (d / cfg["hdf5"]).exists()}
    if out["present"]:
        out["note"] = "count inside HDF5 checked in stage 1 (needs h5py)"
    return out


def verify_mimic(root: Path, cfg: dict) -> dict:
    d = root / cfg["dir"]
    return {"present": d.exists(),
            "note": "not downloaded yet" if not d.exists() else "fill expected_records"}


VERIFIERS = {
    "ptbxl": verify_ptbxl,
    "chapman": verify_chapman,
    "code15": verify_code15,
    "code_test": verify_code_test,
    "samitrop": verify_samitrop,
    "mimic": verify_mimic,
}


def run_verify(config: dict, manifest_path: Path) -> dict:
    root = Path(config["data_root"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "data_root": str(root), "cohorts": {}}
    for name, cfg in config["cohorts"].items():
        manifest["cohorts"][name] = VERIFIERS[name](root, cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
