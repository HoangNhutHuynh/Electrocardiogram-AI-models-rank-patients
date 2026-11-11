"""Stage 2 — harmonize all cohorts into the 6-label space.

Labels: AF, 1dAVb, RBBB, LBBB, SB, STach  (strict 4-cohort intersection).
Semantics per cell: 1 = positive, 0 = negative, NaN = label NOT DEFINED in
this cohort's label space. NaN must be masked out of the loss — NEVER
treated as 0 (tested in tests/unit/test_harmonize.py).

Mapping sources:
  - PTB-XL: SCP-ECG statement codes in `scp_codes` (CinC-2021 merges
    CRBBB≡RBBB, CLBBB≡LBBB; incomplete blocks IRBBB/ILBBB are NOT positives).
  - Chapman/Ningbo: SNOMED-CT codes in the WFDB header Dx field.
  - CODE-15% / CODE-test: native 6 binary columns (ST → STach).
  - MIMIC-IV-ECG: rule-based regex over machine report lines report_0..17
    (UNTESTED until data arrives; must reach P/R >= 0.95 on 200 hand-labels
    before stage 3 — see quy trình giai đoạn 2).

Output per cohort: a tidy table
  record_id, patient_id, cohort, path, age, sex, AF, 1dAVb, RBBB, LBBB, SB, STach

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["AF", "1dAVb", "RBBB", "LBBB", "SB", "STach"]

# --- PTB-XL: SCP statement code -> harmonized label -------------------------
PTBXL_SCP_MAP = {
    "AFIB": "AF",
    "1AVB": "1dAVb",
    "CRBBB": "RBBB",   # complete RBBB only (CinC-2021 CRBBB≡RBBB)
    "CLBBB": "LBBB",
    "SBRAD": "SB",
    "STACH": "STach",
}

# --- Chapman/Ningbo: SNOMED-CT code -> harmonized label ---------------------
CHAPMAN_SNOMED_MAP = {
    "164889003": "AF",      # atrial fibrillation
    "270492004": "1dAVb",   # first degree AV block
    "59118001":  "RBBB",    # right bundle branch block
    "713427006": "RBBB",    # complete RBBB (CinC merge)
    "164909002": "LBBB",    # left bundle branch block
    "733534002": "LBBB",    # complete LBBB (CinC merge)
    "426177001": "SB",      # sinus bradycardia
    "427084000": "STach",   # sinus tachycardia
}

# --- CODE: native column -> harmonized label --------------------------------
CODE_COL_MAP = {"AF": "AF", "1dAVb": "1dAVb", "RBBB": "RBBB",
                "LBBB": "LBBB", "SB": "SB", "ST": "STach"}


def harmonize_ptbxl(db_csv: Path, min_likelihood: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(db_csv, index_col="ecg_id")
    scp = df["scp_codes"].apply(ast.literal_eval)
    out = pd.DataFrame(index=df.index)
    for code, label in PTBXL_SCP_MAP.items():
        pos = scp.apply(lambda d: code in d and d[code] >= min_likelihood)
        out[label] = out.get(label, pd.Series(False, index=df.index)) | pos
    out = out[LABELS].astype(float)  # label space covers all 6 → no NaN
    out.insert(0, "record_id", df.index.astype(str))
    out.insert(1, "patient_id", "ptbxl_" + df["patient_id"].astype(int).astype(str))
    out.insert(2, "cohort", "ptbxl")
    out.insert(3, "path", df["filename_hr"])
    out.insert(4, "age", df["age"])
    out.insert(5, "sex", df["sex"].map({0: "M", 1: "F"}))  # PTB-XL: 0=male,1=female
    out.insert(6, "strat_fold", df["strat_fold"])
    return out.reset_index(drop=True)


def _parse_hea_dx_age_sex(hea_path: Path) -> tuple[list[str], float, str]:
    codes: list[str] = []
    age, sex = np.nan, ""
    for line in hea_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("#Dx:"):
            codes = [c.strip() for c in s[4:].split(",") if c.strip()]
        elif s.startswith("#Age:"):
            try:
                age = float(s[5:].strip())
            except ValueError:
                age = np.nan
        elif s.startswith("#Sex:"):
            sex = s[5:].strip()[:1].upper()  # Male/Female -> M/F
    return codes, age, sex


def harmonize_chapman(wfdb_root: Path, progress_every: int = 5000) -> pd.DataFrame:
    rows = []
    hea_files = sorted(wfdb_root.rglob("*.hea"))
    print(f"chapman: found {len(hea_files)} headers, parsing...", flush=True)
    for i, hea in enumerate(hea_files, 1):
        if progress_every and i % progress_every == 0:
            print(f"chapman: {i}/{len(hea_files)}", flush=True)
        codes, age, sex = _parse_hea_dx_age_sex(hea)
        rec = {lab: 0.0 for lab in LABELS}
        for c in codes:
            lab = CHAPMAN_SNOMED_MAP.get(c)
            if lab:
                rec[lab] = 1.0
        rows.append({"record_id": hea.stem,
                     "patient_id": "chapman_" + hea.stem,  # one record per patient
                     "cohort": "chapman",
                     "path": str(hea.with_suffix("").relative_to(wfdb_root)),
                     "age": age, "sex": sex, **rec})
    return pd.DataFrame(rows, columns=["record_id", "patient_id", "cohort",
                                       "path", "age", "sex", *LABELS])


def harmonize_code15(exams_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(exams_csv)
    out = pd.DataFrame({
        "record_id": "code15_" + df["exam_id"].astype(str),
        "patient_id": "code15_" + df["patient_id"].astype(str),
        "cohort": "code15",
        "path": df["trace_file"] + "#" + df["exam_id"].astype(str),
        "age": df["age"],
        "sex": np.where(df["is_male"].astype(bool), "M", "F"),
    })
    for col, lab in CODE_COL_MAP.items():
        out[lab] = df[col].astype(bool).astype(float)
    return out[["record_id", "patient_id", "cohort", "path", "age", "sex", *LABELS]]


def harmonize_code_test(gold_csv: Path, attributes_csv: Path) -> pd.DataFrame:
    gold = pd.read_csv(gold_csv)
    attrs = pd.read_csv(attributes_csv)
    n = len(gold)
    out = pd.DataFrame({
        "record_id": [f"codetest_{i}" for i in range(n)],
        "patient_id": [f"codetest_{i}" for i in range(n)],
        "cohort": "code_test",
        "path": [f"ecg_tracings.hdf5#{i}" for i in range(n)],
        "age": attrs["age"] if "age" in attrs else np.nan,
        "sex": attrs["sex"] if "sex" in attrs else "",
    })
    for col, lab in CODE_COL_MAP.items():
        out[lab] = gold[col].astype(bool).astype(float)
    return out[["record_id", "patient_id", "cohort", "path", "age", "sex", *LABELS]]


# --- MIMIC-IV-ECG regex labeller (UNTESTED — data not downloaded yet) --------
# Negation-aware match over free-text machine reports. Must be validated on
# 200 hand-labelled reports (P/R >= 0.95, two annotators, kappa >= 0.8)
# BEFORE being used in any training run.
MIMIC_PATTERNS = {
    "AF":    re.compile(r"atrial fibrillation", re.I),
    "1dAVb": re.compile(r"(?:first[- ]degree (?:a[-v]+|atrioventricular) block|1st degree av block)", re.I),
    "RBBB":  re.compile(r"right bundle[- ]branch block", re.I),
    "LBBB":  re.compile(r"left bundle[- ]branch block", re.I),
    "SB":    re.compile(r"sinus bradycardia", re.I),
    "STach": re.compile(r"sinus tachycardia", re.I),
}
# Cues that negate/soften a finding when they appear shortly BEFORE the match.
_NEG_CUES = re.compile(
    r"(?:\bno\b|\bwithout\b|negative for|cannot rule out|rule out|\br/o\b|"
    r"possible|probable|borderline|consider|vs\.?)\s*$", re.I)
_INCOMPLETE = re.compile(r"incomplete\s*$", re.I)
_NEG_WINDOW = 30  # chars before the match inspected for a negation cue


def _is_positive(text: str, m: re.Match) -> bool:
    prefix = text[max(0, m.start() - _NEG_WINDOW): m.start()]
    if _NEG_CUES.search(prefix):
        return False
    if _INCOMPLETE.search(prefix):  # incomplete R/LBBB is not a positive
        return False
    return True


def mimic_labels_from_report(report_text: str) -> dict[str, float]:
    """Label one concatenated machine report. See module docstring caveat."""
    text = " ".join(report_text.split())
    out = {}
    for lab, pat in MIMIC_PATTERNS.items():
        out[lab] = float(any(_is_positive(text, m) for m in pat.finditer(text)))
    return out
