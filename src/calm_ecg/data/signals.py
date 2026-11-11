"""Stage 1 — waveform loading and on-the-fly standardization.

Every record, whatever the cohort, is returned as:
    sig:  float32 (12, L) in mV at 500 Hz, standard lead order (LEADS)
    mask: bool    (L,)    True where the sample is real (False = zero-padding)

Design decision: signals are standardized on the fly inside the Dataset
rather than materialized to disk (413k records ≈ 100+ GB duplicated).
Gates (see quy trình, stage 1):
    - resample round-trip NRMSE < 1% (tests/unit/test_signals.py)
    - median lead-II peak-to-peak per cohort in 0.5–2.5 mV (stage1_sanity.py)
    - no NaN/Inf in output; flat-line records reported, not silently dropped

ASSUMPTION (verified empirically by stage1_sanity.py, not trusted from docs):
    - PTB-XL / Chapman via wfdb: physical units already mV after rdsamp.
    - CODE-15% / CODE-test / SaMi-Trop HDF5 'tracings': documented as mV;
      the Zenodo notes are contradictory, so the sanity gate is the authority.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_FS = 500
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
_LEAD_ALIASES = {"D1": "I", "DI": "I", "D2": "II", "DII": "II",
                 "D3": "III", "DIII": "III"}


def _canon(name: str) -> str:
    n = name.strip().upper()
    return _LEAD_ALIASES.get(n, n)


def reorder_leads(sig: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """(L, 12) or (12, L) with named leads -> (12, L) in LEADS order."""
    if sig.shape[0] != 12 and sig.shape[1] == 12:
        sig = sig.T
    names = [_canon(n) for n in sig_names]
    if sorted(names) != sorted(LEADS):
        raise ValueError(f"unexpected lead set: {names}")
    idx = [names.index(l) for l in LEADS]
    return sig[idx]


def resample_to_target(sig: np.ndarray, fs: int) -> np.ndarray:
    """Polyphase resample (12, L) from fs to TARGET_FS. 400->500 uses 5/4."""
    if fs == TARGET_FS:
        return sig
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(TARGET_FS, fs)
    return resample_poly(sig, TARGET_FS // g, fs // g, axis=-1)


def padding_mask(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """True where real signal. CODE zero-pads to 4096 symmetrically: detect
    leading/trailing all-lead-zero runs. Interior zeros are NOT padding."""
    nz = np.any(np.abs(sig) > eps, axis=0)
    idx = np.flatnonzero(nz)
    mask = np.zeros(sig.shape[-1], dtype=bool)
    if idx.size:
        mask[idx[0]: idx[-1] + 1] = True
    return mask


def quality_flags(sig: np.ndarray) -> dict:
    return {
        "has_nan": bool(np.isnan(sig).any() or np.isinf(sig).any()),
        "flat": bool(np.ptp(sig, axis=-1).max() < 1e-6),
    }


# ---------------------------------------------------------------- loaders ---
def load_wfdb(record_path_no_ext: Path) -> tuple[np.ndarray, np.ndarray]:
    """PTB-XL (records500) and Chapman/Ningbo. Returns (sig mV, mask)."""
    import wfdb
    rec = wfdb.rdrecord(str(record_path_no_ext))  # p_signal: (L, 12) physical
    sig = reorder_leads(rec.p_signal.astype(np.float32), rec.sig_name)
    sig = resample_to_target(sig, int(rec.fs))
    return sig, padding_mask(sig)


def resolve_code_path(code_dir: Path, trace_file: str) -> Path:
    """Resolve a CODE trace_file against both known on-disk layouts:
    flat (CODE-15%/exams_partX.hdf5) and nested after unzip
    (CODE-15%/exams_partX/exams_partX.hdf5)."""
    flat = code_dir / trace_file
    if flat.exists():
        return flat
    nested = code_dir / Path(trace_file).stem / trace_file
    if nested.exists():
        return nested
    raise FileNotFoundError(f"{trace_file} not found under {code_dir} "
                            "(tried flat and nested layouts)")


class CodeHdf5:
    """Random access into CODE-15% parts / CODE-test / SaMi-Trop HDF5.

    CODE-15% parts: datasets 'exam_id' (N,), 'tracings' (N, 4096, 12) @400Hz.
    CODE-test: 'tracings' only, row order == gold_standard.csv order.
    Keeps one open h5py handle per file (cheap; close() when done).
    """

    def __init__(self, path: Path, fs: int = 400):
        import h5py
        self.f = h5py.File(path, "r")
        self.fs = fs
        self.exam_index = None
        if "exam_id" in self.f:
            ids = self.f["exam_id"][:]
            self.exam_index = {int(e): i for i, e in enumerate(ids) if e != 0}

    def __len__(self) -> int:
        return self.f["tracings"].shape[0]

    def by_row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        raw = self.f["tracings"][row].astype(np.float32).T  # (12, 4096)
        mask400 = padding_mask(raw)
        sig = resample_to_target(raw, self.fs)
        # resample the mask by index scaling (padding runs are contiguous)
        mask = np.zeros(sig.shape[-1], dtype=bool)
        idx = np.flatnonzero(mask400)
        if idx.size:
            a = int(idx[0] * sig.shape[-1] / raw.shape[-1])
            b = int((idx[-1] + 1) * sig.shape[-1] / raw.shape[-1])
            mask[a:b] = True
        return sig, mask

    def by_exam_id(self, exam_id: int) -> tuple[np.ndarray, np.ndarray]:
        if self.exam_index is None:
            raise KeyError("this HDF5 has no exam_id dataset; use by_row")
        return self.by_row(self.exam_index[int(exam_id)])

    def close(self) -> None:
        self.f.close()
