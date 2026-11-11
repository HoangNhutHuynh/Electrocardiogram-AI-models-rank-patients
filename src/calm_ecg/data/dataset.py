"""Unified PyTorch Dataset over the harmonized cohorts.

Reads results/labels_all.csv + results/splits_lodo.json and serves
    x:     float32 (12, crop_len)  per-record z-scored, 500 Hz
    pad:   bool    (crop_len,)     True = real sample
    y:     float32 (6,)            harmonized labels, NaN = undefined
    group: int                     demographic group id (sex x age-bin)
Crops are sampled inside the padding mask. Per-record z-score (over real
samples, per lead) absorbs the residual cross-cohort amplitude scale
difference observed in stage 1 (CODE ~2x PTB-XL median p-p; logged in
KNOWN_DISCREPANCIES.md).

Windows note: HDF5/wfdb handles are opened lazily inside __getitem__ and
cached per process, so num_workers>0 with spawn works.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .harmonize import LABELS
from .signals import CodeHdf5, load_wfdb, resolve_code_path

AGE_BINS = [0, 40, 60, 75, 200]  # -> 4 bins; group = sex(2) x age_bin(4)


def demographic_group(sex: str, age: float) -> int:
    """0..7; -1 if sex or age missing (excluded from GroupDRO, kept in data)."""
    if sex not in ("M", "F") or not np.isfinite(age):
        return -1
    a = int(np.digitize([age], AGE_BINS[1:-1])[0])  # 0..3
    return (0 if sex == "M" else 1) * 4 + a


class EcgDataset(Dataset):
    def __init__(self, repo_root: Path, data_root: Path, cohort_dirs: dict,
                 split: str, held_out: str, crop_len: int = 2500,
                 seed: int = 0, train: bool = True,
                 record_ids: list[str] | None = None):
        repo_root, self.data_root = Path(repo_root), Path(data_root)
        self.cohort_dirs = cohort_dirs  # {"ptbxl": "PTB-XL v1.0.3", ...}
        labels = pd.read_csv(repo_root / "results/labels_all.csv",
                             dtype={"record_id": str, "patient_id": str})
        if record_ids is not None:  # explicit set outside the LODO splits (e.g. CODE-test)
            wanted = set(record_ids)
        else:
            splits = json.loads(
                (repo_root / "results/splits_lodo.json").read_text(encoding="utf-8"))
            wanted = set(splits["folds"][held_out][split])
        # explicit, versioned QC exclusions (stage 1b scan) — never silent
        excl_path = repo_root / "results/qc_exclusions.csv"
        self.n_excluded = 0
        if excl_path.exists():
            excl = set(pd.read_csv(excl_path, dtype={"record_id": str})["record_id"])
            self.n_excluded = len(wanted & excl)
            wanted -= excl
            print(f"[{held_out}/{split}] excluded {self.n_excluded} records "
                  f"per {excl_path.name}", flush=True)
        self.df = labels[labels["record_id"].isin(wanted)].reset_index(drop=True)
        assert len(self.df) == len(wanted), "records in split missing from labels"
        self.crop_len, self.train = crop_len, train
        self.rng = np.random.default_rng(seed)
        self._h5_cache: dict[str, CodeHdf5] = {}

    def __len__(self) -> int:
        return len(self.df)

    # --------------------------------------------------------------- load ---
    def _load(self, row) -> tuple[np.ndarray, np.ndarray]:
        cohort, path = row["cohort"], row["path"]
        if cohort in ("ptbxl", "chapman"):
            base = (self.data_root / self.cohort_dirs[cohort] / "WFDBRecords"
                    if cohort == "chapman"
                    else self.data_root / self.cohort_dirs[cohort])
            return load_wfdb(base / path)
        if cohort in ("code15", "code_test"):
            fname, key = path.split("#")
            cache_key = f"{cohort}/{fname}"
            if cache_key not in self._h5_cache:
                d = self.data_root / self.cohort_dirs[cohort]
                p = resolve_code_path(d, fname) if cohort == "code15" else d / fname
                self._h5_cache[cache_key] = CodeHdf5(p)
            h5 = self._h5_cache[cache_key]
            return (h5.by_exam_id(int(key)) if cohort == "code15"
                    else h5.by_row(int(key)))
        raise ValueError(f"unknown cohort {cohort}")

    # --------------------------------------------------------------- crop ---
    def _crop(self, sig: np.ndarray, mask: np.ndarray):
        idx = np.flatnonzero(mask)
        if idx.size == 0:  # fully padded (shouldn't happen; stage1 n_bad=0)
            start = 0
        else:
            lo, hi = idx[0], idx[-1] + 1
            if hi - lo <= self.crop_len:
                start = max(0, lo - max(0, self.crop_len - (hi - lo)) // 2)
            elif self.train:
                start = int(self.rng.integers(lo, hi - self.crop_len + 1))
            else:
                start = lo + (hi - lo - self.crop_len) // 2  # deterministic val
        end = start + self.crop_len
        out = np.zeros((12, self.crop_len), dtype=np.float32)
        m = np.zeros(self.crop_len, dtype=bool)
        seg = sig[:, start:end]
        out[:, :seg.shape[1]] = seg
        m[:seg.shape[1]] = mask[start:end][:seg.shape[1]]
        return out, m

    @staticmethod
    def _zscore(sig: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.any():
            mu = sig[:, mask].mean(axis=1, keepdims=True)
            sd = sig[:, mask].std(axis=1, keepdims=True)
        else:  # fully padded / flat record: pass through unchanged
            mu = np.float32(0.0)
            sd = np.float32(1.0)
        out = (sig - mu) / np.maximum(sd, np.float32(1e-6))
        return out.astype(np.float32, copy=False)  # dtype stable on ALL paths

    @staticmethod
    def _sanitize(sig: np.ndarray, mask: np.ndarray):
        """Zero non-finite samples and drop them from the mask (defense in
        depth behind the stage-1b QC scan — one bad sample must never
        poison the optimizer with NaN)."""
        bad = ~np.isfinite(sig)
        if bad.any():
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            mask = mask & ~bad.any(axis=0)
        return sig, mask

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            sig, mask = self._load(row)
        except Exception as e:  # unreadable file: name the record, never hide
            raise RuntimeError(
                f"failed to load record_id={row['record_id']} "
                f"({row['cohort']}): {type(e).__name__}: {e}. "
                "Run scripts/stage1b_qc_scan.py to build qc_exclusions.csv."
            ) from e
        sig, mask = self._sanitize(sig, mask)
        sig, m = self._crop(sig, mask)
        sig = (self._zscore(sig, m) * m).astype(np.float32, copy=False)
        np.clip(sig, -20.0, 20.0, out=sig)  # near-flat records -> huge z-scores
        y = row[LABELS].to_numpy(dtype=np.float32)
        g = demographic_group(str(row["sex"]), float(row["age"]))
        return (torch.from_numpy(sig), torch.from_numpy(m),
                torch.from_numpy(y), torch.tensor(g, dtype=torch.long))


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor,
                    reduction: str = "mean") -> torch.Tensor:
    """BCE over DEFINED labels only. NaN in y = undefined -> excluded.
    Guarded by tests/unit/test_dataset.py::test_masked_bce_ignores_nan."""
    valid = ~torch.isnan(y)
    if not valid.any():
        return logits.sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[valid], y[valid], reduction=reduction)
    return loss
