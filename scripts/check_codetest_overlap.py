"""Amendment 3 leakage gate: is any CODE-test tracing also in CODE-15%?

Hashes every raw tracing (exact float32 bytes, and rounded to 1e-3 to catch
storage-precision differences) and reports matches.
Output: results/codetest_overlap.json  {n_exact, n_rounded, overlap_record_ids}
Usage: python scripts/check_codetest_overlap.py [configs/data.yaml]
shortcut: exact/rounded hashing misses re-recorded exams of the same patient
(acknowledged limitation in Amendment 3).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.signals import resolve_code_path  # noqa: E402


def hashes(arr: np.ndarray) -> tuple[str, str]:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return (hashlib.sha1(a.tobytes()).hexdigest(),
            hashlib.sha1(np.round(a, 3).tobytes()).hexdigest())


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root, co = Path(cfg["data_root"]), cfg["cohorts"]

    with h5py.File(root / co["code_test"]["dir"] / co["code_test"]["tracings"], "r") as f:
        ct = f["tracings"][:]
    exact, rounded = {}, {}
    for i, row in enumerate(ct):
        e, r = hashes(row)
        exact[e], rounded[r] = i, i
    print(f"CODE-test: {len(ct)} tracings, dtype {ct.dtype}", flush=True)

    hit_e, hit_r = set(), set()
    code_dir = root / co["code15"]["dir"]
    for k in range(18):
        t0 = time.time()
        with h5py.File(resolve_code_path(code_dir, f"exams_part{k}.hdf5"), "r") as f:
            tr = f["tracings"]
            for lo in range(0, tr.shape[0], 2048):
                for row in tr[lo:lo + 2048]:
                    e, r = hashes(row)
                    if e in exact:
                        hit_e.add(exact[e])
                    if r in rounded:
                        hit_r.add(rounded[r])
        print(f"part{k}: {time.time() - t0:.0f}s, matches so far exact={len(hit_e)} "
              f"rounded={len(hit_r)}", flush=True)

    hits = sorted(hit_e | hit_r)
    res = {"n_code_test": int(len(ct)), "n_exact": len(hit_e), "n_rounded": len(hit_r),
           "overlap_record_ids": [f"codetest_{i}" for i in hits],
           "fraction": round(len(hits) / len(ct), 4)}
    (repo / "results/codetest_overlap.json").write_text(json.dumps(res, indent=2),
                                                        encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "overlap_record_ids"}))
    print("GATE:", "in-population analysis allowed" if res["fraction"] <= 0.05
          else "> 5% overlap -> drop in-population analysis (Amendment 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
