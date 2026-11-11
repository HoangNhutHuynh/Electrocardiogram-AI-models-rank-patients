"""Run stage 0: verify datasets, write results/MANIFEST.json, print gate status.

Usage (Windows, from repo root):
    python scripts/stage0_verify.py [configs/data.yaml]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.verify import run_verify  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/data.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    manifest = run_verify(config, repo / "results/MANIFEST.json")

    failed = []
    for name, info in manifest["cohorts"].items():
        status = ("MISSING" if not info.get("present")
                  else "PASS" if info.get("pass_gate")
                  else "CHECK")
        print(f"{name:10s} {status:8s} {info}")
        if info.get("present") and info.get("pass_gate") is False:
            failed.append(name)
    print(f"\nManifest written to {repo / 'results/MANIFEST.json'}")
    if failed:
        print(f"GATE FAILED for: {failed} — investigate before stage 1.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
