"""Run stage 4 SSL pretraining for one LODO fold.

Usage:
    python scripts/stage4_pretrain.py [configs/pretrain.yaml]
Override fold/seed without editing the file:
    python scripts/stage4_pretrain.py configs/pretrain.yaml --held-out chapman --seed 1

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.train.pretrain import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/pretrain.yaml")
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arch", default="s4d", choices=["s4d", "resnet"],
                    help="encoder; non-default is suffixed onto the run name "
                         "so the s4d results keep their paths")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.held_out:
        cfg["held_out"] = args.held_out
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["arch"] = args.arch
    tag = "" if args.arch == "s4d" else f"_{args.arch}"
    cfg["run_name"] = (f"pretrain_heldout-{cfg['held_out']}"
                       f"_seed{cfg['seed']}{tag}")

    out = run(cfg, repo)
    print(f"done. artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
