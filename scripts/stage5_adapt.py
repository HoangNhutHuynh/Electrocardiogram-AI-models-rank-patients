"""Run stage 5 adaptation for one LODO fold.

Usage:
    python scripts/stage5_adapt.py [configs/adapt.yaml]
    python scripts/stage5_adapt.py configs/adapt.yaml --held-out chapman --seed 1
Ablation example (GroupDRO off):
    python scripts/stage5_adapt.py configs/adapt.yaml --set w_dro=0

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.train.adapt import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/adapt.yaml")
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pretrain-seed", type=int, default=None,
                    help="encoder checkpoint seed (default 0: one pretrained "
                         "encoder per fold, adapt seeds vary — note in Methods)")
    ap.add_argument("--arch", default="s4d", choices=["s4d", "resnet"],
                    help="encoder; non-default is suffixed onto the run name "
                         "so the s4d results keep their paths")
    ap.add_argument("--set", action="append", default=[],
                    help="override, e.g. --set w_dro=0 --set lora_r=16")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.held_out:
        cfg["held_out"] = args.held_out
    if args.seed is not None:
        cfg["seed"] = args.seed
    suffix = ""
    for kv in args.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown config key {k}"
        cfg[k] = yaml.safe_load(v)
        suffix += f"_{k}{v}"
    p_seed = args.pretrain_seed if args.pretrain_seed is not None else 0
    cfg["arch"] = args.arch   # read by stage7 to keep architectures apart
    tag = "" if args.arch == "s4d" else f"_{args.arch}"
    cfg["pretrain_ckpt"] = str(
        repo / f"results/stage4/pretrain_heldout-{cfg['held_out']}"
               f"_seed{p_seed}{tag}/ckpt_best.pt")
    cfg["run_name"] = (f"adapt_heldout-{cfg['held_out']}"
                       f"_seed{cfg['seed']}{tag}{suffix}")

    out = run(cfg, repo)
    print(f"done. artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
