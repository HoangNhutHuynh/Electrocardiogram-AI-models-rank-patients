"""Stage 4 — SSL pretraining loop (masked reconstruction).

One run = one LODO fold: pretrains ONLY on that fold's train split (the
held-out cohort is excluded from pretraining by construction — full LODO).
Logs: results/stage4/<run_name>/{config.yaml, log.csv, ckpt_last.pt,
ckpt_best.pt}. Resume-safe. Reproducible: seed + config + git hash logged.

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from calm_ecg.data.dataset import EcgDataset
from calm_ecg.models.s4d import MaskedReconModel, S4DEncoder, count_params


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "no-git"
    except OSError:
        return "no-git"


def build_loaders(cfg: dict, repo: Path):
    common = dict(repo_root=repo, data_root=Path(cfg["data_root"]),
                  cohort_dirs=cfg["cohort_dirs"], held_out=cfg["held_out"],
                  crop_len=cfg["crop_len"], seed=cfg["seed"])
    train_ds = EcgDataset(split="train", train=True, **common)
    val_ds = EcgDataset(split="val", train=False, **common)
    dl = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **dl),
            DataLoader(val_ds, shuffle=False, **dl))


@torch.no_grad()
def validate(model, loader, device, amp_dtype, max_batches: int = 200) -> float:
    model.eval()
    losses = []
    for i, (x, pad, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        x, pad = x.to(device), pad.to(device)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            loss, _, _ = model(x, pad)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run(cfg: dict, repo: Path) -> Path:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)

    out = repo / "results/stage4" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    cfg["git_hash"] = git_hash(repo)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    enc = S4DEncoder(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                     n_state=cfg["n_state"], dropout=cfg["dropout"])
    model = MaskedReconModel(enc).to(device)
    print(f"params: {count_params(model):,} | device: {device} | amp: {amp_dtype}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    train_dl, val_dl = build_loaders(cfg, repo)
    steps_per_epoch = len(train_dl)
    total = cfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    start_epoch, best_val = 0, float("inf")
    ckpt_last = out / "ckpt_last.pt"
    if ckpt_last.exists():  # resume
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        print(f"resumed from epoch {state['epoch']}", flush=True)

    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss,lr,sec\n",
                            encoding="utf-8")

    n_skipped = 0  # non-finite loss guard: skip the step, never update weights
    for epoch in range(start_epoch, cfg["epochs"]):
        t0, running = time.time(), []
        for step, (x, pad, _, _) in enumerate(train_dl):
            x, pad = x.to(device, non_blocking=True), pad.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                loss, _, _ = model(x, pad, mask_ratio=cfg["mask_ratio"],
                                   span=cfg["mask_span"])
            if not torch.isfinite(loss):
                n_skipped += 1
                print(f"WARN ep {epoch} step {step}: non-finite loss, "
                      f"step skipped ({n_skipped} total)", flush=True)
                if n_skipped > 50:
                    raise RuntimeError(
                        ">50 non-finite batches — data problem, not noise. "
                        "Run scripts/stage1b_qc_scan.py and retrain.")
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.item())
            if step % cfg["log_every"] == 0:
                print(f"ep {epoch} step {step}/{steps_per_epoch} "
                      f"loss {np.mean(running[-cfg['log_every']:]):.4f}", flush=True)
        val_loss = validate(model, val_dl, device, amp_dtype)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, (epoch + 1) * steps_per_epoch,
                                    round(float(np.mean(running)), 5),
                                    round(val_loss, 5),
                                    sched.get_last_lr()[0],
                                    round(time.time() - t0)])
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best_val": min(best_val, val_loss), "config": cfg}
        torch.save(state, ckpt_last)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, out / "ckpt_best.pt")
        print(f"epoch {epoch}: train {np.mean(running):.4f} "
              f"val {val_loss:.4f} best {best_val:.4f}", flush=True)
    return out
