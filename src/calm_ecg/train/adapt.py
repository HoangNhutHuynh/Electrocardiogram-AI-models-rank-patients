"""Stage 5 — fairness-aware parameter-efficient adaptation (CALM-ECG).

Loads the stage-4 pretrained encoder for the SAME LODO fold, applies LoRA
to the top blocks, and fine-tunes with:
    L = (1 - w_dro) * BCE_all + w_dro * GroupDRO(BCE_per_sample)
        + w_con * fair-SupCon(pooled embeddings)
Model selection: masked macro-AUROC on the fold's val split (source cohorts
only — the held-out cohort is never touched here).
Ablations = config switches: w_dro=0, w_con=0, lora_top_blocks=0 (head only).

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from calm_ecg.data.dataset import EcgDataset
from calm_ecg.eval.metrics import masked_macro_auroc, worst_group_auroc
from calm_ecg.models.lora import apply_lora, trainable_fraction
from calm_ecg.models import build_encoder
from calm_ecg.models.s4d import ClsModel
from calm_ecg.train.fairness import (GroupDRO, fair_supcon,
                                     per_sample_masked_bce)
from calm_ecg.train.pretrain import build_loaders, git_hash, set_seed


class PooledClsModel(ClsModel):
    """ClsModel that also returns the pooled embedding (for fair-SupCon)."""

    def forward_with_features(self, x, pad):
        h = self.encoder(x)
        s = self.encoder.stem_stride
        p_ds = pad[:, :h.shape[-1] * s].reshape(x.shape[0], -1, s).any(-1)
        w = p_ds.unsqueeze(1).float()
        pooled = (h * w).sum(-1) / w.sum(-1).clamp(min=1)
        return self.fc(pooled), pooled


def load_adapted_model(state: dict) -> PooledClsModel:
    """Rebuild an adapted model from a stage-5 checkpoint dict (CPU, eval)."""
    cfg = state["config"]
    pre = torch.load(Path(cfg["pretrain_ckpt"]), map_location="cpu",
                     weights_only=False)["config"]
    enc = build_encoder(pre, 0.0)
    if cfg["lora_top_blocks"] > 0:
        apply_lora(enc, cfg["lora_top_blocks"], cfg["lora_r"], cfg["lora_alpha"])
    model = PooledClsModel(enc)
    model.load_state_dict(state["model"])
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: int | None = None):
    model.eval()
    ys, ss, gs = [], [], []
    for i, (x, pad, y, g) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, pad = x.to(device), pad.to(device)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            logits = model(x, pad)
        ss.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
    model.train()
    y, s, g = (np.concatenate(v) for v in (ys, ss, gs))
    return {"macro_auroc": masked_macro_auroc(y, s),
            "worst_group_auroc": worst_group_auroc(y, s, g)}


def run(cfg: dict, repo: Path) -> Path:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)
    out = repo / "results/stage5" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    cfg["git_hash"] = git_hash(repo)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # --- encoder from the SAME fold's pretraining ----------------------------
    ck = torch.load(Path(cfg["pretrain_ckpt"]), map_location="cpu",
                    weights_only=False)
    pc = ck["config"]
    assert pc["held_out"] == cfg["held_out"], \
        f"pretrain ckpt fold ({pc['held_out']}) != adapt fold ({cfg['held_out']})"
    enc = build_encoder(pc, cfg["dropout"])
    enc_state = {k[len("encoder."):]: v for k, v in ck["model"].items()
                 if k.startswith("encoder.")}
    enc.load_state_dict(enc_state)

    if cfg["lora_top_blocks"] > 0:
        # Runtime self-check: LoRA must be an exact identity at init on the
        # device that will train (an unreproduced CPU unit-test flake was
        # seen once on Windows). fp32, no autocast, fixed probe input.
        enc.to(device).eval()
        g = torch.Generator(device="cpu").manual_seed(1234)
        probe = torch.randn(2, 12, cfg["crop_len"], generator=g).to(device)
        with torch.no_grad():
            before = enc(probe)
        apply_lora(enc, cfg["lora_top_blocks"], cfg["lora_r"], cfg["lora_alpha"])
        enc.to(device).eval()
        with torch.no_grad():
            after = enc(probe)
        diff = (after - before).abs().max().item()
        cfg["lora_init_identity_maxdiff"] = diff
        (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        print(f"LoRA init identity check: max|after-before| = {diff:.2e}",
              flush=True)
        if diff > 1e-4:
            raise RuntimeError(
                f"LoRA is not identity at init (max diff {diff:.3e}); "
                "aborting before training. Report this output.")
        enc.train()
    else:  # head-only ablation
        for p in enc.parameters():
            p.requires_grad_(False)
    model = PooledClsModel(enc).to(device)
    print(f"trainable fraction: {trainable_fraction(model):.4f}", flush=True)

    train_dl, val_dl = build_loaders(cfg, repo)
    dro = GroupDRO(n_groups=8, eta=cfg["dro_eta"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                total_steps=total, pct_start=0.05)
    # adapt has no resume: a re-run restarts from scratch, so the log is
    # overwritten (appending produced duplicated epoch rows on re-runs)
    log_path = out / "log.csv"
    log_path.write_text(
        "epoch,train_loss,val_macro_auroc,val_worst_group_auroc,sec\n",
        encoding="utf-8")

    best = -1.0
    for epoch in range(cfg["epochs"]):
        t0, running = time.time(), []
        for x, pad, y, g in train_dl:
            x, pad = x.to(device, non_blocking=True), pad.to(device, non_blocking=True)
            y, g = y.to(device), g.to(device)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                logits, pooled = model.forward_with_features(x, pad)
                ps = per_sample_masked_bce(logits.float(), y)
                loss = (1 - cfg["w_dro"]) * ps.mean()
                if cfg["w_dro"] > 0:
                    loss = loss + cfg["w_dro"] * dro(ps, g)
                if cfg["w_con"] > 0:
                    loss = loss + cfg["w_con"] * fair_supcon(
                        pooled.float(), y, g,
                        group_weight=cfg["con_group_weight"])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            sched.step()
            running.append(loss.item())
        m = evaluate(model, val_dl, device, amp_dtype,
                     max_batches=cfg.get("val_max_batches"))
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(float(np.mean(running)), 5),
                                    round(m["macro_auroc"], 5),
                                    round(m["worst_group_auroc"], 5),
                                    round(time.time() - t0)])
        state = {"model": model.state_dict(), "epoch": epoch,
                 "val": m, "config": cfg, "dro_q": dro.q.cpu()}
        torch.save(state, out / "ckpt_last.pt")
        if m["macro_auroc"] > best:
            best = m["macro_auroc"]
            torch.save(state, out / "ckpt_best.pt")
        print(f"epoch {epoch}: loss {np.mean(running):.4f} "
              f"val macro-AUROC {m['macro_auroc']:.4f} "
              f"worst-group {m['worst_group_auroc']:.4f}", flush=True)
    return out
