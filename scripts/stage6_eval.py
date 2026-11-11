"""Stage 6 — ONE-TIME evaluation of one adapted model on its held-out cohort.

Protocol (quy trình): the held-out test cohort is evaluated exactly once per
final model, AFTER all choices were locked on source-val. Every invocation is
appended to results/stage6/EVAL_LEDGER.csv with a timestamp — repeated peeks
at the same test cohort are visible, not hidden.

Pipeline per run:
  1. load adapt ckpt_best; re-assert fold consistency + leakage pre-flight
  2. predict on source VAL (calibration data) and held-out TEST
  3. temperature scaling fitted on val; ECE pre/post on test
  4. Mondrian conformal (sex x age-bin) calibrated on val; coverage on test
  5. macro-AUROC + per-class AUROC/F1@0.5, worst-group AUROC,
     bootstrap 1000x 95% CI for macro-AUROC
  6. write results/stage6/<run_name>/metrics.json (figures read ONLY this)

Usage: python scripts/stage6_eval.py results/stage5/<run_name>/ckpt_best.pt

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.dataset import EcgDataset  # noqa: E402
from calm_ecg.data.splits import assert_no_leakage  # noqa: E402
from calm_ecg.eval.conformal import (LabelConditionalMondrian,  # noqa: E402
                                     MondrianConformal, fit_temperature)
from calm_ecg.eval.metrics import (expected_calibration_error,  # noqa: E402
                                   masked_macro_auroc, per_group_auroc,
                                   worst_group_auroc)
from calm_ecg.train.adapt import load_adapted_model  # noqa: E402


@torch.no_grad()
def predict(model, ds, device, amp_dtype, batch_size=64, num_workers=4):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=True)
    logits, ys, gs = [], [], []
    for x, pad, y, g in dl:
        x, pad = x.to(device), pad.to(device)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            out = model(x, pad)
        logits.append(out.float().cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
    return (np.concatenate(logits), np.concatenate(ys), np.concatenate(gs))


def bootstrap_ci(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(masked_macro_auroc(y[idx], s[idx]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main() -> int:
    ckpt_path = Path(sys.argv[1])
    repo = Path(__file__).resolve().parents[1]
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)

    # ledger: every test-set evaluation is recorded, first or repeated
    ledger = repo / "results/stage6/EVAL_LEDGER.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    first = not ledger.exists()
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["utc", "run_name", "held_out", "ckpt"])
        w.writerow([datetime.now(timezone.utc).isoformat(),
                    cfg["run_name"], cfg["held_out"], str(ckpt_path)])

    # pre-flight leakage audit on the committed splits
    labels = pd.read_csv(repo / "results/labels_all.csv",
                         dtype={"record_id": str, "patient_id": str})
    splits = json.loads((repo / "results/splits_lodo.json")
                        .read_text(encoding="utf-8"))
    assert_no_leakage(splits, labels)

    model = load_adapted_model(state).to(device).eval()

    common = dict(repo_root=repo, data_root=Path(cfg["data_root"]),
                  cohort_dirs=cfg["cohort_dirs"], held_out=cfg["held_out"],
                  crop_len=cfg["crop_len"], seed=0, train=False)
    t0 = time.time()
    val_logits, val_y, val_g = predict(
        model, EcgDataset(split="val", **common), device, amp_dtype)
    test_logits, test_y, test_g = predict(
        model, EcgDataset(split="test", **common), device, amp_dtype)
    print(f"inference done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # temperature from val; probs
    T = fit_temperature(val_logits, val_y)
    p_test = 1 / (1 + np.exp(-test_logits))
    p_test_T = 1 / (1 + np.exp(-test_logits / T))
    p_val_T = 1 / (1 + np.exp(-val_logits / T))

    # conformal calibrated on val (temperature-scaled probs).
    # Both variants reported: (g,c) = pre-registered original; (g,c,y) =
    # Amendment 1 (fixes empty sets / unguaranteed positive coverage).
    conf = MondrianConformal(alpha=0.1).calibrate(p_val_T, val_y, val_g)
    conf_lc = LabelConditionalMondrian(alpha=0.1).calibrate(p_val_T, val_y, val_g)

    from sklearn.metrics import f1_score
    per_class = {}
    from calm_ecg.data.harmonize import LABELS
    for c, lab in enumerate(LABELS):
        valid = ~np.isnan(test_y[:, c])
        yc = test_y[valid, c]
        if len(np.unique(yc)) < 2:
            per_class[lab] = None
            continue
        from sklearn.metrics import roc_auc_score
        per_class[lab] = {
            "auroc": round(float(roc_auc_score(yc, p_test[valid, c])), 4),
            "f1_at_0.5": round(float(
                f1_score(yc, p_test[valid, c] >= 0.5)), 4),
            "prevalence": round(float(yc.mean()), 5)}

    metrics = {
        "run_name": cfg["run_name"], "held_out": cfg["held_out"],
        "seed": cfg["seed"], "ckpt": str(ckpt_path),
        "temperature": round(T, 4),
        "test": {
            "n_records": int(len(test_y)),
            "macro_auroc": round(masked_macro_auroc(test_y, p_test), 4),
            "macro_auroc_ci95": bootstrap_ci(test_y, p_test),
            "per_class": per_class,
            "worst_group_auroc": round(
                worst_group_auroc(test_y, p_test, test_g), 4),
            "per_group_auroc": {k: round(v, 4) for k, v in
                                per_group_auroc(test_y, p_test, test_g).items()},
            "ece_raw": round(expected_calibration_error(test_y, p_test), 4),
            "ece_temp_scaled": round(
                expected_calibration_error(test_y, p_test_T), 4),
            "conformal": conf.coverage(p_test_T, test_y, test_g),
            "conformal_label_conditional": conf_lc.coverage(
                p_test_T, test_y, test_g),
        },
        "val_source": {
            "macro_auroc": round(masked_macro_auroc(
                val_y, 1 / (1 + np.exp(-val_logits))), 4),
        },
    }
    out = repo / "results/stage6" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                      encoding="utf-8")
    # val predictions saved too, so calibration/conformal can be recomputed
    # offline without re-running inference (and without new ledger entries)
    np.savez_compressed(out / "predictions.npz", test_logits=test_logits,
                        test_y=test_y, test_g=test_g, temperature=T,
                        val_logits=val_logits, val_y=val_y, val_g=val_g)
    print(json.dumps(metrics["test"], indent=2))
    print(f"\nmetrics -> {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
