"""Amendment 3: ERM models on CODE-test (gold labels) vs human readers.

Requires results/codetest_overlap.json (run check_codetest_overlap.py first)
and stage6 predictions.npz with val predictions (for thresholds).
Predictions cached in results/code_test/<run>/predictions.npz; every model
inference is appended to results/code_test/EVAL_LEDGER.csv.
Output: results/code_test/summary.json
Usage: python scripts/eval_code_test.py

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from calm_ecg.data.dataset import EcgDataset  # noqa: E402
from calm_ecg.data.harmonize import CODE_COL_MAP, LABELS  # noqa: E402
from calm_ecg.train.adapt import load_adapted_model  # noqa: E402

READERS = ["cardiology_residents", "emergency_residents", "medical_students", "dnn"]
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.argmax(f1)])


def read_annot(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # dnn.csv has an index column
    return np.stack([df[col].to_numpy(float) for col in
                     [c for c, lab in sorted(CODE_COL_MAP.items(),
                                             key=lambda kv: LABELS.index(kv[1]))]], 1)


def macro_f1(y, yhat, idx=None):
    if idx is not None:
        y, yhat = y[idx], yhat[idx]
    return float(np.mean([f1_score(y[:, c], yhat[:, c], zero_division=0)
                          for c in range(y.shape[1])]))


def predict_code_test(run: Path, labels_ids, device) -> np.ndarray:
    cache = REPO / "results/code_test" / run.name / "predictions.npz"
    if cache.exists():
        return np.load(cache)["logits"]
    state = torch.load(run / "ckpt_best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = load_adapted_model(state).to(device)
    ds = EcgDataset(repo_root=REPO, data_root=Path(cfg["data_root"]),
                    cohort_dirs=cfg["cohort_dirs"], split="test", held_out=cfg["held_out"],
                    crop_len=cfg["crop_len"], train=False, record_ids=labels_ids)
    assert len(ds) == len(labels_ids), "QC excluded CODE-test records; handle explicitly"
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(device)
            pad = torch.stack([ds[j][1] for j in range(i, min(i + 64, len(ds)))]).to(device)
            out.append(model(x, pad).float().cpu().numpy())
    logits = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits)
    ledger = REPO / "results/code_test/EVAL_LEDGER.csv"
    new = not ledger.exists()
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "run_name"])
        w.writerow([datetime.now(timezone.utc).isoformat(), run.name])
    return logits


def main() -> int:
    overlap = json.loads((REPO / "results/codetest_overlap.json").read_text(encoding="utf-8"))
    leaked = set(overlap["overlap_record_ids"])
    in_pop_allowed = overlap["fraction"] <= 0.05

    labels = pd.read_csv(REPO / "results/labels_all.csv", dtype={"record_id": str})
    ct = labels[labels["cohort"] == "code_test"].reset_index(drop=True)
    ids, y = ct["record_id"].tolist(), ct[LABELS].to_numpy(float)
    dcfg = yaml.safe_load((REPO / "configs/data.yaml").read_text(encoding="utf-8"))
    ann = Path(dcfg["data_root"]) / dcfg["cohorts"]["code_test"]["dir"] / "annotations"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = {"overlap": {k: overlap[k] for k in ("n_exact", "n_rounded", "fraction")},
           "per_model": {}}
    preds = {}  # fold -> list of binary predictions (827, 6) over seeds
    for run in sorted((REPO / "results/stage5").glob("adapt_heldout-*_w_dro0_w_con0")):
        fold = run.name.split("_")[1].split("-")[1]
        if fold != "code15" and not in_pop_allowed:
            continue
        z = np.load(REPO / "results/stage6" / run.name / "predictions.npz")
        T = float(z["temperature"])
        pv, yv = sig(z["val_logits"] / T), z["val_y"]
        tau = np.array([best_f1_threshold(yv[:, c], pv[:, c]) for c in range(len(LABELS))])
        p = sig(predict_code_test(run, ids, device) / T)
        keep = np.ones(len(ids), bool) if fold == "code15" else \
            np.array([i not in leaked for i in ids])
        yhat = (p >= tau).astype(float)
        res["per_model"][run.name] = {
            "fold": fold, "n_eval": int(keep.sum()), "thresholds": tau.round(4).tolist(),
            "auroc": {lab: round(float(roc_auc_score(y[keep, c], p[keep, c])), 4)
                      for c, lab in enumerate(LABELS) if 0 < y[keep, c].sum() < keep.sum()},
            "f1": {lab: round(float(f1_score(y[keep, c], yhat[keep, c], zero_division=0)), 4)
                   for c, lab in enumerate(LABELS)},
            "macro_f1": round(macro_f1(y[keep], yhat[keep]), 4)}
        preds.setdefault(fold, []).append((yhat, keep))

    gold_csv = read_annot(ann / "gold_standard.csv")
    assert np.array_equal(gold_csv, y), "harmonized CODE-test labels != gold_standard.csv"
    readers = {r: read_annot(ann / f"{r}.csv") for r in READERS}
    res["readers"] = {r: {"macro_f1": round(macro_f1(y, a), 4),
                          "f1": {lab: round(float(f1_score(y[:, c], a[:, c], zero_division=0)), 4)
                                 for c, lab in enumerate(LABELS)}} for r, a in readers.items()}
    c1, c2 = read_annot(ann / "cardiologist1.csv"), read_annot(ann / "cardiologist2.csv")
    res["inter_rater_cardiologist1_vs_2_macro_f1"] = round(macro_f1(c1, c2), 4)

    # paired bootstrap: model (mean over seeds of macro-F1) − reader, same records
    rng = np.random.default_rng(0)
    res["comparisons"] = {}
    for fold, lst in preds.items():
        keep = lst[0][1]
        idx_all = np.flatnonzero(keep)
        def model_f1(idx):
            return float(np.mean([macro_f1(y, yh, idx) for yh, _ in lst]))
        for r, a in readers.items():
            point = model_f1(idx_all) - macro_f1(y, a, idx_all)
            draws = []
            for _ in range(1000):
                b = rng.choice(idx_all, len(idx_all))
                draws.append(model_f1(b) - macro_f1(y, a, b))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res["comparisons"][f"{fold}_ERM_vs_{r}"] = {
                "delta_macro_f1": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "role": "external (primary)" if fold == "code15" else "in-population (secondary)"}

    out = REPO / "results/code_test/summary.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("overlap", "readers", "comparisons",
                                          "inter_rater_cardiologist1_vs_2_macro_f1")}, indent=2))
    for name, m in res["per_model"].items():
        print(name, "macro-F1", m["macro_f1"], "n", m["n_eval"])
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
