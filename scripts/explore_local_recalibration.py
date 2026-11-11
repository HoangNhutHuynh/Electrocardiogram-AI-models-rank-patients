"""E3 (EXPLORATORY): how many locally labeled ECGs does a new site need to
re-select decision thresholds and recover F1?

For each ERM model and k in K: R random splits of CODE-test into k labeled
"local" ECGs (threshold re-selection, best F1 per label) and the rest
(evaluation). A label with < MIN_POS positives among the k keeps its
source-val threshold. Readers are scored on the SAME evaluation records.
'oracle' = thresholds tuned on all of CODE-test (optimistic upper bound).
Uses cached predictions only (no GPU).
Output: results/exploratory/local_recalibration/{summary.csv, splits.csv}
Usage: python scripts/explore_local_recalibration.py [repo_root]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ect", Path(__file__).resolve().parent / "eval_code_test.py")
ect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ect)

K, R, MIN_POS, SEED = (25, 50, 100, 200), 200, 3, 0
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def local_thresholds(y_cal, p_cal, tau_src, min_pos=MIN_POS):
    """Per label: best-F1 threshold on the local sample, else keep source."""
    tau = tau_src.copy()
    for c in range(y_cal.shape[1]):
        if y_cal[:, c].sum() >= min_pos:
            tau[c] = ect.best_f1_threshold(y_cal[:, c], p_cal[:, c])
    return tau


def main() -> int:
    dcfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    ann = Path(dcfg["data_root"]) / dcfg["cohorts"]["code_test"]["dir"] / "annotations"
    if not ann.exists():  # running on a copy: annotations staged next to results
        ann = ROOT / "annotations"
    y = ect.read_annot(ann / "gold_standard.csv")
    readers = {r: ect.read_annot(ann / f"{r}.csv") for r in ect.READERS}

    models = {}
    for run in sorted((ROOT / "results/code_test").glob("adapt_heldout-*_w_dro0_w_con0")):
        z6 = np.load(ROOT / "results/stage6" / run.name / "predictions.npz")
        T = float(z6["temperature"])
        pv, yv = sig(z6["val_logits"] / T), z6["val_y"]
        tau_src = np.array([ect.best_f1_threshold(yv[:, c], pv[:, c]) for c in range(6)])
        p = sig(np.load(run / "predictions.npz")["logits"] / T)
        fold = run.name.split("_")[1].split("-")[1]
        models[run.name] = (fold, p, tau_src)

    rng = np.random.default_rng(SEED)
    n = len(y)
    rows = []
    for k in K:
        for r in range(R):
            perm = rng.permutation(n)
            cal, ev = perm[:k], perm[k:]
            rec = {"k": k, "split": r}
            for name, (fold, p, tau_src) in models.items():
                tau = local_thresholds(y[cal], p[cal], tau_src)
                rec[f"{fold}|{name}|source"] = ect.macro_f1(y, (p >= tau_src).astype(float), ev)
                rec[f"{fold}|{name}|local"] = ect.macro_f1(y, (p >= tau).astype(float), ev)
            for rn, a in readers.items():
                rec[f"reader|{rn}"] = ect.macro_f1(y, a, ev)
            rows.append(rec)
    df = pd.DataFrame(rows)

    # oracle (in-sample, optimistic) per model
    oracle = {name: ect.macro_f1(y, (p >= np.array(
        [ect.best_f1_threshold(y[:, c], p[:, c]) for c in range(6)])).astype(float))
        for name, (_, p, _) in models.items()}

    summ = []
    for k, g in df.groupby("k"):
        for fold in sorted({f for f, _, _ in models.values()}):
            for kind in ("source", "local"):
                cols = [c for c in g.columns if c.startswith(f"{fold}|") and c.endswith(f"|{kind}")]
                v = g[cols].mean(axis=1)  # mean over the fold's 3 seeds, per split
                summ.append({"k": k, "who": f"ERM {fold} ({kind} thresholds)",
                             "macro_f1_mean": v.mean(), "p2.5": v.quantile(.025),
                             "p97.5": v.quantile(.975)})
        for rn in readers:
            v = g[f"reader|{rn}"]
            summ.append({"k": k, "who": rn, "macro_f1_mean": v.mean(),
                         "p2.5": v.quantile(.025), "p97.5": v.quantile(.975)})
    s = pd.DataFrame(summ)
    for fold in sorted({f for f, _, _ in models.values()}):
        vals = [oracle[nm] for nm, (f, _, _) in models.items() if f == fold]
        s = pd.concat([s, pd.DataFrame([{"k": "all", "who": f"ERM {fold} (ORACLE, in-sample)",
                                         "macro_f1_mean": float(np.mean(vals))}])])

    out = ROOT / "results/exploratory/local_recalibration"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "splits.csv", index=False)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
