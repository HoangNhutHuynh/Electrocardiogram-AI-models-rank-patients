"""EXPLORATORY (not pre-registered; motivated by the chapman results):
does EM prior-shift adaptation on the unlabeled target cohort repair
calibration, F1@0.5 and label-conditional conformal coverage?

Diagnostic logic: if EM recovers the true target prevalence and coverage
recovers, the shift is consistent with pure prior shift; if EM misses the
true prevalence, p(x|y) itself moved (e.g. label-definition shift).

Model: ERM (val-selected), 3 seeds per fold, from results/stage6/*/predictions.npz.
Output: results/exploratory/label_shift/{per_label.csv, summary.csv}
Usage: python scripts/explore_label_shift.py [repo_root]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calm_ecg.data.harmonize import LABELS  # noqa: E402
from calm_ecg.eval.conformal import LabelConditionalMondrian  # noqa: E402
from calm_ecg.eval.label_shift import adapt_multilabel  # noqa: E402
from calm_ecg.eval.metrics import expected_calibration_error  # noqa: E402

sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
rows, summ = [], []
for npz in sorted((ROOT / "results/stage6").glob("adapt_heldout-*_w_dro0_w_con0/predictions.npz")):
    z = np.load(npz)
    if "val_logits" not in z:
        print(f"skip {npz.parent.name}: no val predictions (re-run stage6)")
        continue
    fold = npz.parent.name.split("_")[1].split("-")[1]
    seed = int(npz.parent.name.split("_seed")[1][0])
    T = float(z["temperature"])
    pv, pt = sig(z["val_logits"] / T), sig(z["test_logits"] / T)
    yv, yt, gv, gt = z["val_y"], z["test_y"], z["val_g"], z["test_g"]
    pi_s = np.nanmean(yv, axis=0)
    pa, pi_em, iters = adapt_multilabel(pt, pi_s)

    conf = LabelConditionalMondrian(alpha=0.1).calibrate(pv, yv, gv)
    cb, ca = conf.coverage(pt, yt, gt), conf.coverage(pa, yt, gt)
    summ.append({"fold": fold, "seed": seed,
                 "ece_before": expected_calibration_error(yt, pt),
                 "ece_after": expected_calibration_error(yt, pa),
                 "pos_cov_before": cb["positive_coverage"], "pos_cov_after": ca["positive_coverage"],
                 "neg_cov_before": cb["negative_coverage"], "neg_cov_after": ca["negative_coverage"],
                 "empty_before": cb["empty_set_rate"], "empty_after": ca["empty_set_rate"]})
    for c, lab in enumerate(LABELS):
        v = ~np.isnan(yt[:, c])
        y = yt[v, c]
        auc_b, auc_a = roc_auc_score(y, pt[v, c]), roc_auc_score(y, pa[v, c])
        assert abs(auc_b - auc_a) < 1e-9, "prior adjustment must not change ranks"
        rows.append({"fold": fold, "seed": seed, "label": lab,
                     "prev_source": pi_s[c], "prev_true": y.mean(), "prev_em": pi_em[c],
                     "em_iters": iters[c], "auroc": auc_b,
                     "f1_before": f1_score(y, pt[v, c] >= 0.5),
                     "f1_after": f1_score(y, pa[v, c] >= 0.5)})

out = ROOT / "results/exploratory/label_shift"
out.mkdir(parents=True, exist_ok=True)
per = pd.DataFrame(rows)
s = pd.DataFrame(summ)
per.to_csv(out / "per_label.csv", index=False)
s.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 200)
print(per.groupby(["fold", "label"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print()
print(s.groupby("fold").mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print(f"\n-> {out}")
