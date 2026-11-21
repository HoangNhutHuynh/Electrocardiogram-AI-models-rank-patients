"""Export tidy CSV source data for the manuscript figures (read by MATLAB).

Prerequisites (all already produced by the pipeline):
  stage6 metrics.json + predictions.npz, stage7 h1_verdict.json (re-run
  stage7 once so it contains ci95_per_fold), code_test summary + predictions,
  exploratory/{label_shift,af_afl,local_recalibration}.
Output: results/figure_data/fig{2..7}_*.csv  (one file per panel)
Usage: python scripts/export_figure_data.py [repo_root]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ect", HERE / "eval_code_test.py")
ect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ect)
LABELS = ["AF", "1dAVb", "RBBB", "LBBB", "SB", "STach"]
GROUPS = ["M <40", "M 40-60", "M 60-75", "M >=75", "F <40", "F 40-60", "F 60-75", "F >=75"]
FOLD_NAME = {"ptbxl": "PTB-XL", "chapman": "Chapman", "code15": "CODE-15%"}
RES, OUT = ROOT / "results", ROOT / "results/figure_data"
OUT.mkdir(parents=True, exist_ok=True)
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def variant(run: str) -> str:
    v = "ERM" if "w_dro0_w_con0" in run else (
        "DRO+SupCon" if "w_con0.05" in run else "DRO")
    # architecture control: a separate variant, ignored by make_figures.m (VAR)
    return v + " (resnet)" if "_resnet" in run else v


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)
    print(f"{name:28s} {len(df):5d} rows")


# ---- Fig. 2 / Fig. 4: per-run metrics -------------------------------------
lodo, perclass, conf, perclass_all, wg = [], [], [], [], []
for mp in sorted((RES / "stage6").glob("*/metrics.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    run, t = m["run_name"], m["test"]
    v, fold = variant(run), FOLD_NAME[m["held_out"]]
    lodo.append({"fold": fold, "variant": v, "seed": m["seed"],
                 "val_macro_auroc": m["val_source"]["macro_auroc"],
                 "test_macro_auroc": t["macro_auroc"],
                 "test_worst_group_auroc": t["worst_group_auroc"]})
    wg.append({"run": run, "fold": fold, "variant": v, "seed": m["seed"],
               "test_worst_group": t["worst_group_auroc"]})
    for lab, d in t["per_class"].items():
        perclass_all.append({"fold": fold, "variant": v, "label": lab, "seed": m["seed"],
                             "auroc": d["auroc"]})
    if v == "ERM":
        for lab, d in t["per_class"].items():
            perclass.append({"fold": fold, "label": lab, "seed": m["seed"],
                             "auroc": d["auroc"], "f1_at_0.5": d["f1_at_0.5"],
                             "prevalence": d["prevalence"]})
    for g, d in t["conformal_label_conditional"]["per_group"].items():
        conf.append({"fold": fold, "variant": v, "seed": m["seed"],
                     "group_id": int(g), "group": GROUPS[int(g)],
                     "coverage": d["coverage"], "positive_coverage": d["positive_coverage"],
                     "n_pos_cells": d["n_pos_cells"]})
save(pd.DataFrame(lodo), "fig2a_lodo.csv")
save(pd.DataFrame(perclass), "fig2b_perclass_auroc.csv")
save(pd.DataFrame(conf), "fig4_conformal_groups.csv")
save(pd.DataFrame(perclass_all), "figS1_perclass_variants.csv")
wg = [w for w in wg if "(resnet)" not in w["variant"]]  # Fig. S2: state-space encoder only
_val = pd.read_csv(RES / "stage7/runs_stage5_raw.csv")[["run", "val_worst_group"]]
_wg = pd.DataFrame(wg).merge(_val, on="run", validate="one_to_one")
assert len(_wg) == len(wg) == 27, "expected 9 runs x 3 folds with a validation score each"
save(_wg, "figS2_worst_group_val_test.csv")

# ---- Fig. 3: H1 forest -----------------------------------------------------
h1 = json.loads((RES / "stage7/h1_verdict.json").read_text(encoding="utf-8"))
rows = []
for v, d in h1.items():
    if "ci95_per_fold" not in d:
        sys.exit("h1_verdict.json lacks ci95_per_fold: re-run scripts/stage7_tables.py")
    # each variant is compared with ERM of the same encoder (H1 for state-space, E4b for convolutional)
    enc = "Convolutional" if "(resnet)" in v else "State-space"
    base = "DRO+SupCon" if "SupCon" in v else "DRO"
    for f in d["folds"]:
        rows.append({"encoder": enc, "variant": base, "fold": FOLD_NAME[f],
                     "delta": d["delta_per_fold"][f], "lo": d["ci95_per_fold"][f][0],
                     "hi": d["ci95_per_fold"][f][1]})
    rows.append({"encoder": enc, "variant": base, "fold": "Pooled",
                 "delta": d["pooled_delta"], "lo": d["pooled_ci95"][0],
                 "hi": d["pooled_ci95"][1]})
assert len(rows) == 16, "expected 2 encoders x 2 variants x (3 folds + pooled)"
save(pd.DataFrame(rows), "fig3_h1_forest.csv")

# ---- Fig. 5a: prevalence by cohort (Chapman split by hospital) ------------
lab = pd.read_csv(RES / "labels_all.csv", dtype={"record_id": str}, low_memory=False)
lab["site"] = lab["cohort"].map({"ptbxl": "PTB-XL", "code15": "CODE-15%",
                                 "code_test": "CODE-test", "chapman": "Chapman"})
ch = lab["cohort"] == "chapman"
lab.loc[ch, "site"] = np.where(lab.loc[ch, "record_id"].str[2:].astype(int) <= 10646,
                               "Shaoxing", "Ningbo")
prev = lab.groupby("site")[LABELS].mean().reset_index().melt(
    id_vars="site", var_name="label", value_name="prevalence")
dcfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
cinc = Path(dcfg["data_root"]) / "CinC 2021 SNOMED mapping/dx_mapping_scored.csv"
if not cinc.exists():
    cinc = ROOT / "dx_mapping_scored.csv"  # copy next to repo when run elsewhere
afl = pd.read_csv(cinc).drop_duplicates("SNOMEDCTCode").set_index("Abbreviation").loc["AFL"]
n_site = lab.groupby("site").size()
for site, col in [("Shaoxing", "Chapman_Shaoxing"), ("Ningbo", "Ningbo")]:
    # shortcut: CinC counts are pre-QC; QC removed <0.3% of Chapman records
    prev = pd.concat([prev, pd.DataFrame([{"site": site, "label": "AFL (CinC count)",
                                           "prevalence": afl[col] / n_site[site]}])])
save(prev, "fig5a_prevalence.csv")

# ---- Fig. 5b/c: AF re-scoring and EM -----------------------------------------
save(pd.read_csv(RES / "exploratory/af_afl/summary.csv"), "fig5b_af_afl.csv")
em = pd.read_csv(RES / "exploratory/label_shift/per_label.csv")
em = em.groupby(["fold", "label"])[["prev_true", "prev_em", "prev_source"]].mean().reset_index()
em["fold"] = em["fold"].map(FOLD_NAME)
save(em, "fig5c_em_prevalence.csv")

# ---- Fig. 6: PR curves on CODE-test + reader operating points ----------------
ann = Path(dcfg["data_root"]) / dcfg["cohorts"]["code_test"]["dir"] / "annotations"
if not ann.exists():
    ann = ROOT / "annotations"
y = ect.read_annot(ann / "gold_standard.csv")
summ = json.loads((RES / "code_test/summary.json").read_text(encoding="utf-8"))
grid = np.linspace(0, 1, 201)
curves, ops = [], []
for fold, role in [("code15", "External (sources PTB-XL+Chapman)"),
                   ("chapman", "In-population (sources incl. CODE-15%)")]:
    runs = sorted(r for r in summ["per_model"] if f"heldout-{fold}_" in r)
    for c, labn in enumerate(LABELS):
        prec_seeds, op_p, op_r = [], [], []
        for r in runs:
            T = float(np.load(RES / "stage6" / r / "predictions.npz")["temperature"])
            p = sig(np.load(RES / "code_test" / r / "predictions.npz")["logits"][:, c] / T)
            pr, rc, _ = precision_recall_curve(y[:, c], p)
            # interpolated precision: max precision at recall >= r
            prec_seeds.append([pr[rc >= g].max() for g in grid])
            yh = p >= summ["per_model"][r]["thresholds"][c]
            tp = float((yh & (y[:, c] == 1)).sum())
            op_p.append(tp / max(yh.sum(), 1))
            op_r.append(tp / y[:, c].sum())
        mp_ = np.mean(prec_seeds, 0)
        curves += [{"model": role, "label": labn, "recall": g, "precision": q}
                   for g, q in zip(grid, mp_)]
        ops.append({"who": f"Model {role.split(' ')[0].lower()} (source threshold)",
                    "label": labn, "precision": np.mean(op_p), "recall": np.mean(op_r)})
for rn in ect.READERS:
    a = ect.read_annot(ann / f"{rn}.csv")
    for c, labn in enumerate(LABELS):
        tp = float(((a[:, c] == 1) & (y[:, c] == 1)).sum())
        ops.append({"who": rn, "label": labn, "precision": tp / max(a[:, c].sum(), 1),
                    "recall": tp / y[:, c].sum()})
save(pd.DataFrame(curves), "fig6_pr_curves.csv")
save(pd.DataFrame(ops), "fig6_operating_points.csv")

# ---- Fig. 7: local recalibration ---------------------------------------------
save(pd.read_csv(RES / "exploratory/local_recalibration/summary.csv"), "fig7_local_recal.csv")
print(f"-> {OUT}")
