"""Write manuscript/supp_tables.tex from the result files (no number is typed by hand).

Run from the repo root:  python scripts/make_supplement_tables.py

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path("results")
OUT = pathlib.Path("manuscript/supp_tables.tex")
LABELS = ["AF", "1dAVb", "RBBB", "LBBB", "SB", "STach"]
FOLDS = {"ptbxl": "PTB-XL", "chapman": "Chapman", "code15": "CODE-15\\%"}
GROUPS = ["M $<$40", "M 40-59", "M 60-74", "M $\\geq$75",
          "F $<$40", "F 40-59", "F 60-74", "F $\\geq$75"]
out = []


def table(caption, label, header, rows, small=True, head=None):
    cols = "l" * 1 + "r" * (len(header) - 1)
    out.append("\\begin{table}[!htbp]\\centering" + ("\\footnotesize" if small else "")
               + "\\setlength{\\tabcolsep}{3pt}\n\\caption{%s}\\label{%s}\n" % (caption, label)
               + "\\begin{tabular}{@{}%s@{}}\\toprule\n" % cols
               + (head or " & ".join(header) + " \\\\") + "\\midrule\n"
               + "\n".join(" & ".join(str(c) for c in r) + " \\\\" for r in rows)
               + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


# ---- S1/S2: demographics and label prevalence after QC --------------------------------
d = pd.read_csv(R / "labels_all.csv", low_memory=False)
excl = set(pd.read_csv(R / "qc_exclusions.csv").record_id.astype(str))
d = d[~d.record_id.astype(str).isin(excl)].copy()
is_ch = d.cohort == "chapman"
js = pd.to_numeric(d.record_id.astype(str).str.replace("JS", ""), errors="coerce")
d["site"] = d.cohort.map({"ptbxl": "PTB-XL", "code15": "CODE-15\\%", "code_test": "CODE-test"})
d.loc[is_ch, "site"] = np.where(js[is_ch] <= 10646, "Shaoxing", "Ningbo")
SITES = ["PTB-XL", "Shaoxing", "Ningbo", "CODE-15\\%", "CODE-test"]
age = d.age.where(d.age <= 120)  # PTB-XL stores ages above 89 as 300
ok = d.sex.isin(["M", "F"]) & d.age.notna()
d["group"] = np.where(ok, np.digitize(d.age.fillna(0), [40, 60, 75]) + 4 * (d.sex == "F"), -1)

rows = []
for s in SITES:
    m = d.site == s
    a = age[m].dropna()
    q = a.quantile([.25, .5, .75]).round().astype(int).tolist()
    rows.append([s, f"{m.sum():,}", f"{q[1]} [{q[0]}, {q[2]}]", pct((d.sex[m] == "F").mean())]
                + [f"{(d.group[m] == g).sum():,}" for g in range(8)] + [f"{(d.group[m] == -1).sum():,}"])
table("Demographics per site after quality control. Age: median [IQR] in years (PTB-XL ages above 89 "
      "are censored in the source and excluded from the median). Columns M/F: number of records per "
      "sex $\\times$ age group; n/a: missing age or sex.", "tab:s-demo",
      ["Site", "Records", "Age", "Female \\%"] + GROUPS + ["n/a"], rows)

rows = [[s, f"{(d.site == s).sum():,}"] + [pct(np.nanmean(d.loc[d.site == s, l])) for l in LABELS]
        for s in SITES]
table("Prevalence (\\%) of the six harmonized labels per site after quality control.", "tab:s-prev",
      ["Site", "Records"] + LABELS, rows)

# ---- S3/S4: per-label metrics and conformal detail, all variants ----------------------
raw = pd.read_csv(R / "stage7" / "runs_stage6_raw.csv")
runs = {r.run_name: (r.held_out, r.variant) for r in raw.itertuples()}
M = {}
for p in glob.glob(str(R / "stage6" / "*" / "metrics.json")):
    m = json.load(open(p))
    if m["run_name"] in runs:
        M.setdefault(runs[m["run_name"]], []).append(m["test"])
variants = sorted({v for _, v in M}, key=lambda v: (v != "ERM", v))
rows = []
for f in FOLDS:
    for v in variants:
        t = M[(f, v)]
        rows.append([FOLDS[f], v.replace("0.05", "")]
                    + [f3(np.mean([x["per_class"][l]["auroc"] for x in t])) + " / "
                       + f"{np.mean([x['per_class'][l]['f1_at_0.5'] for x in t]):.2f}" for l in LABELS])
table("Held-out AUROC / F1 at a threshold of 0.5 per label (mean of three seeds).", "tab:s-perlabel",
      ["Held out", "Objective"] + LABELS, rows)

rows = []
for f in FOLDS:
    for v in variants:
        t = M[(f, v)]
        lc, oc = [x["conformal_label_conditional"] for x in t], [x["conformal"] for x in t]
        mean = lambda xs, k: f3(float(np.mean([x[k] for x in xs])))
        wg = float(np.mean([min(g["coverage"] for g in x["per_group"].values()) for x in lc]))
        wgp = float(np.mean([min(g["positive_coverage"] for g in x["per_group"].values()
                                 if g["positive_coverage"] is not None) for x in lc]))
        rows.append([FOLDS[f], v.replace("0.05", ""), mean(lc, "marginal_coverage"),
                     mean(lc, "positive_coverage"), mean(lc, "negative_coverage"), f3(wg), f3(wgp),
                     mean(lc, "empty_set_rate"), mean(oc, "marginal_coverage"), mean(oc, "empty_set_rate")])
table("Conformal prediction on held-out cohorts ($\\alpha=0.1$, mean of three seeds). LC: label-conditional "
      "calibration (Amendment 1); Orig.: original calibration by group and label. Worst: lowest value "
      "over the eight groups.", "tab:s-conformal",
      ["Held out", "Objective", "LC all", "LC pos.", "LC neg.", "Worst all", "Worst pos.", "LC empty",
       "Orig. all", "Orig. empty"], rows)

# ---- S5: validation scores incl. single-seed ablations --------------------------------
a = pd.read_csv(R / "stage7" / "ablation_val.csv")
VN = {"DRO+SupCon0.05": "DRO+SupCon", "DRO+SupCon0.2": "DRO+SupCon ($w=0.2$)",
      "SupCon0.2": "SupCon only ($w=0.2$)", "head-only": "Linear head only"}
a["o"] = a.held_out.map(list(FOLDS).index)
a["v"] = a.variant.map(lambda v: (["ERM", "DRO", "DRO+SupCon0.05"] + list(VN)).index(v))
rows = [[FOLDS[r.held_out], VN.get(r.variant, r.variant), int(r.val_macro_auroc_count),
         f"{r.val_macro_auroc_mean:.4f}", f"{r.val_worst_group_mean:.4f}"]
        for r in a.sort_values(["o", "v"]).itertuples()]
table("Source-validation macro-AUROC and worst-group AUROC of all trained variants (mean over the "
      "given number of seeds). Single-seed rows are ablations run on the PTB-XL fold only; "
      "$w$ is the weight of the contrastive term.",
      "tab:s-val", ["Held out", "Variant", "Seeds", "Macro-AUROC", "Worst group"], rows)

# ---- S6: EM prior-shift (E1) -----------------------------------------------------------
e = pd.read_csv(R / "exploratory" / "label_shift" / "per_label.csv").groupby(["fold", "label"]).mean(numeric_only=True)
rows = [[FOLDS[f], l] + [pct(e.loc[(f, l), c]) for c in ["prev_source", "prev_true", "prev_em"]]
        + [f"{e.loc[(f, l), c]:.2f}" for c in ["f1_before", "f1_after"]] for f in FOLDS for l in LABELS]
table("Exploratory analysis E1: prevalence (\\%) in the source cohorts, true prevalence in the held-out "
      "cohort, EM estimate, and F1 at 0.5 before and after posterior adjustment (ERM, mean of three seeds).",
      "tab:s-em", ["Held out", "Label", "Source", "True", "EM", "F1 before", "F1 after"], rows)
s = pd.read_csv(R / "exploratory" / "label_shift" / "summary.csv").groupby("fold").mean(numeric_only=True)
rows = [[FOLDS[f]] + [f3(s.loc[f, c + "_" + w]) for c in ["ece", "pos_cov", "neg_cov", "empty"]
                      for w in ["before", "after"]] for f in FOLDS]
table("Exploratory analysis E1: calibration and label-conditional conformal coverage before / after the "
      "EM adjustment (ERM, mean of three seeds).", "tab:s-em2",
      ["Held out"] + [""] * 8, rows,
      head="& \\multicolumn{2}{c}{ECE} & \\multicolumn{2}{c}{Pos. coverage} & \\multicolumn{2}{c}{Neg. coverage} "
           "& \\multicolumn{2}{c}{Empty sets} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
           "Held out" + " & Before & After" * 4 + " \\\\")

# ---- S7: CODE-test per label ----------------------------------------------------------
c = json.load(open(R / "code_test" / "summary.json"))
by = {}
for m in c["per_model"].values():
    by.setdefault(m["fold"], []).append(m)
RD = {"cardiology_residents": "Cardiology residents", "emergency_residents": "Emergency residents",
      "medical_students": "Medical students", "dnn": "DNN of Ribeiro \\emph{et al.}"}
ORDER = ["code15", "chapman", "ptbxl"]
name = {"code15": "External (PTB-XL, Chapman)", "chapman": "In-pop. (PTB-XL, CODE-15\\%)",
        "ptbxl": "In-pop. (Chapman, CODE-15\\%)"}
rows = [[RD[k], f3(v["macro_f1"])] + [f"{v['f1'][l]:.2f}" for l in LABELS]
        for k, v in c["readers"].items()]
rows += [[name[f] + ", F1", f3(np.mean([m["macro_f1"] for m in ms]))]
         + [f"{np.mean([m['f1'][l] for m in ms]):.2f}" for l in LABELS] for f in ORDER for ms in [by[f]]]
rows += [[name[f] + ", AUROC", ""] + [f3(np.mean([m["auroc"][l] for m in ms])) for l in LABELS]
         for f in ORDER for ms in [by[f]]]
rows.append(["Cardiologist 1 vs. 2", f3(c["inter_rater_cardiologist1_vs_2_macro_f1"])] + [""] * 6)
table("CODE-test (827 ECGs): macro-F1 and per-label F1 of readers and models, and per-label AUROC of the "
      "models (ERM, mean of three seeds, thresholds from source validation).", "tab:s-codetest",
      ["Reader or model", "Macro-F1"] + LABELS, rows)
rows = [[name[k.split("_")[0]], RD[k.split("_vs_")[1]], f"${v['delta_macro_f1']:+.3f}$",
         f"$[{v['ci95'][0]:+.3f},\\ {v['ci95'][1]:+.3f}]$"]
        for f in ORDER for k, v in c["comparisons"].items() if k.startswith(f + "_")]
table("CODE-test: difference in macro-F1 between each model and each reader, with 95\\% paired bootstrap "
      "confidence interval over the 827 ECGs.", "tab:s-codetest-delta",
      ["Model", "Reader", "$\\Delta$", "95\\% CI"], rows)

# ---- S8: local threshold reselection (E3) ---------------------------------------------
l = pd.read_csv(R / "exploratory" / "local_recalibration" / "summary.csv")
rows = [[r.k, r.who.replace("code15", "external").replace("chapman", "in-pop. A").replace("ptbxl", "in-pop. B")
         .replace("_", " ").replace("dnn", "DNN"), f3(r.macro_f1_mean), "--" if pd.isna(r._4) else f"[{r._4:.3f}, {r._5:.3f}]"]
        for r in l.itertuples()]
table("Exploratory analysis E3: macro-F1 on the remaining CODE-test records after re-selecting thresholds on "
      "$k$ local ECGs (mean and 2.5th to 97.5th percentile over 200 random splits). In-pop. A: sources PTB-XL "
      "and CODE-15\\%; in-pop. B: sources Chapman and CODE-15\\%. Readers are scored on the same records.",
      "tab:s-local", ["$k$", "Reader or model", "Macro-F1", "Percentiles"], rows)

# ---- S9: compute ----------------------------------------------------------------------
rows = []
for stage, pat in [("Pretraining (20 epochs)", "stage4/*/log.csv"), ("Adaptation (10 epochs)", "stage5/*/log.csv")]:
    for f in FOLDS:
        # sec is the duration of each epoch
        h = [pd.read_csv(p).sec.sum() / 3600 for p in glob.glob(str(R / pat)) if f"heldout-{f}_" in p]
        rows.append([stage, FOLDS[f], len(h), f"{np.mean(h):.1f}"])
table("Wall-clock training time per run on one 8-GB GPU (mean over the listed runs; sum of the per-epoch times "
      "in each training log).", "tab:s-compute", ["Stage", "Held out", "Runs", "Hours"], rows)

led = [len(pd.read_csv(R / p / "EVAL_LEDGER.csv")) for p in ["stage6", "code_test"]]
out.append("%% ledger rows: stage6=%d code_test=%d\n" % tuple(led))
assert len(out) == 12
OUT.write_text("\n".join(out), encoding="utf-8")
print("wrote", OUT, "| ledger rows", led)
