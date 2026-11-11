"""E2 (EXPLORATORY): AF vs atrial flutter (AFL) coding on the chapman fold.

CinC 2021 counts: Ningbo has 0 AF and 7,615 AFL; Shaoxing 1,780 AF / 445 AFL.
Re-scores the chapman-fold ERM models' AF output (no retraining):
  original | Shaoxing only (Ningbo AF undefined) | AF ∪ AFL positive
and checks whether EM's AF prevalence estimate tracks AF ∪ AFL.
Output: results/exploratory/af_afl/summary.csv
Usage: python scripts/explore_af_afl.py   (parses 45k headers: a few minutes)

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from calm_ecg.data.harmonize import LABELS, _parse_hea_dx_age_sex  # noqa: E402
from calm_ecg.eval.label_shift import em_prior  # noqa: E402

AFL = "164890007"
AF = LABELS.index("AF")
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

dcfg = yaml.safe_load((REPO / "configs/data.yaml").read_text(encoding="utf-8"))
wfdb_root = Path(dcfg["data_root"]) / dcfg["cohorts"]["chapman"]["dir"] / "WFDBRecords"

# rebuild the exact stage6 test order for the chapman fold (EcgDataset logic)
labels = pd.read_csv(REPO / "results/labels_all.csv", dtype={"record_id": str})
test_ids = set(json.loads((REPO / "results/splits_lodo.json").read_text(encoding="utf-8"))
               ["folds"]["chapman"]["test"])
excl = set(pd.read_csv(REPO / "results/qc_exclusions.csv", dtype={"record_id": str})["record_id"])
df = labels[labels["record_id"].isin(test_ids - excl)].reset_index(drop=True)

print(f"parsing {len(df)} headers for AFL...", flush=True)
afl = np.array([AFL in _parse_hea_dx_age_sex(wfdb_root / f"{p}.hea")[0] for p in df["path"]])
ningbo = df["record_id"].str[2:].astype(int).to_numpy() > 10646
af = df["AF"].to_numpy(float)
print(f"AF={int(af.sum())} AFL={int(afl.sum())} (Shaoxing AFL={int((afl & ~ningbo).sum())}, "
      f"Ningbo AFL={int((afl & ningbo).sum())}), AF&AFL overlap={int((afl & (af == 1)).sum())}")

rows = []
for npz in sorted((REPO / "results/stage6").glob("adapt_heldout-chapman_seed*_w_dro0_w_con0/predictions.npz")):
    z = np.load(npz)
    assert len(z["test_logits"]) == len(df), "test order mismatch with stage6"
    assert np.array_equal(z["test_y"][:, AF], af), "AF labels misaligned with stage6"
    s = z["test_logits"][:, AF]
    p = sig(s / float(z["temperature"]))
    either = ((af == 1) | afl).astype(float)
    rows.append({
        "run": npz.parent.name,
        "auroc_original": roc_auc_score(af, s),
        "auroc_shaoxing_only": roc_auc_score(af[~ningbo], s[~ningbo]),
        "auroc_af_or_afl": roc_auc_score(either, s),
        "auroc_ningbo_afl_as_target": roc_auc_score(afl[ningbo], s[ningbo]),
        "prev_af": af.mean(), "prev_af_or_afl": either.mean(),
        "prev_em": em_prior(p, float(np.nanmean(z["val_y"][:, AF])))[0]})

out = REPO / "results/exploratory/af_afl"
out.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(rows)
res.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
print(res.drop(columns="run").mean().round(4).to_string())
print(f"\n-> {out}")
