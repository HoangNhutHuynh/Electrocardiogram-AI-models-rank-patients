"""Unit tests for stage 2 label harmonization. Run: pytest tests/unit -q

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calm_ecg.data.harmonize import (CHAPMAN_SNOMED_MAP, LABELS,  # noqa: E402
                                     PTBXL_SCP_MAP, harmonize_ptbxl,
                                     mimic_labels_from_report)


def test_label_order_fixed():
    assert LABELS == ["AF", "1dAVb", "RBBB", "LBBB", "SB", "STach"]


def test_ptbxl_map_excludes_incomplete_blocks():
    assert "IRBBB" not in PTBXL_SCP_MAP and "ILBBB" not in PTBXL_SCP_MAP
    assert PTBXL_SCP_MAP["CRBBB"] == "RBBB" and PTBXL_SCP_MAP["CLBBB"] == "LBBB"


def test_chapman_snomed_merges():
    assert CHAPMAN_SNOMED_MAP["59118001"] == "RBBB"
    assert CHAPMAN_SNOMED_MAP["713427006"] == "RBBB"  # CRBBB ≡ RBBB
    assert CHAPMAN_SNOMED_MAP["733534002"] == "LBBB"  # CLBBB ≡ LBBB


def test_harmonize_ptbxl_on_fixture(tmp_path):
    csv = tmp_path / "db.csv"
    csv.write_text(
        "ecg_id,patient_id,age,sex,scp_codes,filename_hr,strat_fold\n"
        "1,10,60,0,\"{'AFIB': 100.0}\",r/1,1\n"
        "2,11,50,1,\"{'IRBBB': 100.0}\",r/2,2\n"       # incomplete → all 0
        "3,12,40,0,\"{'CRBBB': 80.0, 'STACH': 0.0}\",r/3,3\n",
        encoding="utf-8")
    df = harmonize_ptbxl(csv, min_likelihood=0.0)
    assert df.loc[0, "AF"] == 1.0 and df.loc[0, LABELS].sum() == 1.0
    assert df.loc[1, LABELS].sum() == 0.0
    assert df.loc[2, "RBBB"] == 1.0 and df.loc[2, "STach"] == 1.0  # lik 0 counts
    assert df.loc[0, "sex"] == "M" and df.loc[1, "sex"] == "F"
    # no NaN for PTB-XL: its label space covers all six
    assert not df[LABELS].isna().any().any()


def test_masked_nan_is_not_zero():
    """NaN (label undefined) must stay NaN through a masked mean — the most
    common silent partial-label bug is treating NaN as negative."""
    col = pd.Series([1.0, 0.0, np.nan])
    assert np.isclose(col.mean(), 0.5)          # masked mean ignores NaN
    assert not np.isclose(col.fillna(0).mean(), 0.5)  # the bug we guard against


def test_mimic_regex_positive_and_negation():
    r = mimic_labels_from_report("Atrial fibrillation with rapid response.")
    assert r["AF"] == 1.0
    r = mimic_labels_from_report("No atrial fibrillation. Sinus bradycardia.")
    assert r["AF"] == 0.0 and r["SB"] == 1.0
    r = mimic_labels_from_report("Cannot rule out atrial fibrillation.")
    assert r["AF"] == 0.0
    r = mimic_labels_from_report("Incomplete right bundle branch block.")
    assert r["RBBB"] == 0.0
    r = mimic_labels_from_report("RIGHT BUNDLE-BRANCH BLOCK. 1st degree AV block.")
    assert r["RBBB"] == 1.0 and r["1dAVb"] == 1.0
