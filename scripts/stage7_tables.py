"""Stage 7 — aggregate all run artifacts into paper tables. READ-ONLY over
results/: every number in the manuscript must come from these outputs,
never typed by hand.

Outputs (results/stage7/):
  ablation_val.csv / .md   — per (held_out, variant): val macro-AUROC and
                             worst-group AUROC as mean ± std over seeds
                             (from stage5 */log.csv best-epoch rows + config)
  lodo_test.csv / .md      — per (held_out, variant): test macro-AUROC [CI],
                             worst-group, ECE raw/scaled, conformal coverage
                             (from stage6 */metrics.json)
A variant is identified by (w_dro, w_con, lora_top_blocks) from each run's
config.yaml, so renamed folders cannot mislabel a row.

Usage: python scripts/stage7_tables.py

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def variant_name(cfg: dict) -> str:
    # The architecture control is its own variant, so it is never averaged
    # into the s4d rows; configs from before it carry no 'arch' key.
    arch = cfg.get("arch", "s4d")
    name = _objective_name(cfg)
    return name if arch == "s4d" else f"{name} ({arch})"


def _objective_name(cfg: dict) -> str:
    if cfg.get("lora_top_blocks", 0) == 0:
        return "head-only"
    w_dro, w_con = cfg.get("w_dro", 0), cfg.get("w_con", 0)
    if w_dro == 0 and w_con == 0:
        return "ERM"
    parts = []
    if w_dro > 0:
        parts.append("DRO")
    if w_con > 0:
        parts.append(f"SupCon{w_con:g}")
    return "+".join(parts)


def collect_stage5(repo: Path) -> pd.DataFrame:
    rows = []
    for run in sorted((repo / "results/stage5").glob("*/")):
        cfg_p, log_p = run / "config.yaml", run / "log.csv"
        if not (cfg_p.exists() and log_p.exists()):
            continue
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        log = pd.read_csv(log_p)
        # legacy logs from re-runs contain appended duplicate epochs: keep last
        log = log.drop_duplicates(subset="epoch", keep="last")
        if log.empty:
            continue
        best = log.loc[log["val_macro_auroc"].idxmax()]
        rows.append({"held_out": cfg["held_out"], "variant": variant_name(cfg),
                     "seed": cfg["seed"], "run": run.name,
                     "val_macro_auroc": best["val_macro_auroc"],
                     "val_worst_group": best["val_worst_group_auroc"],
                     "best_epoch": int(best["epoch"])})
    return pd.DataFrame(rows)


def collect_stage6(repo: Path) -> pd.DataFrame:
    rows = []
    for mp in sorted((repo / "results/stage6").glob("*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        cfg_run = repo / "results/stage5" / m["run_name"] / "config.yaml"
        var = (variant_name(yaml.safe_load(cfg_run.read_text(encoding="utf-8")))
               if cfg_run.exists() else "?")
        t = m["test"]
        lc = t.get("conformal_label_conditional") or {}
        rows.append({"held_out": m["held_out"], "variant": var,
                     "seed": m["seed"], "run_name": m["run_name"],
                     "conf_lc_cov": lc.get("marginal_coverage", float("nan")),
                     "conf_lc_pos_cov": lc.get("positive_coverage") or float("nan"),
                     "conf_lc_abstain": lc.get("abstain_rate", float("nan")),
                     "test_macro_auroc": t["macro_auroc"],
                     "ci_lo": t["macro_auroc_ci95"][0],
                     "ci_hi": t["macro_auroc_ci95"][1],
                     "test_worst_group": t["worst_group_auroc"],
                     "ece_raw": t["ece_raw"],
                     "ece_temp": t["ece_temp_scaled"],
                     "conf_marginal_cov": t["conformal"]["marginal_coverage"],
                     "conf_worst_group_cov": min(
                         (d["coverage"] for d in
                          t["conformal"]["per_group"].values()),
                         default=float("nan"))})
    return pd.DataFrame(rows)


def h1_analysis(repo: Path, s6: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Pre-registered H1 (PREREGISTRATION.md): paired bootstrap of
    Δ worst-group AUROC (variant − ERM) per fold, pooled verdict."""
    sys.path.insert(0, str(repo / "src"))
    from calm_ecg.eval.paired import fold_delta_draws, pooled_verdict

    def load_probs(run_name: str):
        z = np.load(repo / "results/stage6" / run_name / "predictions.npz")
        return z["test_y"], z["test_g"], 1 / (1 + np.exp(-z["test_logits"]))

    out = {}
    # Each architecture is compared with its own ERM; nothing is pooled
    # across architectures (E4b in PREREGISTRATION.md).
    pairs = [(v + sfx, "ERM" + sfx) for sfx in ("", " (resnet)")
             for v in ("DRO", "DRO+SupCon0.05")]
    for variant, erm in pairs:
        fold_res, macro_drop = {}, {}
        for fold in sorted(s6["held_out"].unique()):
            sub = s6[s6["held_out"] == fold]
            rv = sub[sub["variant"] == variant]["run_name"].tolist()
            re = sub[sub["variant"] == erm]["run_name"].tolist()
            if not rv or not re:
                continue
            ys, pv, pe = [], [], []
            for r in rv:
                y, g, p = load_probs(r)
                ys.append((y, g)); pv.append(p)
            for r in re:
                y, g, p = load_probs(r)
                ys.append((y, g)); pe.append(p)
            y0, g0 = ys[0]
            assert all(np.array_equal(np.nan_to_num(y, nan=-1),
                                      np.nan_to_num(y0, nan=-1))
                       and np.array_equal(g, g0) for y, g in ys), \
                f"[{fold}] test records differ across models — cannot pair"
            print(f"H1 {variant} vs {erm}, fold {fold}: bootstrapping "
                  f"{n_boot}x on {len(y0)} records...", flush=True)
            fold_res[fold] = fold_delta_draws(y0, g0, pv, pe, n_boot=n_boot, progress=True)
            macro_drop[fold] = float(
                sub[sub["variant"] == erm]["test_macro_auroc"].mean()
                - sub[sub["variant"] == variant]["test_macro_auroc"].mean())
        if fold_res:
            out[variant] = pooled_verdict(fold_res, macro_drop)
    return out


def h2_fold(metrics_list: list[dict]) -> dict:
    """Amendment 2 rule for one fold, given the ERM seeds' metrics.json."""
    g4, male, nfb = [], [], []
    for m in metrics_list:
        lc = m["test"]["conformal_label_conditional"]
        pg = {int(k): v for k, v in lc["per_group"].items()}
        nfb.append(lc["n_fallback_cells"])
        if pg.get(4, {}).get("positive_coverage") is None:
            return {"status": "NOT EVALUABLE (<20 positive cells in group 4)"}
        g4.append(pg[4]["positive_coverage"])
        cells = [(pg[k]["positive_coverage"], pg[k]["n_pos_cells"])
                 for k in range(4) if pg.get(k, {}).get("positive_coverage") is not None]
        male.append(sum(c * n for c, n in cells) / sum(n for _, n in cells))
    g4m, malem = float(np.mean(g4)), float(np.mean(male))
    ok = g4m < 0.90 and g4m < malem
    return {"status": "CONFIRMED" if ok else "NOT CONFIRMED",
            "pos_cov_g4": round(g4m, 4), "pos_cov_male": round(malem, 4),
            "n_fallback_cells": nfb}


def h2_analysis(repo: Path, s6: pd.DataFrame) -> dict:
    out = {}
    for fold in sorted(s6["held_out"].unique()):
        for variant in ("ERM", "DRO", "DRO+SupCon0.05"):
            runs = s6[(s6["held_out"] == fold) & (s6["variant"] == variant)]["run_name"]
            ms = [json.loads((repo / "results/stage6" / r / "metrics.json")
                             .read_text(encoding="utf-8")) for r in runs]
            ms = [m for m in ms if "conformal_label_conditional" in m["test"]]
            if ms:
                out.setdefault(fold, {})[variant] = h2_fold(ms)
    conf = [f for f in ("chapman", "code15") if f in out and "ERM" in out[f]
            and not out[f]["ERM"]["status"].startswith("NOT EVALUABLE")]
    n_ok = sum(out[f]["ERM"]["status"] == "CONFIRMED" for f in conf)
    if not all(f in out and "ERM" in out[f] for f in ("chapman", "code15")):
        out["verdict_ERM"] = "INCOMPLETE"
    elif not conf:
        out["verdict_ERM"] = "NOT EVALUABLE"
    else:
        out["verdict_ERM"] = ["REJECTED", "PARTIAL", "SUPPORTED"][n_ok]
    return out


def agg_mean_std(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["held_out", "variant"])
    out = g[cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.round(4).reset_index()


def to_md(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", "",
             "| " + " | ".join(df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "results/stage7"
    out.mkdir(parents=True, exist_ok=True)

    s5 = collect_stage5(repo)
    if s5.empty:
        print("no stage5 runs found")
        return 1
    s5.to_csv(out / "runs_stage5_raw.csv", index=False)
    abl = agg_mean_std(s5, ["val_macro_auroc", "val_worst_group"])
    abl.to_csv(out / "ablation_val.csv", index=False)
    to_md(abl, out / "ablation_val.md", "Ablation (source-val, mean/std/seeds)")
    print("=== ablation (val) ===")
    print(abl.to_string(index=False))

    s6 = collect_stage6(repo)
    if not s6.empty:
        s6.to_csv(out / "runs_stage6_raw.csv", index=False)
        lodo = agg_mean_std(s6, ["test_macro_auroc", "test_worst_group",
                                 "ece_raw", "ece_temp",
                                 "conf_marginal_cov", "conf_worst_group_cov",
                                 "conf_lc_cov", "conf_lc_pos_cov",
                                 "conf_lc_abstain"])
        lodo.to_csv(out / "lodo_test.csv", index=False)
        to_md(lodo, out / "lodo_test.md", "LODO test (mean/std/seeds)")
        print("\n=== LODO (test) ===")
        print(lodo.to_string(index=False))

        n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
        h1 = h1_analysis(repo, s6, n_boot=n_boot)
        (out / "h1_verdict.json").write_text(json.dumps(h1, indent=2),
                                             encoding="utf-8")
        print("\n=== Pre-registered H1 (worst-group AUROC, variant − ERM) ===")
        print(json.dumps(h1, indent=2))

        h2 = h2_analysis(repo, s6)
        (out / "h2_verdict.json").write_text(json.dumps(h2, indent=2),
                                             encoding="utf-8")
        print("\n=== Amendment 2 H2 (group-4 positive conformal coverage) ===")
        print(json.dumps(h2, indent=2))
    else:
        print("\n(no stage6 metrics yet — LODO table skipped)")
    print(f"\ntables -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
