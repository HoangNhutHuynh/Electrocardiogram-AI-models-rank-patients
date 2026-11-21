# CALM-ECG — Pre-registered hypothesis and decision rules

Written 2026-11-09, BEFORE any held-out test cohort was evaluated
(results/stage6/EVAL_LEDGER.csv did not exist at time of writing).
This file is frozen: later changes go in an "Amendments" section with a
date and reason, never by editing the text below.

## Context (source-val evidence, fold held-out = PTB-XL, 3 adapt seeds)

| Variant | macro-AUROC | worst-group AUROC |
|---|---|---|
| ERM (LoRA, no fairness term) | 0.9923 ± 0.0002 | 0.9781 ± 0.0029 |
| DRO | 0.9913 ± 0.0001 | 0.9723 ± 0.0032 |
| DRO + fair-SupCon (w=0.05) | 0.9910 ± 0.0002 | 0.9735 ± 0.0056 |

In-distribution, the fairness terms did not improve worst-group AUROC.

## Primary hypothesis H1

Under cohort shift (leave-one-dataset-out test), fairness-aware adaptation
(DRO, DRO+SupCon0.05) yields HIGHER worst-group AUROC on the held-out cohort
than ERM, at the cost of at most 0.5 points macro-AUROC.

## Fixed design

- Folds: held-out ∈ {ptbxl, chapman, code15} (MIMIC added later if
  downloaded; it would be reported as a separate amendment).
- Variants: ERM, DRO, DRO+SupCon0.05 — all three reported on every fold.
- Seeds: 3 adaptation seeds (0, 1, 2) per variant; one pretrained encoder
  per fold (pretrain seed 0).
- Checkpoint: ckpt_best.pt selected by source-val macro-AUROC (no test use).
- Groups: sex (M/F) × age bin {<40, 40–60, 60–75, ≥75}; groups with < 50
  test records are excluded from worst-group statistics (as coded).

## Primary metric and decision rule

Primary metric: test worst-group AUROC (stage6 metrics.json
`test.worst_group_auroc`).

For each fairness variant V vs ERM, compute the per-fold paired difference
Δ_f = mean_seeds(V) − mean_seeds(ERM).

- H1 SUPPORTED for V if: Δ_f > 0 in at least 2 of 3 folds AND the pooled
  mean Δ over folds has a 95% CI excluding 0 (bootstrap over test records,
  paired across variants, 1000 resamples), AND macro-AUROC drop ≤ 0.5 pt.
- H1 REJECTED for V if: pooled mean Δ ≤ 0 or its CI includes 0.
- Either outcome is reported in full. A rejection reframes the fairness
  component as a negative result; the paper's other contributions
  (harmonized labels, full LODO, PEFT, Mondrian conformal) stand regardless.

## Secondary metrics (reported, not used for the H1 verdict)

- macro-AUROC with 95% CI; per-class AUROC and F1@0.5
- ECE raw and after temperature scaling (fitted on source-val)
- Mondrian conformal (α = 0.1): marginal coverage, worst-group coverage,
  mean set size
- LODO gap: source-val macro-AUROC − test macro-AUROC

## Model reported as "proposed"

Chosen by source-val only. Current val evidence favors ERM; this choice is
not revisited after seeing test results. If H1 is supported, the fairness
variant is reported as a robustness gain under shift, not retro-labeled as
the selected model.

## Amendments

### Amendment 1 — 2026-11-09, after the PTB-XL fold test evaluation
(EVAL_LEDGER rows 1–9; chapman and code15 folds NOT yet evaluated)

What changed (secondary metrics only; H1, its metric and its decision rule
are unchanged):

1. Conformal: added label-conditional Mondrian, thresholds per
   (group, label, true value). The original (group, label) Mondrian is
   RETAINED and reported alongside.
   Reason: a methodological defect, visible from summary statistics alone.
   On PTB-XL test the original gave mean set size 0.86 < 1, i.e. ~14% empty
   prediction sets: one threshold per label is dominated by abundant easy
   negatives, so coverage of rare positives (sensitivity) is unguaranteed.
   Label-conditional calibration guarantees per-group coverage separately for
   positives and negatives. Calibration data unchanged (source-val only).
2. Temperature scaling: grid search replaced by continuous bounded
   minimization. Reason: grid resolution (~1.7%) made T identical across
   seeds. Calibration data unchanged.
3. stage6 now stores val predictions, so (1)–(2) are recomputed offline.
   Re-running stage6 for the 9 PTB-XL models adds ledger rows; those re-runs
   are disclosed here and change no primary-metric value (inference is
   deterministic, verified).

H1 status at time of amendment (fold ptbxl only, verdict INCOMPLETE):
Δ(DRO) = −0.0005 [−0.0127, 0.0101]; Δ(DRO+SupCon0.05) = +0.0084
[−0.0010, 0.0219].

### Amendment 2 — 2026-11-10, BEFORE chapman and code15 test evaluation
(stage5 for both folds complete; zero stage6 entries for either fold)

New secondary hypothesis H2, generated on the PTB-XL fold (discovery) and
tested on chapman and code15 (confirmation). PTB-XL is NOT counted as
confirmation.

Discovery observation (PTB-XL, ERM seed 0, label-conditional Mondrian,
α = 0.1): positive-class coverage 0.774 in group 4 (female, < 40 y;
115 positive cells) vs 0.887–0.921 in male groups 0–3.

H2: under cohort shift, label-conditional Mondrian conformal calibrated on
source-val under-covers POSITIVE cells of group 4 (female, < 40 y): its
positive coverage is below the 0.90 target AND below the pooled positive
coverage of male groups 0–3.

Decision rule (model = ERM, the val-selected model; 3 seeds averaged):
- Per confirmation fold: CONFIRMED if mean_seeds(pos_cov_g4) < 0.90 and
  < mean_seeds(pos_cov_male), where pos_cov_male pools groups 0–3 weighted
  by positive-cell count. NOT EVALUABLE if group 4 has < 20 positive cells
  (reported as such, not as a failure or a success).
- H2 SUPPORTED if CONFIRMED in both confirmation folds; PARTIAL if in one;
  REJECTED if in neither (among evaluable folds).
- (see "Exploratory analyses" below for post-hoc work)
- Also reported, not part of the rule: same statistics for DRO and
  DRO+SupCon0.05; n_fallback_cells (pooled-threshold fallback is a
  competing explanation for group-4 shortfall and is reported per fold).

### Amendment 3 — 2026-11-10, BEFORE any evaluation on CODE-test
(no model has produced predictions on CODE-test; labels harmonized only)

Data: CODE-test, 827 ECGs, gold-standard labels (cardiologist consensus;
annotations/gold_standard.csv). Reader annotations on the same ECGs:
cardiology_residents, emergency_residents, medical_students, dnn (Ribeiro
et al., Nat Commun 2020). cardiologist1/2 contributed to the gold standard,
so they are reported only as an inter-rater reference, not as comparators.

Leakage gate (run first): exact hash of every CODE-test tracing against all
345,779 CODE-15% tracings. Matching CODE-test records are excluded for
models whose source data included CODE-15% (ptbxl and chapman folds). If
> 5% of CODE-test matches, the in-population analysis is dropped and only
the external analysis is reported. Limitation acknowledged: exact hashing
does not detect a different exam of the same patient.

Models: ERM only (the val-selected model), 3 seeds per fold.
- External (primary): code15-fold models (sources PTB-XL + Chapman; never
  saw any Brazilian data).
- In-population (secondary): ptbxl- and chapman-fold models (sources include
  CODE-15% weak labels).

Thresholds: per label, τ_c = the threshold maximizing F1 on the model's own
source-val predictions (temperature-scaled). Never tuned on CODE-test.

Primary analysis (descriptive, no directional hypothesis): macro-F1 of the
external ERM (mean of 3 seeds' F1) vs each reader group, Δ with 95% paired
bootstrap CI over the 827 records (1000 resamples). Per-label F1 reported.
Secondary: per-label AUROC; same comparison for in-population models.

## Exploratory analyses (NOT pre-registered)

Labelled as exploratory in the manuscript; no confirmatory claim rests on them.

- E1 (2026-11-10, after all 36 test evaluations): EM prior-shift adaptation
  (Saerens et al. 2002) on unlabeled target predictions, ERM model, 3 seeds
  per fold. Motivation: chapman SB prevalence 36.7% vs 1.6–2.9% elsewhere and
  positive conformal coverage 0.44. Script: scripts/explore_label_shift.py.

- E2 (2026-11-10, after all test evaluations): AF/AFL sensitivity on the
  chapman fold. CinC 2021 counts show Ningbo with 0 AF and 7,615 AFL vs
  Shaoxing 1,780 AF / 445 AFL. AF re-scored (a) Shaoxing only (Ningbo AF set
  undefined), (b) AF ∪ AFL as positive. Evaluation-only, no retraining.
  Script: scripts/explore_af_afl.py.
- E3 (2026-11-10, after CODE-test evaluation): local threshold re-selection
  with k ∈ {25, 50, 100, 200} labeled CODE-test ECGs (200 random splits;
  labels with < 3 local positives keep source-val thresholds), evaluated on
  the remaining records, readers scored on the same records; plus an
  in-sample oracle upper bound. Script: scripts/explore_local_recalibration.py.

- E4 (2026-11-14, written BEFORE any ResNet run was trained or evaluated):
  architecture control. Question: do the LODO gap, the conformal coverage
  failure and the decision-level failure depend on the state-space encoder?
  Backbone: dilated residual CNN (src/calm_ecg/models/resnet.py), same stem,
  LayerNorm instead of BatchNorm (BatchNorm would adapt its statistics to the
  held-out cohort), 4,079,872 encoder parameters vs 4,084,480 for S4D,
  LoRA trainable fraction 1.18% vs 1.47%. Pipeline, splits, pretraining
  recipe, adaptation recipe and evaluation unchanged; one pretrained encoder
  per fold (seed 0); ERM only, adaptation seeds 0, 1, 2; all three folds.
  Run names carry the suffix "_resnet"; stage7 reports them as a separate
  variant "ERM (resnet)" and they never enter H1, Table II or the CODE-test
  comparison of the main model. Reported in the supplement as a sensitivity
  analysis. What would count as replication, fixed now: the ResNet ERM
  model shows (a) a LODO macro-AUROC gap in the same ordering across folds
  (Chapman largest), and (b) positive-class label-conditional coverage on
  the Chapman fold below 0.70. Either failing is reported as a failure.
  Every evaluation is appended to results/stage6/EVAL_LEDGER.csv as usual.

- E4b (2026-11-15, written BEFORE any fairness-aware ResNet run was trained
  or evaluated; the ResNet ERM runs of E4 were still in progress): H1
  sensitivity on the architecture control. Same two variants as H1 (group
  DRO, w_dro = 0.5, eta = 0.01; DRO + fair-SupCon, w_con = 0.05), seeds
  0, 1, 2, all three folds, same pretrained ResNet encoders as E4. The
  H1 decision rule is applied unchanged, but each variant is compared with
  the ResNet ERM model only; no quantity is pooled across architectures,
  and the pre-registered H1 verdict for the state-space encoder is not
  revised by this analysis. Reported in the supplement next to E4.
