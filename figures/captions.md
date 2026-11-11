# Figure captions (draft, English) — numbers in captions are design facts only;
# result values are read from the figures/tables, never typed here.

**Fig. 1.** Study overview. Top: the five stages of the pipeline; the panel
beneath each stage shows a real output of that stage. (a) Cohort sizes.
(b) A 5-s window of one PTB-XL record (three of twelve leads). (c) The same
lead II with the pretraining mask (0.2-s spans, 40% of the window); shaded
spans are the reconstruction targets. (d) Adaptation: the pretrained encoder is
frozen; low-rank updates on the two top blocks and a linear head are trained.
(e) Macro-AUROC of the ERM model on source validation (open) and on the
held-out cohort (filled); the y-axis is truncated. (f) Leave-one-dataset-out
rotation. (g) Order of pre-registration, amendments, and logged evaluations.

**Fig. 2.** Leave-one-dataset-out generalization. (a) Macro-AUROC on source
validation data (open markers) and on the held-out cohort (filled markers, mean
± SD over 3 adaptation seeds) for each variant; the vertical segment is the
LODO gap. The y-axis is truncated to resolve differences near the ceiling.
(b) Per-label test AUROC of the ERM model (mean of 3 seeds).

**Fig. 3.** Pre-registered hypothesis H1. Difference in held-out worst-group
AUROC between each fairness-aware variant and ERM, per fold and pooled (diamond);
bars are 95% paired bootstrap confidence intervals (1000 resamples of test
records). Groups: sex × four age bins. Negative values favour ERM.

**Fig. 4.** Group-conditional coverage of label-conditional Mondrian conformal
prediction (α = 0.1) calibrated on source validation data and applied to each
held-out cohort (ERM, mean ± SD over 3 seeds). Filled markers: coverage of
positive cells by sex × age group (number of positive cells in parentheses);
crosses: coverage over all cells. Dashed line: nominal target 0.90.

**Fig. 5.** Label-definition shift behind a shared SNOMED-CT code. (a) Label
prevalence per site, with Chapman split into its two hospitals; AFL bars use the
CinC 2021 mapping counts. (b) AF AUROC on the Chapman fold scored as harmonized,
on Shaoxing only, with AF ∪ AFL as positive, and with Ningbo-coded AFL as the
target (ERM, mean ± SD over 3 seeds). (c) Target prevalence estimated by EM
prior-shift adaptation from unlabeled predictions vs. true prevalence, per fold
and label (log–log; identity line in grey; labels mark > 2-fold misses;
down-pointing triangles mark estimates that collapsed to zero, drawn at the axis
floor). Panels (b) and (c) are exploratory analyses (E2, E1).

**Fig. 6.** Precision–recall on CODE-test (827 ECGs, cardiologist gold standard)
per label. Curves: ERM averaged over 3 seeds (vertical averaging of interpolated
precision) for the external model (sources PTB-XL + Chapman) and the
in-population model (sources include CODE-15% weak labels). Coloured dots:
operating point at the source-validation threshold. Black markers: human readers
and the DNN of Ribeiro et al. on the same ECGs.

**Fig. 7.** Local threshold re-selection (exploratory analysis E3). Macro-F1 on
the remaining CODE-test records when thresholds are re-selected on k locally
labeled ECGs (200 random splits; band: 2.5th–97.5th percentile; labels with < 3
local positives keep source thresholds). Dotted: source thresholds. Diamonds:
in-sample oracle thresholds (optimistic upper bound). Dashed grey: readers on
the same records.

**Fig. S1.** Held-out AUROC per label for the three training objectives (mean ±
SD over 3 seeds). The y-axis is truncated and shared across panels.

**Fig. S2.** Worst-group AUROC on source validation and on the held-out cohort
for the three training objectives (mean ± SD over 3 seeds). The y-axis differs
between panels.
