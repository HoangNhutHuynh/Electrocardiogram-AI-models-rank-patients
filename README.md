# Leave-one-dataset-out audit of 12-lead ECG models

Code, pre-registration and results for the manuscript *"Deep learning
electrocardiogram models rank patients well but make poor diagnostic decisions
in an unseen country"*.

Author: Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan.

A compact state-space encoder is pretrained without labels on two of three
public ECG cohorts and adapted with low-rank updates, then evaluated on the
third cohort, which is absent from every training stage. Three objectives are
compared under a decision rule written down before any held-out evaluation:
empirical risk minimization (ERM), group distributionally robust optimization
(group DRO), and group DRO with a group-aware contrastive term. A fourth
dataset, with cardiologist consensus labels and independent human readings,
serves as an external reference. As an architecture control, a
parameter-matched dilated convolutional encoder repeats the pipeline
(exploratory analyses E4 and E4b, registered before their models were trained).

**Reviewers: start at [VERIFY.md](VERIFY.md).** It maps every number in the
manuscript to the file it came from and to the command that regenerates it.
All result files are committed, so the tables and figures can be rebuilt
without a GPU and without downloading the raw data.

## Cohorts

| Cohort | Country | Records | Role |
|---|---|---|---|
| PTB-XL | Germany | 21,799 | fold 1 held out |
| Chapman-Shaoxing + Ningbo | China | 45,152 | fold 2 held out |
| CODE-15% | Brazil | 345,779 | fold 3 held out |
| CODE-test | Brazil | 827 | external gold standard, cardiologist consensus |

Quality control removed 769 records (97 with non-finite samples, 2 unreadable,
670 flat), leaving 411,961. Six labels are harmonized through the SNOMED-CT
mapping of the PhysioNet/CinC 2021 challenge: AF, 1dAVb, RBBB, LBBB, SB,
STach. In the label tables, `1` is positive, `0` negative, and an empty cell
means the label is not defined in that cohort; undefined labels are masked in
the loss and never treated as negatives.

The datasets are not redistributed here. Download them from PhysioNet
(PTB-XL, Chapman-Shaoxing, Ningbo) and Zenodo (CODE-15%, CODE-test), then set
`data_root` in `configs/data.yaml`, `configs/pretrain.yaml` and
`configs/adapt.yaml`.

## Repository layout

```
configs/            data.yaml, pretrain.yaml, adapt.yaml
src/calm_ecg/       data/   harmonize, splits, signals, dataset
                    models/ s4d, resnet (architecture control), lora
                    train/  pretrain, adapt, fairness
                    eval/   metrics, conformal, paired, label_shift
scripts/            stage0..stage7 pipeline, exploratory analyses, table export
tests/unit/         67 unit tests
results/            every artifact the manuscript cites (see VERIFY.md)
figures/            make_figures.m (Fig. 2-6, S1-S3), make_fig1.py, out/,
                    graphical_abstract/
manuscript/         supp_tables.tex (manuscript sources follow after publication)
PREREGISTRATION.md  hypotheses, three dated amendments, exploratory analyses E1-E4b
```

Model checkpoints (126 files, 2.5 GB) are excluded by `.gitignore`. Everything needed to
check the reported numbers is committed; re-running stages 4 and 5 reproduces
the weights.

## Setup

```bash
conda create -n calmecg python=3.11 -y
conda activate calmecg
pip install -r requirements.txt     # PyTorch is installed separately
pytest tests/unit -q                # 67 tests, no data needed
```

PyTorch is not pinned in `requirements.txt` because the right build depends on
the GPU. The reported runs used torch 2.11 with CUDA 12.8 on a single 8 GB
card. Figures are drawn in MATLAB R2021a or newer, base installation only.

## Pipeline

Stages 0 to 3 need the raw data; stages 4 and 5 need a GPU. Stage 6 appends a
row to `results/stage6/EVAL_LEDGER.csv` on every invocation, so each look at a
held-out cohort leaves a trace.

```bash
python scripts/stage0_verify.py            # cohort counts, writes MANIFEST.json
python scripts/stage1_sanity.py            # signal overlays
python scripts/stage1b_qc_scan.py          # quality control, exclusion list
python scripts/stage2_labels.py            # harmonized labels per cohort
python scripts/stage3_splits.py            # LODO splits, patient level, leakage check

python scripts/stage4_pretrain.py --held-out ptbxl --seed 0
python scripts/stage5_adapt.py --held-out ptbxl --seed 0 --pretrain-seed 0
python scripts/stage6_eval.py results/stage5/<run_name>/ckpt_best.pt
python scripts/stage7_tables.py            # H1, H2, LODO table

# architecture control (E4, E4b): the same stages with the convolutional encoder;
# run names get the suffix _resnet and stage 7 reports them as separate variants
python scripts/stage4_pretrain.py --held-out ptbxl --seed 0 --arch resnet
python scripts/stage5_adapt.py --held-out ptbxl --seed 0 --pretrain-seed 0 --arch resnet

python scripts/check_codetest_overlap.py   # CODE-test against CODE-15%, by hashing
python scripts/eval_code_test.py           # comparison with human readers
python scripts/explore_label_shift.py      # E1, EM prior shift
python scripts/explore_af_afl.py           # E2, AF rescoring
python scripts/explore_local_recalibration.py   # E3, local thresholds

python scripts/export_figure_data.py       # tidy CSVs read by the figure scripts
python scripts/make_supplement_tables.py   # manuscript/supp_tables.tex
python figures/graphical_abstract/make_graphical_abstract.py   # needs cairosvg
```

Fairness objectives are selected through `--set`, for example
`--set w_dro=0.5` for group DRO and `--set w_con=0.05` to add the contrastive
term. Stage 2 walks all 45,152 Chapman headers and takes a few minutes on
first run.

## Citation

The manuscript is under review. Until it appears, please cite this repository
together with the four datasets it builds on.
