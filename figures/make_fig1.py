"""Fig. 1 — study overview: five pipeline stages, each with one REAL output
directly beneath it, plus the LODO rotation and the pre-registration order.

Thumbnails come from pipeline outputs (results/figure_data/*.csv) and one
PTB-XL record; nothing is illustrative data. Panel (c) masks that record with
the span length and ratio used in pretraining (0.2-s spans, 40%); it shows the
masked input and the target, not a model reconstruction.
Output: figures/out/Fig1_overview.{pdf,svg,png}   (7.16 in wide, text >= 7 pt)
Usage: python figures/make_fig1.py [repo_root] [data_root]

Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(cfg["data_root"])
FD = ROOT / "results/figure_data"

FS = 7.0                                               # minimum text size (pt)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": FS,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 2, "legend.frameon": False})
BLUE, VERM, GREEN, GRAY, LIGHT = "#0072B2", "#D55E00", "#009E73", "#737373", "#F2F2F2"

# ---- one real ECG (PTB-XL record 00001, WFDB format 16, 1000 adu/mV, 500 Hz) ----
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
raw = np.fromfile(DATA / cfg["cohorts"]["ptbxl"]["dir"] / "records500/00000/00001_hr.dat",
                  dtype="<i2").reshape(-1, 12) / 1000.0            # mV
win = raw[1000:3500]                                               # 5-s window
win = win - np.median(win, axis=0)
t = np.arange(len(win)) / 500.0
slots = np.random.default_rng(0).choice(25, 10, replace=False)     # 10 of 25 spans = 40%
mask = np.zeros(len(win), bool)
for s in slots:
    mask[s * 100:(s + 1) * 100] = True

fig = plt.figure(figsize=(7.16, 4.05))
gs = fig.add_gridspec(3, 5, height_ratios=[0.36, 1.0, 0.50], hspace=0.30, wspace=0.34,
                      left=0.045, right=0.99, top=0.99, bottom=0.03)

STAGES = [("1  Cohorts", "3 countries\n6 harmonized labels"),
          ("2  Preprocessing", "500 Hz, 5-s window\n769 excluded by QC"),
          ("3  Pretraining", "S4D, 4.1 M weights\nmasked signal"),
          ("4  Adaptation", "LoRA, 1.47% trained\n3 objectives"),
          ("5  Evaluation", "held-out cohort\nunseen at all stages")]
hdr = []
for col, (title, lines) in enumerate(STAGES):
    ax = fig.add_subplot(gs[0, col]); ax.axis("off"); hdr.append(ax)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=LIGHT, ec="#404040", lw=0.6, transform=ax.transAxes,
                                clip_on=False))
    ax.text(0.5, 0.77, title, ha="center", va="center", fontsize=FS + 0.5, fontweight="bold")
    ax.text(0.5, 0.33, lines, ha="center", va="center", fontsize=FS, linespacing=1.2)
for a, b in zip(hdr[:-1], hdr[1:]):
    fig.add_artist(ConnectionPatch((1.0, 0.5), (0.0, 0.5), coordsA=a.transAxes,
                                   coordsB=b.transAxes, arrowstyle="-|>", mutation_scale=7,
                                   lw=0.8, color="#404040", shrinkA=1, shrinkB=1))


def tag(ax, s):
    ax.text(0.0, 1.03, s, transform=ax.transAxes, fontsize=FS + 1, fontweight="bold",
            ha="left", va="bottom")


def inset(ax, left=0.0, bottom=0.045, top=0.03):
    """Shrink an axes inside its grid cell to leave room for labels."""
    p = ax.get_position()
    ax.set_position([p.x0 + left, p.y0 + bottom, p.width - left, p.height - bottom - top])


# (a) cohort sizes
ax = fig.add_subplot(gs[1, 0])
names = ["PTB-XL", "Chapman", "CODE-15%", "CODE-test"]
n = [21799, 45152, 345779, 827]
ax.barh(range(4), n, color=[BLUE, BLUE, BLUE, GRAY], height=0.6)
for i, v in enumerate(n):
    ax.text(v * 1.2, i, f"{v:,}", va="center", fontsize=FS)
ax.set_xscale("log"); ax.set_xlim(300, 2e7); ax.invert_yaxis()
ax.set_yticks(range(4)); ax.set_yticklabels(names)
ax.set_xticks([1e3, 1e5]); ax.set_xlabel("ECG records")
ax.tick_params(axis="y", length=0, pad=1)
inset(ax, left=0.055); tag(ax, "a")

# (b) three leads of the window, in mV with a scale bar
ax = fig.add_subplot(gs[1, 1])
for k, lead in enumerate(["II", "V1", "V5"]):
    ax.plot(t, win[:, LEADS.index(lead)] - 1.5 * k, color="k", lw=0.5)
    ax.text(-0.15, -1.5 * k, lead, ha="right", va="center", fontsize=FS)
ax.plot([4.8, 4.8], [0.5, 1.5], color="k", lw=1.0)
ax.text(4.7, 1.0, "1 mV", ha="right", va="center", fontsize=FS)
ax.set_xlim(0, 5); ax.set_ylim(-3.7, 1.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Time (s)")
inset(ax, left=0.02); tag(ax, "b")

# (c) masked input and target (lead II)
ax = fig.add_subplot(gs[1, 2])
x = win[:, LEADS.index("II")]
edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
for a, b in zip(edges[::2], edges[1::2]):
    ax.axvspan(a / 500, b / 500, color=VERM, alpha=0.15, lw=0)
ax.plot(t, x, color=GRAY, lw=0.5)
ax.plot(t, np.where(mask, 0.0, x) - 1.7, color=BLUE, lw=0.5)
ax.text(0.03, 1.15, "target", fontsize=FS, color=GRAY, va="center")
ax.text(0.03, -2.45, "masked input", fontsize=FS, color=BLUE, va="center")
ax.set_xlim(0, 5); ax.set_ylim(-2.8, 1.5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Time (s)")
inset(ax); tag(ax, "c")

# (d) adaptation schematic: frozen encoder, trained LoRA + head
ax = fig.add_subplot(gs[1, 3]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
bw, bh, y0, gap = 0.50, 0.095, 0.03, 0.022
ax.add_patch(Rectangle((0.02, y0), bw, bh, fc=LIGHT, ec="k", lw=0.5))
ax.text(0.02 + bw / 2, y0 + bh / 2, "stem", ha="center", va="center", fontsize=FS)
for i in range(6):
    yb = y0 + (i + 1) * (bh + gap)
    ax.add_patch(Rectangle((0.02, yb), bw, bh, fc=LIGHT, ec="k", lw=0.5))
    ax.text(0.02 + bw / 2, yb + bh / 2, f"block {i + 1}", ha="center", va="center", fontsize=FS)
    if i >= 4:
        ax.add_patch(Rectangle((0.56, yb), 0.30, bh, fc=VERM, ec="k", lw=0.5))
        ax.text(0.71, yb + bh / 2, "LoRA", ha="center", va="center", fontsize=FS, color="w")
yh = y0 + 7 * (bh + gap)
ax.add_patch(Rectangle((0.02, yh), 0.84, bh, fc=VERM, ec="k", lw=0.5))
ax.text(0.44, yh + bh / 2, "head, 6 labels", ha="center", va="center", fontsize=FS, color="w")
yf0, yf1 = y0, y0 + 4 * (bh + gap) + bh                             # bracket: stem..block 4
ax.plot([0.57, 0.60, 0.60, 0.57], [yf0, yf0, yf1, yf1], color=GRAY, lw=0.7)
ax.text(0.64, (yf0 + yf1) / 2, "frozen", color=GRAY, fontsize=FS, va="center")
ax.text(0.94, yh - 0.07, "trained", color=VERM, fontsize=FS, va="center", ha="center", rotation=90)
tag(ax, "d")

# (e) LODO gap of the ERM model (real)
L = pd.read_csv(FD / "fig2a_lodo.csv"); L = L[L.variant == "ERM"]
ax = fig.add_subplot(gs[1, 4])
for i, f in enumerate(["PTB-XL", "Chapman", "CODE-15%"]):
    s = L[L.fold == f]
    v, te = s.val_macro_auroc.mean(), s.test_macro_auroc.mean()
    ax.plot([i, i], [v, te], color=BLUE, lw=0.9)
    ax.plot(i, v, "o", ms=4, mfc="w", mec=BLUE, mew=0.9)
    ax.plot(i, te, "o", ms=4, color=BLUE)
ax.plot([], [], "o", ms=4, mfc="w", mec=BLUE, mew=0.9, label="val")
ax.plot([], [], "o", ms=4, color=BLUE, label="test")
ax.set_xticks(range(3)); ax.set_xticklabels(["PTB-XL", "Chapman", "CODE-15%"], rotation=30, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.945, 1.0); ax.set_yticks([0.96, 0.98, 1.0])
ax.set_ylabel("Macro-AUROC", labelpad=1)
ax.legend(loc="lower left", handletextpad=0.0, borderaxespad=0.1, labelspacing=0.2)
inset(ax, left=0.045, bottom=0.075, top=0.03); tag(ax, "e")

# (f) leave-one-dataset-out rotation
ax = fig.add_subplot(gs[2, 0:2])
for r in range(3):
    for c in range(3):
        held = r == c
        ax.add_patch(Rectangle((c * 1.25, 2 - r), 1.18, 0.82, fc=VERM if held else BLUE, ec="w", lw=0.5))
        ax.text(c * 1.25 + 0.59, 2.41 - r, "held out" if held else "source", ha="center",
                va="center", color="w", fontsize=FS)
ax.set_xlim(0, 5.6); ax.set_ylim(0, 3)
ax.set_xticks([0.59, 1.84, 3.09]); ax.set_xticklabels(["PTB-XL", "Chapman", "CODE-15%"])
ax.set_yticks([0.41, 1.41, 2.41]); ax.set_yticklabels(["Fold 3", "Fold 2", "Fold 1"])
ax.tick_params(length=0, pad=1); [s.set_visible(False) for s in ax.spines.values()]
ax.text(3.95, 1.5, "per fold:\n3 objectives\n3 seeds", fontsize=FS, va="center", linespacing=1.3)
inset(ax, left=0.03, bottom=0.05, top=0.02); tag(ax, "f")

# (g) order of pre-registration, amendments and evaluations
ax = fig.add_subplot(gs[2, 2:5]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [("Protocol,\nH1 rule", GRAY), ("Fold 1\ntest", BLUE), ("Amend.\nH2", GRAY),
         ("Folds 2, 3\ntest", BLUE), ("Amend.\nCODE-test", GRAY), ("Reader\ncomparison", BLUE),
         ("Explor-\natory", GREEN)]
xs = np.linspace(0.07, 0.93, len(steps))
ax.annotate("", xy=(1.0, 0.60), xytext=(0.0, 0.60),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#404040", mutation_scale=7))
for xk, (s, col) in zip(xs, steps):
    ax.plot(xk, 0.60, "o", ms=6, color=col, zorder=3)
    ax.text(xk, 0.44, s, ha="center", va="top", fontsize=FS, linespacing=1.15)
for xk, lab, col in [(0.10, "written first", GRAY), (0.38, "logged evaluation", BLUE),
                     (0.72, "post hoc, labeled", GREEN)]:
    ax.plot(xk, 0.90, "o", ms=4, color=col)
    ax.text(xk + 0.02, 0.90, lab, va="center", fontsize=FS, color=col)
tag(ax, "g")

out = HERE / "out"; out.mkdir(exist_ok=True)
for ext, kw in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
    fig.savefig(out / f"Fig1_overview.{ext}", **kw)
print("->", out / "Fig1_overview.pdf")
