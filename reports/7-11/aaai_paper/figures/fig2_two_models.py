"""Merged Fig (Fig 4 in the paper): TF vs FR on BOTH backbones (LeWM + frozen DINOv2).

SINGLE-COLUMN design: one panel, grouped by domain, 4 bars per domain
  LeWM-TF, LeWM-FR, DINOv2-TF, DINOv2-FR
Colour = protocol, using the heatmap's endpoints (Fig 2, "bwv"): vermillion =
teacher forcing (the worse/higher-error arm), blue = free rollout (better/lower),
i.e. the same good=blue / bad=vermillion semantic as the scan. Backbone = fill
style: LeWM full colour + solid, frozen DINOv2 light tint + same-hue hatch, which
makes the caption's "solid = LeWM, hatched = frozen DINOv2" literally true.

Per-bar value labels are dropped for the single-column size (the exact three-seed
means live in App. C's by-horizon table); the TF/FR fold-change is kept as the
figure's message. Writes into aaai_paper/figures/ AND paper/figures/.

Data sources (3-seed means, seeds 3072/1234/42; err = sample std over seeds):
  LeWM:   aaai_p0/rollout_*_baseline_{tf,fr}_s*.log ; physionpp/eval_pp_{tf,fr}_s*.log  (ledger C1)
  dinowm: dinowm/rollout_dinowm_{um,par,col,pp}_{tf,fr}_s*.log  (cross_model_dinowm.md §1)
  In-repo copy of every log: raw_data/runs/ (server /data1/... is retired).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_FIG = HERE.parent / "paper" / "figures"
# Muted "seaborn-deep" palette -- the understated top-conference look, validated
# by dataviz validate_palette.js (chroma / CVD / normal-vision all PASS light;
# the orange's sub-3:1 contrast WARN is relieved by the legend + the by-horizon
# table). slate blue = free rollout (good, lower error), soft orange = teacher
# forcing (baseline, higher error) -- good=cool / bad=warm, as the scan.
BLUE, VERM = "#4c72b0", "#dd8452"
VERM_L, BLUE_L = "#eec2a9", "#a6b9d8"        # light tints for frozen DINOv2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SECOND = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
                     "font.family": "DejaVu Sans"})

doms = ["uniform", "parab.\n(r/m)", "collis.", "Phys++\n(h64)"]
# Every bar is a 3-seed mean (seeds 3072/1234/42); err = sample std over those seeds.
lewm_tf, lewm_tf_sd = [0.300, 0.443, 1.152, 1.174], [0.007, 0.048, 0.048, 0.041]
lewm_fr, lewm_fr_sd = [0.136, 0.122, 0.479, 0.141], [0.008, 0.007, 0.079, 0.018]
lewm_mult = ["2.2×", "3.6×", "2.4×", "8.3×"]
dino_tf, dino_tf_sd = [0.594, 0.380, 0.803, 1.030], [0.078, 0.018, 0.041, 0.034]
dino_fr, dino_fr_sd = [0.427, 0.225, 0.479, 0.258], [0.047, 0.023, 0.011, 0.006]
dino_mult = ["1.4×", "1.7×", "1.7×", "4.0×"]

x = np.arange(len(doms))
w = 0.17
HATCH = "///"
off = {"lt": -0.27, "lf": -0.09, "dt": +0.09, "df": +0.27}

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ekw = dict(ecolor=INK, capsize=1.6, elinewidth=0.8, capthick=0.8)

def bars(dx, vals, sd, face, edge, hatch=""):
    ax.bar(x + dx, vals, w, yerr=sd, color=face, edgecolor=edge,
           linewidth=0.5, hatch=hatch, error_kw=ekw)

bars(off["lt"], lewm_tf, lewm_tf_sd, VERM, VERM)
bars(off["lf"], lewm_fr, lewm_fr_sd, BLUE, BLUE)
bars(off["dt"], dino_tf, dino_tf_sd, VERM_L, VERM, HATCH)
bars(off["df"], dino_fr, dino_fr_sd, BLUE_L, BLUE, HATCH)

# fold-change over each backbone's pair (per-bar values dropped at this size)
for xi in range(len(doms)):
    ax.text(xi - 0.18, max(lewm_tf[xi] + lewm_tf_sd[xi], lewm_fr[xi] + lewm_fr_sd[xi]) + 0.05,
            lewm_mult[xi], ha="center", va="bottom", fontweight="bold", fontsize=6.5, color=SECOND)
    ax.text(xi + 0.20, max(dino_tf[xi] + dino_tf_sd[xi], dino_fr[xi] + dino_fr_sd[xi]) + 0.05,
            dino_mult[xi], ha="center", va="bottom", fontweight="bold", fontsize=5.8, color=MUTED)

for xd in np.arange(len(doms) - 1) + 0.5:
    ax.axvline(xd, color=GRID, lw=0.8)

ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=6.5)
ax.set_ylabel("rollout error  (nMSE, ↓)", fontsize=7.5)
ax.set_ylim(0, 1.55); ax.tick_params(labelsize=7)
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)

h = [Patch(facecolor=VERM, label="LeWM TF"),
     Patch(facecolor=BLUE, label="LeWM FR"),
     Patch(facecolor=VERM_L, edgecolor=VERM, hatch=HATCH, label="DINO TF"),
     Patch(facecolor=BLUE_L, edgecolor=BLUE, hatch=HATCH, label="DINO FR")]
ax.legend(handles=h, frameon=False, fontsize=7.5, loc="upper left",
          ncol=2, columnspacing=0.8, handlelength=1.1, bbox_to_anchor=(0, 1.0))

fig.tight_layout()
for out in [HERE / "fig2_free_rollout_2models", PAPER_FIG / "fig2_free_rollout_2models"]:
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print("wrote", out)
plt.close(fig)
