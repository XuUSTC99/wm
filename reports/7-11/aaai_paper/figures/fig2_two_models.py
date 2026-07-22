"""Merged Fig 2: TF vs FR on BOTH backbones (LeWM + frozen-DINOv2 variant).
Two panels, same style/palette as storyline fig2. Writes into aaai_paper/figures/
AND paper/figures/ (the copy the LaTeX includes).

Data sources:
  LeWM:   aaai_p0/rollout_*_baseline_{tf,fr}_s{3072,1234,42}.log ;
          physionpp/eval_pp_{tf,fr}_s*.log       (3-seed means, ledger C1)
  dinowm: dinowm/rollout_dinowm_{um,par,col}_{tf,fr}_s*.log ;
          dinowm/rollout_dinowm_pp_{tf,fr}_s*.log (3-seed means, cross_model_dinowm.md §1)
"""
#
# ---------------------------------------------------------------------------
# DATA SOURCE (updated 2026-07-19)
#   Use  /home/qlib/am/wm/raw_data/runs/   -- the in-repo copy of every log.
#   The /data1/likun-share/junjxu/runs/ paths quoted below are the ORIGINAL
#   locations; that server is being retired, so treat them as provenance
#   notes, not as a path to read from right now.
#   NOTE: seed naming differs by seed (see raw_data/README.md #0) --
#     seed 3072 (default) has NO _s suffix and lives in runs/structdyn_eval/;
#     seeds 1234 / 42 carry _s1234 / _s42 and live in runs/aaai_p0/.
#   Also: a run with no pwN in its name is the default slot weight w=1.
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_FIG = HERE.parent / "paper" / "figures"
BLUE, VERM = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#D9D9D9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "axes.titleweight": "bold",
                     "font.family": "DejaVu Sans"})

doms = ["uniform", "parabola\n(r/m-OOD)", "collision", "Physion++\n(sim, h64)"]
# Every bar is a 3-seed mean (seeds 3072/1234/42); err = sample std over those seeds.
# Verified against the per-seed logs on 2026-07-19 (all 16 bars have n=3).
panels = [
    ("LeWM (trainable ViT-tiny)",
     [0.300, 0.443, 1.152, 1.174], [0.007, 0.048, 0.048, 0.041],   # TF mean, std
     [0.136, 0.122, 0.479, 0.141], [0.008, 0.007, 0.079, 0.018],   # FR mean, std
     ["2.2×", "3.6×", "2.4×", "8.3×"]),
    ("frozen DINOv2 + adapter (DINO-WM-style)",
     [0.594, 0.380, 0.803, 1.030], [0.078, 0.018, 0.041, 0.034],
     [0.427, 0.225, 0.479, 0.258], [0.047, 0.023, 0.011, 0.006],
     ["1.4×", "1.7×", "1.7×", "4.0×"]),
]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.5), sharey=True)
for ax, (title, tf, tf_sd, fr, fr_sd, mult) in zip(axes, panels):
    x = np.arange(len(doms)); w = 0.38
    ekw = dict(ecolor=INK, capsize=3, elinewidth=1.1, capthick=1.1)
    ax.bar(x - w/2, tf, w, yerr=tf_sd, color=VERM, label="teacher-forced", error_kw=ekw)
    ax.bar(x + w/2, fr, w, yerr=fr_sd, color=BLUE, label="free rollout", error_kw=ekw)
    for xi, (t, ts, f, fs, m) in enumerate(zip(tf, tf_sd, fr, fr_sd, mult)):
        ax.text(xi - w/2, t + ts + 0.02, f"{t:.2f}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
        ax.text(xi + w/2, f + fs + 0.02, f"{f:.2f}", ha="center", va="bottom", fontsize=8.5, color=BLUE)
        ax.text(xi, max(t + ts, f + fs) + 0.13, m, ha="center", fontweight="bold", color=INK, fontsize=11.5)
    ax.axvline(2.5, color=GRID, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, 1.62)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7); ax.set_axisbelow(True)
axes[0].set_ylabel("rollout error  (nMSE, ↓)")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()
for out in [HERE / "fig2_free_rollout_2models", PAPER_FIG / "fig2_free_rollout_2models"]:
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print("wrote", out)
plt.close(fig)
