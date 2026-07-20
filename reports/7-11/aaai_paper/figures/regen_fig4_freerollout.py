"""Figure 4 -- teacher forcing vs free rollout, both backbones on one axis.

Four domain groups; within each, LeWM (solid) and frozen-DINOv2 (hatched) each
contribute a teacher-forced / free-rollout pair. Fold-change is printed over
each pair, so the headline -- the protocol lever works on both backbones --
reads off one axis instead of requiring a left/right comparison between panels.

Data provenance
---------------
Three-seed means and sample std over seeds 3072/1234/42, extracted from
raw_data/runs/ by extract_fig5_seeds.py. Identical to fig2_two_models.py.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, MUTED

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig_style.apply(base=8.0)

doms = ["uniform", "parabola\n(r/m-OOD)", "collision", "Physion++\n(sim, h64)"]
LEWM_TF = [0.300, 0.443, 1.152, 1.174]; LEWM_TF_E = [0.007, 0.048, 0.048, 0.041]
LEWM_FR = [0.136, 0.122, 0.479, 0.141]; LEWM_FR_E = [0.008, 0.007, 0.079, 0.018]
LEWM_X = ["2.2×", "3.6×", "2.4×", "8.3×"]
DINO_TF = [0.594, 0.380, 0.803, 1.030]; DINO_TF_E = [0.078, 0.018, 0.041, 0.034]
DINO_FR = [0.427, 0.225, 0.479, 0.258]; DINO_FR_E = [0.047, 0.023, 0.011, 0.006]
DINO_X = ["1.4×", "1.7×", "1.7×", "4.0×"]

x = np.arange(len(doms)); w = 0.20
fig, ax = plt.subplots(figsize=(3.4, 2.45))
EB = dict(ecolor="#4A4A4A", capsize=1.6, elinewidth=0.7, capthick=0.7)

ax.bar(x - 1.5 * w, LEWM_TF, w, color=SIENNA, yerr=LEWM_TF_E, error_kw=EB)
ax.bar(x - 0.5 * w, LEWM_FR, w, color=BLUE, yerr=LEWM_FR_E, error_kw=EB)
ax.bar(x + 0.5 * w, DINO_TF, w, color=SIENNA, hatch="////", edgecolor="white",
       linewidth=0.0, yerr=DINO_TF_E, error_kw=EB)
ax.bar(x + 1.5 * w, DINO_FR, w, color=BLUE, hatch="////", edgecolor="white",
       linewidth=0.0, yerr=DINO_FR_E, error_kw=EB)

for xs, tf, fr, mult, e in [(x - w, LEWM_TF, LEWM_FR, LEWM_X, LEWM_TF_E),
                            (x + w, DINO_TF, DINO_FR, DINO_X, DINO_TF_E)]:
    for xi, t, f, m, ei in zip(xs, tf, fr, mult, e):
        ax.text(xi, max(t, f) + ei + 0.05, m, ha="center", fontsize=6.2, color=INK)

key = [Patch(facecolor=SIENNA, label="teacher forcing"),
       Patch(facecolor=BLUE, label="free rollout"),
       Patch(facecolor="0.68", edgecolor="0.45", label="LeWM (trainable ViT-tiny)"),
       Patch(facecolor="0.68", hatch="////", edgecolor="white",
             label="frozen DINOv2 + adapter")]
ax.legend(handles=key, fontsize=6.8, ncol=2, loc="upper left",
          columnspacing=1.0, handlelength=1.5, borderpad=0.2)

ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=6.4)
ax.set_ylabel("rollout error (nMSE, $\\downarrow$)", fontsize=7.4)
ax.set_ylim(0, 1.95)
ax.tick_params(labelsize=6.6, length=2.5, width=0.7)
ax.grid(True, axis="y", color="#EAE8E4", lw=0.7)
ax.set_axisbelow(True)
fig_style.despine(ax)

fig.tight_layout(pad=0.25)
here = pathlib.Path(__file__).resolve().parent
for out in [here / "fig2_free_rollout_2models",
            here.parent / "paper" / "figures" / "fig2_free_rollout_2models"]:
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png")
    print("wrote", out)
