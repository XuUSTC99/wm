"""Figure 3 -- decodable, used, and still drifting. THREE small multiples, one
per domain (uniform / parabola / collision), single-column row.

Each panel: the shaded band is that domain's position decodability (probe rho on
the two coordinates pos0/pos1, both-OOD, from REAL frame embeddings) -- flat with
horizon, the "presence" ceiling. Two curves fall away from it: teacher forcing
(sienna) and free rollout (blue). The gap to the band grows with dynamics
complexity across the three panels; free rollout recovers much of it with no
physics added. Splitting the old single-axes version into three panels removes
the six-line pile-up and lets each domain's band sit at its own level.

Data provenance
---------------
Rollout cosine: storyline_figures.py FIG 13 (h = 1/4/8/16/28); collision matches
the earlier single-domain figure exactly.
Probe band: both-OOD REAL-embedding position rho, per coordinate (pos0, pos1),
from the per-domain table in detail/why_physics_structure_fails.md.

NOTE (why no Physion/Physion++ 4th panel): the matching experiment does not
exist there. Physion++ tells its long-horizon story in nMSE (bounded), has only
aggregate/per-scene cosine (not a per-horizon curve), and its probe number is
baseline *velocity* decodability (0.44) under the probe-injection variant -- not
a flat, high position-decodability band. A 4th panel would require new runs.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, MUTED

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig_style.apply(base=8.0)

BAND = "#C7CBD1"   # neutral grey-blue: the probe band must not read as data

H = [1, 4, 8, 16, 28]
DOMS = ["uniform", "parabola", "collision"]
TF = {"uniform": [0.99, 0.96, 0.91, 0.84, 0.84],
      "parabola": [0.98, 0.88, 0.84, 0.52, 0.57],
      "collision": [0.99, 0.88, 0.64, 0.36, 0.24]}
FR = {"uniform": [0.99, 0.98, 0.97, 0.95, 0.95],
      "parabola": [0.98, 0.94, 0.94, 0.79, 0.93],
      "collision": [1.00, 0.98, 0.95, 0.76, 0.48]}
# per-domain band = the two position coordinates' decodability rho (pos0, pos1)
PROBE = {"uniform": (0.862, 0.955), "parabola": (0.831, 0.893),
         "collision": (0.796, 0.889)}

fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.72), sharey=True)

for ax, dom in zip(axes, DOMS):
    lo, hi = PROBE[dom]
    ax.axhspan(lo, hi, color=BAND, alpha=0.65, lw=0, zorder=1)
    ax.plot(H, TF[dom], color=SIENNA, marker="o", ms=2.6, lw=1.4,
            mec="white", mew=0.4, zorder=4)
    ax.plot(H, FR[dom], color=BLUE, marker="o", ms=2.6, lw=1.4,
            mec="white", mew=0.4, zorder=3)
    ax.set_title(dom, fontsize=7.4, pad=2.0)
    ax.set_xticks([1, 8, 28])
    ax.set_xlim(0.4, 29)
    ax.set_ylim(0.10, 1.04)
    ax.tick_params(labelsize=6.2, length=2.2, width=0.6)
    ax.grid(True, axis="y", color="#EAE8E4", lw=0.6)
    ax.set_axisbelow(True)
    fig_style.despine(ax)

axes[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
axes[0].set_ylabel("agreement with truth\n(latent cosine, $\\uparrow$)",
                   fontsize=6.8, linespacing=1.25)
fig.supxlabel("rollout horizon (steps)", fontsize=7.2, y=0.14)

key = [Line2D([], [], color=SIENNA, marker="o", ms=3, lw=1.6, mec="white",
              mew=0.4, label="teacher forcing"),
       Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.6, mec="white",
              mew=0.4, label="free rollout"),
       Patch(facecolor=BAND, alpha=0.65, label="decodable (probe $\\rho$)")]
fig.legend(handles=key, fontsize=6.2, loc="lower center", ncol=3,
           columnspacing=1.2, handlelength=1.6, handletextpad=0.5,
           borderpad=0.0, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(pad=0.3, rect=(0, 0.10, 1, 1))
here = pathlib.Path(__file__).resolve().parent
for out in [here / "fig1_thesis_presence_not_use",
            here.parent / "paper" / "figures" / "fig1_thesis_presence_not_use"]:
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png")
    print("wrote", out)
print("  per-domain bands:", {d: PROBE[d] for d in DOMS})
print("  TF@h28:", ", ".join(f"{d} {TF[d][-1]:.2f}" for d in DOMS))
