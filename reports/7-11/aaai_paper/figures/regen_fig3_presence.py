"""Figure 3 -- decodable but not load-bearing, all three domains in ONE axes.

Colour encodes the protocol (sienna = teacher forcing, blue = free rollout,
matching Figure 4); line style encodes the domain. The probe values for the
three domains are drawn as a single band, which is the point: the state is
decodable to the same degree everywhere, while the teacher-forced rollouts
fall away from it by an amount that tracks dynamics complexity.

Data provenance
---------------
Rollout cosine: storyline_figures.py FIG 13 (h = 1/4/8/16/28); collision
matches the earlier single-domain figure exactly.
Probe: both-OOD REAL-embedding position rho, mean of pos0/pos1, from the
per-domain table in detail/why_physics_structure_fails.md.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, MUTED, RULE

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig_style.apply(base=8.0)

BAND = "#C7CBD1"   # neutral grey-blue: the probe band must not read as data

H = [1, 4, 8, 16, 28]
TF = {"uniform": [0.99, 0.96, 0.91, 0.84, 0.84],
      "parabola": [0.98, 0.88, 0.84, 0.52, 0.57],
      "collision": [0.99, 0.88, 0.64, 0.36, 0.24]}
FR = {"uniform": [0.99, 0.98, 0.97, 0.95, 0.95],
      "parabola": [0.98, 0.94, 0.94, 0.79, 0.93],
      "collision": [1.00, 0.98, 0.95, 0.76, 0.48]}
PROBE = {"uniform": (0.955 + 0.862) / 2, "parabola": (0.831 + 0.893) / 2,
         "collision": (0.889 + 0.796) / 2}
STYLE = {"uniform": ("-", "o"), "parabola": ("--", "s"), "collision": (":", "^")}

lo, hi = min(PROBE.values()), max(PROBE.values())

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axhspan(lo, hi, color=BAND, alpha=0.5, lw=0, zorder=1)

for dom, (ls, mk) in STYLE.items():
    ax.plot(H, TF[dom], color=SIENNA, ls=ls, marker=mk, ms=2.8, lw=1.6,
            mec="white", mew=0.5, zorder=4)
    ax.plot(H, FR[dom], color=BLUE, ls=ls, marker=mk, ms=2.8, lw=1.6,
            mec="white", mew=0.5, zorder=3)

key = [Line2D([], [], color=SIENNA, lw=2.0, label="teacher forcing"),
       Line2D([], [], color=BLUE, lw=2.0, label="free rollout"),
       Patch(facecolor=BAND, alpha=0.5, label="decodable (probe $\\rho$)"),
       Line2D([], [], color=MUTED, lw=1.2, ls="-",  label="uniform"),
       Line2D([], [], color=MUTED, lw=1.2, ls="--", label="parabola"),
       Line2D([], [], color=MUTED, lw=1.2, ls=":",  label="collision")]
ax.legend(handles=key, fontsize=6.1, loc="lower left", ncol=2, columnspacing=0.9,
          handlelength=1.8, borderpad=0.0, labelspacing=0.3, handletextpad=0.5)

ax.set_xlabel("rollout horizon (steps)", fontsize=7.6)
ax.set_ylabel("agreement with truth\n(latent cosine, $\\uparrow$)",
              fontsize=7.2, linespacing=1.3)
ax.set_xticks(H); ax.set_xlim(0.4, 29)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylim(0.10, 1.04)
ax.tick_params(labelsize=7.0, length=2.5, width=0.7)
ax.grid(True, axis="y", color="#EAE8E4", lw=0.7)
ax.set_axisbelow(True)
fig_style.despine(ax)

fig.tight_layout(pad=0.25)
here = pathlib.Path(__file__).resolve().parent
for out in [here / "fig1_thesis_presence_not_use",
            here.parent / "paper" / "figures" / "fig1_thesis_presence_not_use"]:
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png")
    print("wrote", out)
print(f"  probe band {lo:.3f}-{hi:.3f}; TF@h28 " +
      ", ".join(f"{d} {TF[d][-1]:.2f}" for d in TF))
