"""Rebuild fig8_lbr_ablation as an all-partition, all-domain dose-response panel.

Data source: the `lbr` dict in reports/7-11/aaai_paper/figures/storyline_figures.py
(3 domains x 4 partitions x 6 slot weights). The previous version plotted only a
slice of this; here every decision cell is visible, so "re-weighting reaches parity
in only 2 of 4 cells" can be read straight off the figure.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import INK, MUTED, CAT
import pathlib
_HERE = pathlib.Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt

fig_style.apply(base=10.5)
GRAY, ORANGE, SKY, BLUE = CAT
VERM = fig_style.SIENNA
GRID = "#EAE8E4"

W = [1, 3, 10, 30, 100, 300]
lbr = {
 "uniform":  {"base": {"ID":0.020,"r/m-OOD":0.173,"v-OOD":0.030,"both-OOD":0.131},
              "ID":[0.019,0.018,0.016,0.014,0.022,0.019], "r/m-OOD":[0.234,0.242,0.266,0.183,0.237,0.209],
              "v-OOD":[0.029,0.029,0.025,0.023,0.032,0.032], "both-OOD":[0.183,0.158,0.162,0.114,0.136,0.139]},
 "parabola": {"base": {"ID":0.012,"r/m-OOD":0.127,"v-OOD":0.148,"both-OOD":0.313},
              "ID":[0.014,0.013,0.015,0.015,0.015,0.019], "r/m-OOD":[0.151,0.144,0.164,0.160,0.122,0.094],
              "v-OOD":[0.117,0.168,0.121,0.128,0.102,0.124], "both-OOD":[0.321,0.309,0.312,0.341,0.286,0.225]},
 "collision":{"base": {"ID":0.379,"r/m-OOD":0.183,"v-OOD":0.609,"both-OOD":0.393},
              "ID":[0.435,0.445,0.412,0.453,0.444,0.450], "r/m-OOD":[0.254,0.248,0.237,0.274,0.226,0.252],
              "v-OOD":[0.659,0.537,0.608,0.618,0.496,0.801], "both-OOD":[0.590,0.569,0.595,0.596,0.610,0.693]},
}
PARTS = [("ID", GRAY, "."), ("r/m-OOD", ORANGE, "s"), ("v-OOD", SKY, "^"), ("both-OOD", BLUE, "o")]
# the four cells the paper judges on (domain, partition) -> reached parity?
JUDGED = {("uniform","both-OOD"), ("uniform","r/m-OOD"),
          ("parabola","r/m-OOD"), ("collision","both-OOD")}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
for ax, dom in zip(axes, ["uniform", "parabola", "collision"]):
    d = lbr[dom]
    ax.axhline(1.0, color=INK, ls="--", lw=1.0, zorder=1)
    for part, c, mk in PARTS:
        r = np.array(d[part]) / d["base"][part]
        judged = (dom, part) in JUDGED
        ax.plot(W, r, color=c, marker=mk, ms=4.5, lw=2.0 if judged else 1.1,
                alpha=1.0 if judged else 0.55, ls="-" if judged else ":",
                label=part if ax is axes[0] else None, zorder=3 if judged else 2)
    ax.set_xscale("log"); ax.set_xticks(W); ax.set_xticklabels(W, fontsize=10.5)
    ax.set_title(dom, fontsize=10.5, fontweight="bold", pad=4)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.tick_params(labelsize=10.5)
axes[0].set_ylim(0.55, 1.85)
fig.supxlabel("slot weight $w$ (log scale)", fontsize=10.5, y=0.04)
fig.supylabel("nMSE / baseline", fontsize=10.5, x=0.025)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=10.5, loc="upper center",
           ncol=4, columnspacing=1.4, handlelength=1.8,
           bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.76, wspace=0.10)
for out in [str(_HERE / "fig8_lbr_ablation"),
            str(_HERE.parent / "paper" / "figures" / "fig8_lbr_ablation")]:
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    print("wrote", out)

# sanity: which judged cells reach parity (ratio <= 1.02) at any weight?
for dom, part in sorted(JUDGED):
    r = np.array(lbr[dom][part]) / lbr[dom]["base"][part]
    print(f"  {dom:9s} {part:9s} min ratio {r.min():.3f} @ w={W[int(r.argmin())]}"
          f"  -> {'parity' if r.min() <= 1.02 else 'never'}")
