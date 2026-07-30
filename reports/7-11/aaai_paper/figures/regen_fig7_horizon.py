"""Appendix figure -- Physion++ training horizon ladder.

Longer training rollout keeps improving long-horizon accuracy, with no plateau
found: np28+scale reaches 0.014 at horizon 64, about 1/19 of the np8 baseline.

Data source: physionpp/eval_pp_fr{,_np20,_np20sc,_np28sc,_scnp16}_e20*.log
Extracted values are unchanged from the original storyline_figures.py block.

Split out of storyline_figures.py so the paper figure can be refreshed without
running that script, which also writes fig16_scan_paper and
fig1_thesis_presence_not_use and would clobber their three-seed versions.

Sized for a single column: the original was 7.6in wide but placed at
0.95\\linewidth, so its 9pt labels landed at roughly 3.8pt on the page.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, CAT

import matplotlib.pyplot as plt

fig_style.apply(base=9.0)

hz = np.array([16, 32, 64])
series = {
    "np8 (baseline)": ([0.0164, 0.1404, 0.2797], SIENNA, "o"),
    "np20":           ([0.0215, 0.0326, 0.2203], CAT[1], "s"),
    "np20 + scale":   ([0.0115, 0.0236, 0.0866], CAT[2], "^"),
    "np28 + scale":   ([0.0076, 0.0101, 0.0144], BLUE,   "D"),
}

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for name, (vals, c, mk) in series.items():
    ax.plot(hz, vals, color=c, marker=mk, ms=3.2, lw=1.6, label=name,
            mec="white", mew=0.5)

ax.set_yscale("log")
ax.set_xticks(hz)
# Symmetric 3-step margin around the 16/32/64 sweep. No callout to make room
# for: the 0.014 endpoint and the 19x figure are already in the caption, in
# §4.4, and in the appendix text, so printing them here too was a fourth copy.
ax.set_xlim(13, 67)
ax.set_ylim(0.0055, 0.46)   # headroom so the np8 curve stops short of the frame
ax.set_xlabel("rollout horizon (steps)", fontsize=9.0)
ax.set_ylabel("nMSE  (log, $\\downarrow$)", fontsize=9.0)
ax.legend(fontsize=9.0, loc="upper left", bbox_to_anchor=(0.005, 1.0),
          handlelength=1.6, labelspacing=0.3)
ax.tick_params(labelsize=9.0, length=2.5, width=0.7)
ax.grid(True, color="#EAE8E4", lw=0.7, which="both")
ax.set_axisbelow(True)
fig_style.despine(ax)

fig.tight_layout(pad=0.25)
here = pathlib.Path(__file__).resolve().parent
for out in [here / "fig7_realdata_num_preds",
            here.parent / "paper" / "figures" / "fig7_realdata_num_preds"]:
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png")
    print("wrote", out)
