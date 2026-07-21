"""Figure 2 -- the 10 variant x 3 domain injection scan.

Data provenance
---------------
`raw` holds the single-seed (3072) run for every cell. `seed3` then OVERRIDES
the four cells that were later re-run over seeds 3072/1234/42 with their
three-seed means; those are the cells the figure marks with a dagger. Read
both together -- `raw` alone is NOT what the figure shows, and three of those
four cells are exactly the ones whose apparent gain the extra seeds refuted.

Values verified against /home/qlib/am/wm/raw_data/runs/ (30/30 cells) and
against the shipped PDF (30/30) on 2026-07-20.

Grouping
--------
Separator lines encode the two-level structure of Table 1: heavy rules at the
three-group boundaries (state / evolution / label-free), light rules at the
mechanism boundaries inside a group.
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, PAPER, SCAN_CMAP

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

fig_style.apply(base=8.0)

arms = ["[slot] structpos", "[slot] +reweight (w=30)", "[slot] +velocity",
        "[probe] probe", "[probe] +slot", "[dyn] free MLP", "[dyn] strict a=g",
        "[cons] consistency", "[free] label-free", "[free] grounded"]
cols = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]
base = np.array([0.131, 0.127, 0.393])

raw = np.array([
    [0.183, 0.156, 0.651],
    [0.114, 0.160, 0.596],
    [0.207, 0.093, 0.621],
    [0.167, 0.115, 0.647],
    [0.125, 0.127, 0.607],
    [0.155, 0.178, 0.560],
    [0.206, 0.173, 0.559],
    [0.151, 0.147, 0.640],
    [0.171, 0.172, 0.653],
    [0.166, 0.156, 0.524],
])
disp = raw.copy()
seed3 = {(1, 0): 0.132, (3, 1): 0.137, (4, 0): 0.141, (2, 1): 0.096}
for (i, j), v in seed3.items():
    disp[i, j] = v

ratio = disp / base
parity3 = {(1, 0), (3, 1), (4, 0)}   # dagger: three-seed, lands at parity
gain = {(2, 1)}                      # dagger+check: the one seed-robust gain

norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.7)

fig, ax = plt.subplots(figsize=(3.45, 4.7))
im = ax.imshow(ratio, cmap=SCAN_CMAP, norm=norm, aspect="auto")


def ink_or_white(rgba):
    """Pick the legible text colour for a cell by its relative luminance."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else "white"


for i in range(len(arms)):
    for j in range(3):
        # Double dagger, not a check mark: Nimbus Roman has no U+2713 and
        # matplotlib would draw a hollow .notdef box without warning.
        mark = "†" if (i, j) in parity3 else ("‡" if (i, j) in gain else "")
        ax.text(j, i, f"{ratio[i, j]:.2f}×\n({disp[i, j]:.3f}){mark}",
                ha="center", va="center", fontsize=5.9,
                color=ink_or_white(SCAN_CMAP(norm(ratio[i, j]))))

ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=6.2)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=6.2)
ax.tick_params(length=0, pad=2.5)
for s in ax.spines.values():
    s.set_visible(False)

# Group boundaries: state | evolution | label-free, matching Table 1.
# A thin dark rule, not a thick white one -- white bands across a filled
# heatmap read as missing cells or a rendering seam rather than as structure.
# The mechanism level is left undrawn; the row labels already prefix it.
for y in (4.5, 7.5):
    ax.axhline(y, color=INK, lw=0.8, alpha=0.85)

cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("nMSE / baseline", fontsize=6.2)
cb.ax.tick_params(labelsize=5.8, length=2, width=0.6)
cb.outline.set_visible(False)

out = pathlib.Path(__file__).resolve().parent / "fig16_scan_paper"
fig.savefig(str(out) + ".pdf")
fig.savefig(str(out) + ".png")
print("wrote", out)
print("cells:", " ".join(f"{v:.3f}" for v in disp.ravel()))
