"""Figure 2 -- the 10 variant x 3 domain injection scan, multi-seed edition.

Supersedes regen_fig16_scan.py, which hard-coded a single seed-3072 value in 26
of the 30 cells and divided them by a single-seed baseline. Both halves of that
ratio are now three-seed means read straight from the rollout logs by
scan_resolve.py -- nothing in this file is a transcribed number.

Two things the single-seed figure could not say, and this one does:

  * The baseline denominator moves a lot between seeds (collision: 0.393 /
    0.495 / 0.548). The old figure divided by 0.393, the luckiest draw, which
    inflated the whole collision column.
  * A ratio above 1.0 is only evidence of harm if the seeds actually separate.
    Cells whose seed range overlaps the baseline's are drawn hatched and are
    reported as "within noise", not as harm.

Run scan_resolve.py first to check coverage; cells still short of three draws
are drawn grey and labelled with their n.
"""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import INK, SCAN_CMAP
from scan_resolve import ARMS, DOMS, CELLS, BASELINE, resolve

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

fig_style.apply(base=8.0)

DOMCOL = {"uniform": "uniform\n(both-OOD)",
          "parabola": "parabola\n(r/m-OOD)",
          "collision": "collision\n(both-OOD)"}

# ---- gather ----------------------------------------------------------------
base_draws = {d: resolve(BASELINE[d], d) for d in DOMS}
base_mean = {d: float(np.mean(v)) for d, v in base_draws.items()}

ratio = np.full((len(ARMS), len(DOMS)), np.nan)
value = np.full_like(ratio, np.nan)
nseed = np.zeros_like(ratio, dtype=int)
# Separation is directional and must be tracked BOTH ways: a cell can clear the
# baseline by being entirely above it (worse) or entirely below it (better).
# Testing only the "worse" direction would leave a genuine gain classified as
# non-separated, and hatch the one cell the figure exists to single out.
worse_sep = np.zeros_like(ratio, dtype=bool)
better_sep = np.zeros_like(ratio, dtype=bool)

for i, arm in enumerate(ARMS):
    for j, dom in enumerate(DOMS):
        draws = resolve(CELLS[(arm, dom)], dom)
        nseed[i, j] = len(draws)
        if not draws:
            continue
        value[i, j] = float(np.mean(draws))
        ratio[i, j] = value[i, j] / base_mean[dom]
        worse_sep[i, j] = min(draws) > max(base_draws[dom])
        better_sep[i, j] = max(draws) < min(base_draws[dom])

# ---- draw ------------------------------------------------------------------
norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.7)
fig, ax = plt.subplots(figsize=(3.7, 4.9))
masked = np.ma.masked_invalid(ratio)
cmap = SCAN_CMAP.copy()
cmap.set_bad("#E8E8E8")
im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")


def ink_or_white(rgba):
    r, g, b = rgba[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


for i in range(len(ARMS)):
    for j in range(len(DOMS)):
        n = nseed[i, j]
        if n == 0:
            ax.text(j, i, "no data", ha="center", va="center", fontsize=5.6, color="#888")
            continue
        col = ink_or_white(SCAN_CMAP(norm(ratio[i, j])))
        tag = "" if n >= 3 else f"\nn={n}"
        ax.text(j, i, f"{ratio[i, j]:.2f}×\n({value[i, j]:.3f}){tag}",
                ha="center", va="center", fontsize=5.9, color=col)
        # Hatch every cell whose three seeds do NOT separate from the
        # baseline's -- in EITHER direction. A ratio below 1.0 that fails to
        # separate (e.g. probe+slot on parabola, 0.81x) is not a gain; leaving
        # it un-hatched would read as a second blue win beside the one real
        # one. Only fully separated cells stay solid, so the single solid-blue
        # cell (velocity+slot on parabola) is the sole genuine gain.
        if n >= 3 and not (worse_sep[i, j] or better_sep[i, j]):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       hatch="///", edgecolor=col,
                                       linewidth=0.0, alpha=0.45))

ax.set_xticks(range(len(DOMS)))
ax.set_xticklabels([DOMCOL[d] for d in DOMS], fontsize=6.2)
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels(ARMS, fontsize=6.2)
ax.tick_params(length=0, pad=2.5)
for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("nMSE / baseline  (both three-seed means)", fontsize=6.0)
cb.ax.tick_params(labelsize=5.8, length=2, width=0.6)
cb.outline.set_visible(False)

out = pathlib.Path(__file__).resolve().parent / "fig16_scan_paper_v2"
fig.savefig(str(out) + ".pdf")
fig.savefig(str(out) + ".png")
for p in [pathlib.Path(__file__).resolve().parent.parent / "paper" / "figures" / "fig16_scan_paper_v2"]:
    fig.savefig(str(p) + ".pdf")
print("wrote", out)

# ---- console ledger: what changed against the published single-seed figure --
OLD = np.array([[0.183, 0.156, 0.651], [0.114, 0.160, 0.596], [0.207, 0.093, 0.621],
                [0.167, 0.115, 0.647], [0.125, 0.127, 0.607], [0.155, 0.178, 0.560],
                [0.206, 0.173, 0.559], [0.151, 0.147, 0.640], [0.171, 0.172, 0.653],
                [0.166, 0.156, 0.524]])
OLD_SEED3 = {(1, 0): 0.132, (3, 1): 0.137, (4, 0): 0.141, (2, 1): 0.096}
for (i, j), v in OLD_SEED3.items():
    OLD[i, j] = v
OLD_BASE = np.array([0.131, 0.127, 0.393])

print(f"\nbaseline denominator: "
      + "  ".join(f"{d} {OLD_BASE[j]:.3f} -> {base_mean[d]:.3f}" for j, d in enumerate(DOMS)))
print(f"\n{'cell':<34} {'old':>10} {'new':>10}  verdict")
print("-" * 72)
worse = parity = better = short = 0
for i, arm in enumerate(ARMS):
    for j, dom in enumerate(DOMS):
        if nseed[i, j] == 0:
            short += 1
            continue
        o = OLD[i, j] / OLD_BASE[j]
        n_ = ratio[i, j]
        draws = resolve(CELLS[(arm, dom)], dom)
        if nseed[i, j] < 3:
            v = f"n={nseed[i,j]} only"
            short += 1
        elif worse_sep[i, j]:
            # every draw above every baseline draw
            v = "worse (seeds separate)"; worse += 1
        elif better_sep[i, j]:
            # the mirror condition -- every draw below every baseline draw.
            # Anything weaker than full separation is noise at n=3, in both
            # directions; an asymmetric test would manufacture "gains".
            v = "BETTER (seeds separate)"; better += 1
        else:
            v = "within noise"; parity += 1
        flip = "  <-- flipped" if (o > 1.0) != (n_ > 1.0) else ""
        print(f"{arm+' / '+dom:<34} {o:>9.2f}× {n_:>9.2f}×  {v}{flip}")
print("-" * 72)
print(f"worse {worse}   within-noise {parity}   better {better}   incomplete {short}")
