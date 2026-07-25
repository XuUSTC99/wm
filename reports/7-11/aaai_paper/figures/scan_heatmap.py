"""One drawing routine for both injection-scan heatmaps (paper fig 2 + fig 6).

The two figures previously used different colormaps, norms, fonts, annotation
schemes (hatching vs daggers) and verdict rules (seed separation vs a ratio
threshold). Everything style- or verdict-related now lives here so the pair
cannot drift apart again.

Verdict rule (shared): a cell is WORSE if every draw exceeds every baseline
draw, BETTER if every draw is below every baseline draw, otherwise within
noise. A degree marker denotes within noise; unmarked cells are separated in
every draw.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import INK, SCAN_CMAP

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

COLS = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]
NORM = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.7)


def _ink_or_white(rgba):
    r, g, b = rgba[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def verdicts(cell_draws, base_draws):
    """-> (ratio, mean, n, worse_sep, better_sep) for one cell."""
    d, bd = cell_draws, base_draws
    mu = float(np.mean(d))
    return (mu / float(np.mean(bd)), mu, len(d),
            min(d) > max(bd), max(d) < min(bd))


def draw(rows, get_draws, get_base, outs, cbar_label="nMSE / baseline (seed means)",
         annotation_newline=True, annotation_fontsize=5.9,
         tick_fontsize=6.2, cbar_label_fontsize=6.0,
         cbar_tick_fontsize=5.8, cbar_orientation="vertical"):
    """rows: [(row_key, label)] resolved by get_draws(row_key, dom_index).

    get_base(dom_index) -> baseline draws. Writes .pdf (+ .png for the first
    out) and returns the verdict tally.
    """
    R = len(rows)
    ratio = np.full((R, 3), np.nan)
    mean = np.full_like(ratio, np.nan)
    nseed = np.zeros((R, 3), int)
    wsep = np.zeros((R, 3), bool)
    bsep = np.zeros((R, 3), bool)
    for i, (key, _) in enumerate(rows):
        for j in range(3):
            d, bd = get_draws(key, j), get_base(j)
            if not d:
                continue
            ratio[i, j], mean[i, j], nseed[i, j], wsep[i, j], bsep[i, j] = verdicts(d, bd)

    fig_style.apply(base=8.0)
    fig, ax = plt.subplots(figsize=(3.7, 4.5))
    cmap = SCAN_CMAP.copy()
    cmap.set_bad("#E8E8E8")
    im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, norm=NORM, aspect="auto")

    for i in range(R):
        for j in range(3):
            if nseed[i, j] == 0:
                ax.text(j, i, "no data", ha="center", va="center", fontsize=5.6, color="#888")
                continue
            col = _ink_or_white(SCAN_CMAP(NORM(ratio[i, j])))
            noise_marker = "°" if not (wsep[i, j] or bsep[i, j]) else ""
            separator = "\n" if annotation_newline else " "
            ax.text(j, i,
                    f"{ratio[i, j]:.2f}×{noise_marker}{separator}({mean[i, j]:.3f})",
                    ha="center", va="center", fontsize=annotation_fontsize, color=col)

    ax.set_xticks(range(3))
    ax.set_xticklabels(COLS, fontsize=tick_fontsize)
    ax.set_yticks(range(R))
    ax.set_yticklabels([lab for _, lab in rows], fontsize=tick_fontsize)
    ax.tick_params(length=0, pad=2.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if cbar_orientation == "horizontal":
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.045, pad=0.10, aspect=35)
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize, length=2, width=0.6)
    cb.outline.set_visible(False)

    for k, out in enumerate(outs):
        fig.savefig(f"{out}.pdf")
        if k == 0:
            fig.savefig(f"{out}.png")
        print("wrote", out)
    plt.close(fig)

    ok = nseed > 0
    tally = dict(worse=int(wsep.sum()), noise=int((ok & ~wsep & ~bsep).sum()),
                 better=int(bsep.sum()))
    print(f"verdicts: {tally['worse']} worse / {tally['noise']} within-noise / "
          f"{tally['better']} better  ({tally['worse']+tally['noise']} of {int(ok.sum())} at or below)")
    return tally
