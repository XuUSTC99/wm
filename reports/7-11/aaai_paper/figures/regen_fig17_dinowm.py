import pathlib
"""Restore fig17_dinowm_scan from recovered values, with CVD-safe blue-white-vermillion map.

Values recovered from the pre-recolor render of the original figure (all 30 cells + dagger
markers). Ratios and raw nMSE are hardcoded exactly as the original displayed them, so this
reproduces the figure without needing the run logs on /data1.
"""
import pathlib
_HERE = pathlib.Path(__file__).resolve().parent
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
fig_style.apply(base=8.0)

ARMS = ["[slot] structpos", "[slot] +reweight", "[slot] +velocity",
        "[probe] probe", "[probe] +slot", "[dyn] free MLP", "[dyn] strict a=g",
        "[cons] consistency", "[free] label-free", "[free] grounded"]
DOMLAB = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]

RATIO = np.array([
    [1.17, 1.02, 0.97],
    [0.81, 1.05, 1.01],
    [0.93, 1.01, 0.99],
    [1.02, 1.03, 1.03],
    [0.91, 1.10, 1.02],
    [0.99, 1.04, 1.11],
    [1.08, 1.16, 1.27],
    [1.00, 1.01, 0.96],
    [0.96, 1.00, 1.07],
    [1.11, 1.13, 1.27],
])
RAW = np.array([
    [0.502, 0.229, 0.463],
    [0.347, 0.236, 0.485],
    [0.398, 0.227, 0.473],
    [0.434, 0.231, 0.493],
    [0.390, 0.247, 0.487],
    [0.423, 0.234, 0.530],
    [0.459, 0.261, 0.610],
    [0.428, 0.227, 0.461],
    [0.411, 0.224, 0.512],
    [0.473, 0.254, 0.607],
])
# every row is three-seed except the last ([free] grounded)
DAGGER = np.ones_like(RATIO, dtype=bool); DAGGER[9, :] = False

cmap = fig_style.SCAN_CMAP
norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.4)

fig, ax = plt.subplots(figsize=(5.6, 6.6))
ax.imshow(RATIO, cmap=cmap, norm=norm, aspect="auto")
for i in range(len(ARMS)):
    for j in range(3):
        r = RATIO[i, j]
        dag = r"$^\dagger$" if DAGGER[i, j] else ""
        col = "white" if (r <= 0.85 or r >= 1.24) else "#1a1a1a"
        ax.text(j, i, f"{r:.2f}×{dag}\n({RAW[i,j]:.3f})", ha="center", va="center",
                fontsize=8, color=col)
ax.set_xticks(range(3)); ax.set_xticklabels(DOMLAB, fontsize=9)
ax.set_yticks(range(len(ARMS))); ax.set_yticklabels(ARMS, fontsize=8.5)
ax.set_title("Injection scan on the frozen-DINOv2 backbone\n"
             "(nMSE / free-rollout baseline; warm = worse)", fontsize=10)
fig.tight_layout()
for out in [str(_HERE / "fig17_dinowm_scan"),
            str(_HERE.parent / "paper" / "figures" / "fig17_dinowm_scan")]:
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print("restored", out)
worse = int((RATIO > 1.05).sum()); par = int(((RATIO >= 0.95) & (RATIO <= 1.05)).sum())
bet = int((RATIO < 0.95).sum())
print(f"cells: {worse} worse / {par} parity / {bet} better  (caption claims 10/17/3)")
