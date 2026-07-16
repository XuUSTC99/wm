"""dinowm 30-cell physics-injection heatmap (cross-model analog of LeWM Fig 16).
Reads grid30.json produced by the collection step. Color = nMSE/baseline ratio
(red worse, white parity, green better). Mirrors the LeWM figure's layout so the
two can sit side by side in the paper/appendix.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

d = json.load(open("/data1/likun-share/junjxu/runs/dinowm/grid30.json"))
base, grid = d["base"], d["grid"]
arms = list(grid.keys())          # 10 arms, in inserted order
doms = ["um", "par", "col"]
domlab = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]

M = np.array([[grid[a][dm]["ratio"] or np.nan for dm in doms] for a in arms])

fig, ax = plt.subplots(figsize=(5.2, 6.4))
norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.6)
im = ax.imshow(M, cmap="RdYlGn_r", norm=norm, aspect="auto")

for i, a in enumerate(arms):
    for j, dm in enumerate(doms):
        r = grid[a][dm]["ratio"]; nm = grid[a][dm]["nmse"]
        if r is None:
            ax.text(j, i, "--", ha="center", va="center", fontsize=8)
        else:
            mark = " " + ("✓" if r < 0.95 else "")
            ax.text(j, i, f"{r:.2f}x{mark}\n({nm:.3f})", ha="center", va="center",
                    fontsize=7.5, color="black")

ax.set_xticks(range(3)); ax.set_xticklabels(domlab, fontsize=9)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=8.5)
ax.set_title("dinowm (frozen DINOv2): physics injection\nnMSE / baseline  (red=worse, green=better)",
             fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("nMSE / baseline", fontsize=8)
n_worse = int(np.sum(M > 1.05)); n_par = int(np.sum((M >= 0.95) & (M <= 1.05)))
n_better = int(np.sum(M < 0.95))
fig.text(0.5, 0.005,
         f"{n_worse} worse / {n_par} parity / {n_better} better  "
         f"(all 3 'better' cells are uniform, shown by shuffle-control to be regularization)",
         ha="center", fontsize=7.5)
plt.tight_layout(rect=[0, 0.03, 1, 1])
out = "/home/likun-share/junjxu/wm/reports/7-11/aaai_paper/figures/dinowm_injection_heatmap.png"
plt.savefig(out, dpi=140); plt.close()
print("saved", out)
print(f"cells: {n_worse} worse, {n_par} parity, {n_better} better (of 30)")
