"""Appendix figure: dinowm (frozen DINOv2) injection scan, paper style.
Recomputes every cell from rollout logs (means over available seeds; dagger = 3 seeds).
Writes fig17_dinowm_scan.pdf into aaai_paper/figures/ and paper/figures/.
Source: /data1/likun-share/junjxu/runs/dinowm/rollout_dinowm_*.log
"""
#
# ---------------------------------------------------------------------------
# DATA SOURCE (updated 2026-07-19)
#   Use  /home/qlib/am/wm/raw_data/runs/   -- the in-repo copy of every log.
#   The /data1/likun-share/junjxu/runs/ paths quoted below are the ORIGINAL
#   locations; that server is being retired, so treat them as provenance
#   notes, not as a path to read from right now.
#   NOTE: seed naming differs by seed (see raw_data/README.md #0) --
#     seed 3072 (default) has NO _s suffix and lives in runs/structdyn_eval/;
#     seeds 1234 / 42 carry _s1234 / _s42 and live in runs/aaai_p0/.
#   Also: a run with no pwN in its name is the default slot weight w=1.
# ---------------------------------------------------------------------------

import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# In-repo copy of the logs. Derived from this file's location rather than
# hard-coded: the path used to be an absolute one on a machine that no longer
# exists, which made every figure here unreproducible from a fresh clone.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]                      # figures -> aaai_paper -> 7-11 -> reports -> repo
LOG = str(REPO / "raw_data" / "runs" / "dinowm")
PAPER_FIG = HERE.parent / "paper" / "figures"
PART = {"um": "both-OOD", "par": "r/m-OOD", "col": "both-OOD"}


def nmse(tag, part):
    f = f"{LOG}/rollout_{tag}.log"
    if not os.path.exists(f):
        return None
    pat = re.compile(r"^\s*" + re.escape(part) + r"\s+n=\s*\d+\s+cos=[+\-\d.]+\s+nMSE=([\d.]+)")
    for line in open(f):
        m = pat.match(line)
        if m:
            return float(m.group(1))
    return None


def cell(dom, key):
    tags = [f"dinowm_{dom}_{key}", f"dinowm_{dom}_{key}_s1234", f"dinowm_{dom}_{key}_s42"]
    v = [nmse(t, PART[dom]) for t in tags]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


ARMS = [
    ("structpos_plain", "[slot] structpos"),
    ("structpos_pw30", "[slot] +reweight"),
    ("posvel_pw30", "[slot] +velocity"),
    ("probeF2", "[probe] probe"),
    ("probe_structpos_pw30", "[probe] +slot"),
    ("dyn_mlp", "[dyn] free MLP"),
    ("grounded_const", "[dyn] strict a=g"),
    ("cons", "[cons] consistency"),
    ("labelfree_const", "[free] label-free"),
    ("grounded2_const", "[free] grounded"),
]
DOMS = ["um", "par", "col"]
DOMLAB = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]

base = {d: cell(d, "fr_s3072")[0] for d in DOMS}  # placeholder; recompute as 3-seed below
for d in DOMS:
    v = [nmse(f"dinowm_{d}_fr_s{s}", PART[d]) for s in [3072, 1234, 42]]
    base[d] = float(np.mean([x for x in v if x is not None]))

M = np.full((len(ARMS), 3), np.nan)
N = np.zeros((len(ARMS), 3), int)
for i, (k, _) in enumerate(ARMS):
    for j, d in enumerate(DOMS):
        m, n = cell(d, k)
        if m is not None:
            M[i, j] = m / base[d]
            N[i, j] = n

if np.all(np.isnan(M)):
    raise SystemExit(f"[abort] no run data parsed from {LOG}; refusing to overwrite the "
                     "existing figure with an empty one.")

fig, ax = plt.subplots(figsize=(5.6, 6.6))
norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.4)
_cmap = LinearSegmentedColormap.from_list("bwv", ["#0072B2", "#f7f7f7", "#D55E00"])
ax.imshow(M, cmap=_cmap, norm=norm, aspect="auto")
for i in range(len(ARMS)):
    for j in range(3):
        if np.isnan(M[i, j]):
            ax.text(j, i, "--", ha="center", va="center")
            continue
        dag = r"$^\dagger$" if N[i, j] >= 3 else ""
        raw = M[i, j] * base[DOMS[j]]
        ax.text(j, i, f"{M[i,j]:.2f}×{dag}\n({raw:.3f})", ha="center", va="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(DOMLAB, fontsize=9)
ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([lab for _, lab in ARMS], fontsize=8.5)
ax.set_title("Injection scan on the frozen-DINOv2 backbone\n(nMSE / free-rollout baseline; vermillion = worse)",
             fontsize=10)
fig.tight_layout()
for out in [HERE / "fig17_dinowm_scan", PAPER_FIG / "fig17_dinowm_scan"]:
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print("wrote", out)
worse = int(np.nansum(M > 1.05)); par = int(np.nansum((M >= 0.95) & (M <= 1.05))); bet = int(np.nansum(M < 0.95))
print(f"cells: {worse} worse / {par} parity / {bet} better")
