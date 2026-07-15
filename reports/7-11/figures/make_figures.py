"""Figures for 7-11 LBR ablation + PIWM baseline.
Run: le-wm/.venv/bin/python reports/7-11/figures/make_figures.py

All numbers are inlined below with their raw-data source paths so the figures are
reproducible and auditable. Palette = Okabe-Ito (CVD-safe by construction).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
# Okabe-Ito colorblind-safe categorical palette (fixed order)
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

# ============================================================================
# FIG 1 — LBR pos_weight ablation.
# Source logs: /data1/likun-share/junjxu/runs/structdyn_eval/rollout_{uniform_motion,parabola,collision}_structpos_fr_pw*_id1k.log
#              + seed runs *_s{1234,42}_id1k.log ; baselines /data1/.../aaai_p0/rollout_*_baseline_fr_s*.log
# ============================================================================
pw = np.array([1, 3, 10, 30, 100, 300])
# uniform, both-OOD nMSE (mean over seeds where available); '' seeds = 3072 single
um_both = np.array([0.162, 0.158, 0.164, 0.132, 0.140, 0.139])
um_both_err = np.array([0.017, 0.0, 0.025, 0.014, 0.008, 0.0])
um_rm = np.array([0.235, 0.242, 0.253, 0.230, 0.256, 0.209])
um_rm_err = np.array([0.001, 0.0, 0.009, 0.038, 0.056, 0.0])
UM_BOTH_BASE, UM_BOTH_BASE_E = 0.136, 0.007
UM_RM_BASE, UM_RM_BASE_E = 0.140, 0.025
# parabola, r/m-OOD nMSE (both-OOD unusable: h28 blowup). pw100/300 3-seed.
par_rm = np.array([0.151, 0.144, 0.164, 0.160, 0.138, 0.106])
par_rm_err = np.array([0.0, 0.0, 0.0, 0.0, 0.026, 0.014])
PAR_RM_BASE, PAR_RM_BASE_E = 0.122, 0.007
# collision, both-OOD nMSE (single seed; effect >> noise)
col_both = np.array([0.590, 0.569, 0.595, 0.596, 0.610, 0.693])
COL_BOTH_BASE = 0.393

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.0))

# -- Panel A: uniform, both-OOD recovers to parity but r/m-OOD never does --
axL.axhspan(UM_BOTH_BASE - UM_BOTH_BASE_E, UM_BOTH_BASE + UM_BOTH_BASE_E, color=BLUE, alpha=0.12, lw=0)
axL.axhspan(UM_RM_BASE - UM_RM_BASE_E, UM_RM_BASE + UM_RM_BASE_E, color=ORANGE, alpha=0.12, lw=0)
axL.errorbar(pw, um_both, yerr=um_both_err, color=BLUE, marker="o", ms=6, lw=2, capsize=3, label="both-OOD")
axL.errorbar(pw, um_rm, yerr=um_rm_err, color=ORANGE, marker="s", ms=6, lw=2, capsize=3, label="r/m-OOD")
axL.axhline(UM_BOTH_BASE, color=BLUE, ls=":", lw=1.2)
axL.axhline(UM_RM_BASE, color=ORANGE, ls=":", lw=1.2)
axL.text(300, UM_BOTH_BASE - 0.006, "free-rollout baseline", color=BLUE, ha="right", va="top", fontsize=8)
axL.set_xscale("log"); axL.set_xticks(pw); axL.set_xticklabels(pw)
axL.set_xlabel("pos_weight $\\lambda$ (log)"); axL.set_ylabel("latent nMSE  (↓ better)")
axL.set_title("(a) uniform: both-OOD → parity, r/m-OOD never", fontsize=10, color=INK)
axL.legend(frameon=False, fontsize=9); axL.grid(True, color=GRID, lw=0.6, alpha=0.7); axL.set_axisbelow(True)
axL.annotate("harm", (1, 0.162), (1.4, 0.20), color=MUTED, fontsize=8, arrowprops=dict(arrowstyle="->", color=MUTED))
# Explicit ylim: autoscale only saw the data, so the "parity" callout below the
# curve fell outside the axes and was clipped.
axL.set_ylim(0.088, 0.325)
axL.annotate("parity", (30, 0.132 - 0.017), (30, 0.101), color=MUTED, fontsize=8, ha="center",
             va="top", arrowprops=dict(arrowstyle="->", color=MUTED))

# -- Panel B: three-domain ratio-to-baseline (transition shifts right w/ difficulty) --
axR.axhline(1.0, color=MUTED, ls="--", lw=1.2)
axR.text(1.05, 1.005, "parity (= baseline)", color=MUTED, fontsize=8, va="bottom")
axR.plot(pw, um_both / UM_BOTH_BASE, color=BLUE, marker="o", ms=6, lw=2, label="uniform (both-OOD)")
axR.plot(pw, par_rm / PAR_RM_BASE, color=GREEN, marker="^", ms=6, lw=2, label="parabola (r/m-OOD)")
axR.plot(pw, col_both / COL_BOTH_BASE, color=VERM, marker="D", ms=6, lw=2, label="collision (both-OOD)")
axR.set_xscale("log"); axR.set_xticks(pw); axR.set_xticklabels(pw)
axR.set_xlabel("pos_weight $\\lambda$ (log)"); axR.set_ylabel("nMSE / baseline nMSE")
axR.set_title("(b) recovery point shifts right with domain difficulty", fontsize=10, color=INK)
axR.legend(frameon=False, fontsize=9, loc="upper left"); axR.grid(True, color=GRID, lw=0.6, alpha=0.7); axR.set_axisbelow(True)
# Was drawn AT the last data point (1.76) with va="bottom", i.e. pushed off the top
# of the axes. Park it in the empty band between the collision and parabola curves.
axR.set_ylim(0.82, 1.88)
axR.text(55, 1.40, "collision:\nnever recovers", color=VERM, fontsize=8, ha="center", va="center")

fig.suptitle("LBR (pos_weight up-weighting) restores parity in smooth domains only — never a net gain",
             fontsize=11, y=1.02, color=INK)
fig.savefig(OUT / "fig1_lbr_ablation.png"); fig.savefig(OUT / "fig1_lbr_ablation.pdf")
print("wrote fig1_lbr_ablation.{png,pdf}")

# ============================================================================
# FIG 2 — PIWM (official port) vs LeWM free-rollout: rolled-out position rho by partition.
# Source: /data1/likun-share/junjxu/runs/piwm_baseline/eval_{uniform_motion,parabola}_d0.json
#         /data1/likun-share/junjxu/runs/aaai_p0/rollout_{uniform,parabola}_baseline_fr_s42.log (PRED pos rho)
# rho = mean of the 2 position dims.
# ============================================================================
parts = ["ID", "r/m-OOD", "v-OOD", "both-OOD"]
piwm = {"uniform": [0.957, 0.332, 0.972, 0.478], "parabola": [0.982, 0.563, 0.977, 0.437]}
lewm = {"uniform": [0.930, 0.889, 0.865, 0.871], "parabola": [0.705, 0.735, 0.721, 0.513]}

fig2, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
x = np.arange(len(parts)); w = 0.38
for ax, dom in zip(axes, ["uniform", "parabola"]):
    b1 = ax.bar(x - w/2, piwm[dom], w, color=PURPLE, label="PIWM (official port)")
    b2 = ax.bar(x + w/2, lewm[dom], w, color=BLUE, label="LeWM free-rollout")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK)
    # shade the OOD-collapse partitions where PIWM loses
    for xi in (1, 3):
        ax.axvspan(xi - 0.5, xi + 0.5, color=VERM, alpha=0.06, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(parts, fontsize=9)
    ax.set_title(f"{dom}", fontsize=10, color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.12)
axes[0].set_ylabel("rolled-out position $\\rho$  (↑ better)")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
axes[1].text(1, 0.05, "shaded = size/mass-OOD\nPIWM collapses, LeWM holds", color=VERM, fontsize=8, ha="center")
fig2.suptitle("PIWM learns clean physics (wins ID & v-OOD) but its VAE encoder collapses on size/mass-OOD",
              fontsize=11, y=1.02, color=INK)
fig2.savefig(OUT / "fig2_piwm_vs_lewm.png"); fig2.savefig(OUT / "fig2_piwm_vs_lewm.pdf")
print("wrote fig2_piwm_vs_lewm.{png,pdf}")
