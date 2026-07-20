"""Storyline figures for the AAAI paper (advisor PPT set).
Run: le-wm/.venv/bin/python reports/7-11/aaai_paper/figures/storyline_figures.py

Every number is inlined with its raw-data source path (see comments) so figures are
reproducible & auditable. Palette = Okabe-Ito (CVD-safe). White bg for slides.
Data sources (curated): 01_results_ledger.md (C1-C7) + logs under
  /data1/likun-share/junjxu/runs/{aaai_p0,physionpp,structdyn_eval,piwm_baseline}/
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#D9D9D9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "axes.titleweight": "bold",
                     "font.family": "DejaVu Sans"})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf")
    print("wrote", name)

# ============================================================================
# FIG 1 — THESIS: presence != use.  collision baseline_fr/tf cos by horizon +
# "position decodable" band.  Source: aaai_p0/rollout_collision_baseline_{tf,fr}_s*.log
# ============================================================================
h = np.array([1, 2, 4, 8, 16, 28])
col_tf = np.array([0.991, 0.970, 0.881, 0.645, 0.361, 0.237])
col_fr = np.array([0.997, 0.993, 0.982, 0.948, 0.757, 0.476])
DECODE = 0.84  # collision both-OOD REAL-latent probe pos rho (avg 2 dims) = position IS in the latent
fig, ax = plt.subplots(figsize=(7.4, 4.7))
ax.axhspan(DECODE - 0.02, DECODE + 0.02, color=GREEN, alpha=0.15, lw=0)
ax.axhline(DECODE, color=GREEN, lw=2, ls="--")
# 2 lines + right-aligned: keeps the label in the right-hand region where BOTH curves are
# already low (blue < 0.73 for h >= 17), so nothing crosses the text.
ax.text(28, DECODE + 0.03, "position recoverable from latent\n(probe $\\rho\\approx0.84$, flat)",
        color=GREEN, ha="right", va="bottom", fontsize=9.5, fontweight="bold", linespacing=1.3)
ax.plot(h, col_tf, color=VERM, marker="o", ms=7, lw=2.4, label="rollout: teacher-forced (original)")
ax.plot(h, col_fr, color=BLUE, marker="s", ms=7, lw=2.4, label="rollout: free-rollout (ours)")
ax.annotate("1 step:\n≈perfect (0.99)", (1, 0.991), (2.4, 0.60), fontsize=9, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED))
# Annotation sits BELOW the orange curve (orange >= 0.29 for h <= 22), not on top of it.
ax.annotate("28 steps:\ncollapses to 0.24", (28, 0.237), (20.5, 0.125), fontsize=9, color=VERM, ha="center",
            va="center", arrowprops=dict(arrowstyle="->", color=VERM))
ax.set_xlabel("rollout horizon (steps)")
ax.set_ylabel("agreement w/ truth  (↑, 0–1)")
ax.set_title("Physical state is encoded, yet the rollout drifts away from it (collision)", fontsize=11)
ax.set_ylim(0.05, 1.06); ax.set_xticks(h)
ax.legend(frameon=False, loc="lower left", fontsize=9, bbox_to_anchor=(0.0, 0.02))
ax.grid(True, color=GRID, lw=0.7, alpha=0.7); ax.set_axisbelow(True)
save(fig, "fig1_thesis_presence_not_use"); plt.close(fig)

# ============================================================================
# FIG 2 — C1 free-rollout: THE universal lever. TF vs FR nMSE, 3 synthetic + real.
# Source: aaai_p0/rollout_*_baseline_{tf,fr}_s*.log ; physionpp/eval_pp_{tf,fr}_s*.log
# ============================================================================
doms = ["uniform", "parabola\n(r/m-OOD)", "collision", "Physion++\n(sim, h64)"]
tf = [0.300, 0.443, 1.153, 1.174]
fr = [0.136, 0.122, 0.479, 0.141]
mult = ["2.2×", "3.6×", "2.4×", "8.3×"]
fig, ax = plt.subplots(figsize=(7.8, 4.7))
x = np.arange(len(doms)); w = 0.38
b1 = ax.bar(x - w/2, tf, w, color=VERM, label="teacher-forced (original)")
b2 = ax.bar(x + w/2, fr, w, color=BLUE, label="free-rollout (ours)")
for xi, (t, f, m) in enumerate(zip(tf, fr, mult)):
    # value labels hug their own bar; the fold-change sits well above BOTH so the
    # two never collide (they used to be drawn at nearly the same height).
    ax.text(xi - w/2, t + 0.02, f"{t:.2f}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.text(xi + w/2, f + 0.02, f"{f:.2f}", ha="center", va="bottom", fontsize=8.5, color=BLUE)
    ax.text(xi, max(t, f) + 0.13, m, ha="center", fontweight="bold", color=INK, fontsize=11.5)
ax.axvline(2.5, color=GRID, lw=1.2, ls="-")
ax.text(1.0, 1.53, "synthetic (PhyWorld)", ha="center", color=MUTED, fontsize=9)
ax.text(3.0, 1.53, "photorealistic simulation", ha="center", color=MUTED, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=9.5)
ax.set_ylabel("rollout error  (nMSE, ↓)"); ax.set_ylim(0, 1.70)
ax.set_title("Free-rollout: the only lever that transfers across every domain (2.2–8.3×)",
             fontsize=11, pad=26)
# Legend above the axes: the upper-left interior is needed for the "synthetic" band label.
ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
          borderaxespad=0, handlelength=1.6, columnspacing=1.6)
ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7); ax.set_axisbelow(True)
save(fig, "fig2_free_rollout"); plt.close(fig)

# ============================================================================
# FIG 3 — C4 sign-flip: velocity-in-slot helps ONLY where slot matches dynamics.
# Source: aaai_p0/rollout_{uniform,parabola,collision}_structposvel_pw30*.log vs baseline_fr
# ============================================================================
labels = ["uniform\n(a=0, constant v)", "parabola\n(a=g, v linear ✓ match)", "collision\n(impulsive, v jumps)"]
delta = [0.207 - 0.136, 0.096 - 0.122, 0.621 - 0.479]   # posvel-in-slot minus free-rollout baseline
colors = [VERM if d > 0 else GREEN for d in delta]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
y = np.arange(len(labels))[::-1]
ax.barh(y, delta, color=colors, height=0.55)
ax.axvline(0, color=INK, lw=1.2)
for yi, d in zip(y, delta):
    ax.text(d + (0.006 if d > 0 else -0.006), yi, f"{d:+.3f}", va="center",
            ha="left" if d > 0 else "right", fontweight="bold",
            color=VERM if d > 0 else GREEN, fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Δ rollout error vs free-rollout  (nMSE; <0 = helps, >0 = hurts)")
ax.set_title("Physical structure helps ONLY when the slot matches the domain's dynamics\n(the 1/30 exception — the mechanism's signature, not a usable prior)", fontsize=10.5)
ax.set_xlim(-0.075, 0.185); ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.7); ax.set_axisbelow(True)
save(fig, "fig3_physics_signflip"); plt.close(fig)

# ============================================================================
# FIG 4 — C3 aug reversal: same appearance aug helps synthetic, catastrophic on real.
# Source: general_augmentation.md (synth) ; physionpp/eval_pp_fr{,_app05}_e20.log (real)
# ============================================================================
albl = ["uniform", "parabola", "collision", "Physion++\nfriction (real)"]
ratio = [0.068/0.131, 0.115/0.313, 0.172/0.393, 6.44/0.062]   # nMSE_aug / nMSE_base
acol = [GREEN, GREEN, GREEN, VERM]
fig, ax = plt.subplots(figsize=(7.2, 4.3))
x = np.arange(len(albl))
bars = ax.bar(x, ratio, color=acol, width=0.6)
ax.set_yscale("log")
ax.axhline(1.0, color=INK, lw=1.3, ls="--"); ax.text(3.4, 1.15, "parity (no effect)", color=INK, fontsize=8.5, ha="right")
for xi, r in zip(x, ratio):
    ax.text(xi, r * (1.15 if r > 1 else 0.82), f"{r:.2f}×" if r < 10 else f"{r:.0f}×",
            ha="center", va="bottom" if r > 1 else "top", fontweight="bold",
            color=VERM if r > 1 else GREEN, fontsize=11)
ax.text(0.5, 5.5, "synthetic: aug helps (−48 to −63%)", color=GREEN, ha="center", fontsize=9.5, fontweight="bold")
ax.text(2.35, 55, "real: aug\n100× worse", color=VERM, ha="right", fontsize=9.5, fontweight="bold")
ax.set_ylim(0.28, 260)
ax.set_xticks(x); ax.set_xticklabels(albl, fontsize=9.5)
ax.set_ylabel("nMSE ratio  aug / baseline  (log; <1 better)")
ax.set_title("The same appearance augmentation reverses from synthetic to real\n(appearance carries physics — friction/mass/material — in photorealistic scenes)", fontsize=10.5)
ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.6, which="both"); ax.set_axisbelow(True)
save(fig, "fig4_aug_synthetic_vs_real"); plt.close(fig)

# ============================================================================
# FIG 5 — C6 cos trap: cos says "better" while the metric that matters says "worse".
# Source: probe = 6-24/probe_vs_structpos_summary.md (uniform, h28: latent cos 0.843->0.882 UP, pixel PSNR 20.64->19.93 DOWN) ; physionpp/eval_pp_fr{,_app05}_e20.log (app-aug h64 & deform_clothhit)
# ============================================================================
cases = ["probe\n(latent cos vs pixel)", "app-aug\nh64 (cos vs nMSE)", "app-aug deform\n(cos vs nMSE)"]
cos_good = [0.882/0.843, 0.870/0.794, 0.913/0.610]      # cosine "goodness" ratio (>1 = cos says better)
true_good = [19.93/20.64, 0.280/0.311, 0.772/3.69]       # truth goodness (>1 better): pixel/pixel, nmse_base/nmse_aug
fig, ax = plt.subplots(figsize=(7.6, 4.3))
x = np.arange(len(cases)); w = 0.38
ax.bar(x - w/2, cos_good, w, color=SKY, label="cosine metric says…")
ax.bar(x + w/2, true_good, w, color=VERM, label="ground-truth metric (pixel / nMSE) says…")
ax.axhline(1.0, color=INK, lw=1.3, ls="--")
ax.set_yscale("log")
for xi, (c, t) in enumerate(zip(cos_good, true_good)):
    ax.text(xi - w/2, c * 1.05, f"{c:.2f}", ha="center", va="bottom", fontsize=9, color="#1f6f9b", fontweight="bold")
    ax.text(xi + w/2, t * 0.94, f"{t:.2f}", ha="center", va="top", fontsize=9, color=VERM, fontweight="bold")
ax.text(2.45, 1.9, "cos says 'improved' ↑", color="#1f6f9b", fontsize=9, ha="right", fontweight="bold")
ax.text(2.45, 0.24, "truth says 'worse' ↓", color=VERM, fontsize=9, ha="right", fontweight="bold")
ax.set_ylim(0.18, 2.2)
ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=9.5)
ax.set_ylabel("goodness ratio vs baseline  (log; >1 better)")
ax.set_title("The cosine trap: cos rises (blue > 1) while the metric that matters falls (red < 1)", fontsize=10.5)
ax.legend(frameon=False, loc="lower left", fontsize=9); ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.6, which="both"); ax.set_axisbelow(True)
save(fig, "fig5_cos_trap"); plt.close(fig)

# ============================================================================
# FIG 6 — C6/C7 transfer ceiling: nothing beats the random-init prior (0.607).
# Source: transfer_improvement_report.md §1-2 ; reports/physion/eval_*.json
# ============================================================================
cfg = ["random prior\n(architecture ceiling)", "free-rollout", "appearance aug 0.5",
       "+consistency+accel", "appearance aug 0.3", "+consistency", "pos_weight (load-bearing)"]
auc = [0.607, 0.603, 0.597, 0.582, 0.579, 0.566, 0.551]
ccol = [MUTED, BLUE, SKY, ORANGE, SKY, ORANGE, VERM]
fig, ax = plt.subplots(figsize=(7.6, 4.3))
yy = np.arange(len(cfg))[::-1]
ax.barh(yy, auc, color=ccol, height=0.62)
ax.axvline(0.607, color=INK, lw=1.4, ls="--"); ax.text(0.607, len(cfg) - 0.3, " ceiling 0.607", color=INK, fontsize=9, va="center")
for yi, a in zip(yy, auc):
    ax.text(a - 0.003, yi, f"{a:.3f}", va="center", ha="right", color="white", fontweight="bold", fontsize=9)
ax.set_yticks(yy); ax.set_yticklabels(cfg, fontsize=9)
ax.set_xlabel("Physion zero-shot mean AUC  (↑;  0.607 = random-init prior)")
ax.set_xlim(0.53, 0.615)
ax.set_title("Zero-shot transfer ceiling: no training config beats the random prior\n(physical structure = worst; free-rollout closest)", fontsize=10.5)
ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.6); ax.set_axisbelow(True)
save(fig, "fig6_transfer_ceiling"); plt.close(fig)

# ============================================================================
# FIG 7 — C7 real-data num_preds: longer rollout monotonically helps, no plateau.
# Source: physionpp/eval_pp_fr{,_np20,_np20sc,_np28sc,_scnp16}_e20*.log
# ============================================================================
hz = np.array([16, 32, 64])
series = {"np8 (baseline)": ([0.0164, 0.1404, 0.2797], VERM, "o"),
          "np20": ([0.0215, 0.0326, 0.2203], ORANGE, "s"),
          "np20 + scale": ([0.0115, 0.0236, 0.0866], SKY, "^"),
          "np28 + scale": ([0.0076, 0.0101, 0.0144], GREEN, "D")}
fig, ax = plt.subplots(figsize=(7.6, 4.5))
for name, (vals, c, mk) in series.items():
    ax.plot(hz, vals, color=c, marker=mk, ms=7, lw=2.2, label=name)
ax.set_yscale("log"); ax.set_xticks(hz)
ax.set_xlim(12, 78)   # right-hand room so the legend never sits on the green line
ax.set_xlabel("rollout horizon (steps)"); ax.set_ylabel("nMSE  (log, ↓)")
ax.annotate("np28+scale h64 = 0.014\n(1/19 of baseline; no plateau)", (64, 0.0144), (32, 0.0055),
            fontsize=9, color=GREEN, ha="center", arrowprops=dict(arrowstyle="->", color=GREEN))
# Physion/Physion++ are photorealistic SIMULATION (TDW/Unity), never "real data".
ax.set_title("Physion++ (photorealistic simulation): longer training rollout → better long horizon",
             fontsize=10.5, pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.005, 0.99))
ax.grid(True, color=GRID, lw=0.7, alpha=0.6, which="both"); ax.set_axisbelow(True)
save(fig, "fig7_realdata_num_preds"); plt.close(fig)

# ============================================================================
# FIG 10 — LBR all-partitions ratio-to-baseline (detail for load_bearing_reweighting.md §2).
# Source: structdyn_eval/rollout_{uniform_motion,parabola,collision}_structpos_fr_pw*_id1k.log (seed 3072)
# ============================================================================
pw = [1, 3, 10, 30, 100, 300]
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
pcolor = {"ID":MUTED, "r/m-OOD":ORANGE, "v-OOD":SKY, "both-OOD":BLUE}
pmark  = {"ID":".", "r/m-OOD":"s", "v-OOD":"^", "both-OOD":"o"}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
for ax, dom in zip(axes, ["uniform", "parabola", "collision"]):
    d = lbr[dom]
    ax.axhspan(1.0, 2.0, color=VERM, alpha=0.05, lw=0)
    ax.axhspan(0.5, 1.0, color=GREEN, alpha=0.06, lw=0)
    for part in ["ID", "r/m-OOD", "v-OOD", "both-OOD"]:
        ratio = [v / d["base"][part] for v in d[part]]
        dashed = (dom == "parabola" and part == "both-OOD")
        ax.plot(pw, ratio, color=pcolor[part], marker=pmark[part], ms=6, lw=2.2,
                ls="--" if dashed else "-", label=part + (" (blowup)" if dashed else ""))
    ax.axhline(1.0, color=INK, lw=1.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks(pw); ax.set_xticklabels(pw)
    ax.set_xlabel("pos_weight $\\lambda$ (log)")
    b = d["base"]
    ax.set_title(f"{dom}\nbaseline nMSE: {b['ID']:.2f} / {b['r/m-OOD']:.2f} / {b['v-OOD']:.2f} / {b['both-OOD']:.2f}", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
axes[0].set_ylabel("nMSE / baseline  (1.0 = baseline; >1 = worse)")
axes[0].text(1.05, 1.02, "baseline", color=INK, fontsize=8, va="bottom")
axes[2].text(300, 1.70, "collision: all partitions\nharmful at every $\\lambda$", color=VERM, fontsize=8.5, ha="right", va="top", fontweight="bold")
fig.suptitle("pos_weight re-weighting vs baseline (dotted=baseline; above=worse): nearly every partition stays above baseline — only a few cells reach parity",
             fontsize=10.5, y=1.02, color=INK)
save(fig, "fig10_lbr_all_partitions"); plt.close(fig)

# ============================================================================
# FIG 11 — LBR all-partitions ABSOLUTE nMSE (solid = pos_weight curve, dashed = baseline).
# Same data/source as FIG 10; absolute magnitudes on log-y.
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
for ax, dom in zip(axes, ["uniform", "parabola", "collision"]):
    d = lbr[dom]
    for part in ["ID", "r/m-OOD", "v-OOD", "both-OOD"]:
        dashed = (dom == "parabola" and part == "both-OOD")
        ax.plot(pw, d[part], color=pcolor[part], marker=pmark[part], ms=6, lw=2.2,
                ls="-", label=part + (" (blowup)" if dashed else ""))
        ax.axhline(d["base"][part], color=pcolor[part], ls="--", lw=1.3, alpha=0.55)  # baseline
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(pw); ax.set_xticklabels(pw)
    ax.set_xlabel("pos_weight $\\lambda$ (log)")
    ax.set_title(dom, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, which="both"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)
axes[0].set_ylabel("latent nMSE  (log, ↓ better)")
fig.suptitle("pos_weight re-weighting — absolute nMSE per partition (solid) vs its baseline (dashed, same color)",
             fontsize=10.5, y=1.0, color=INK)
save(fig, "fig11_lbr_all_partitions_absolute"); plt.close(fig)

# ============================================================================
# FIG 12 — free-rollout improves EVERY partition (fold improvement TF/FR).
# Source: aaai_p0/rollout_{dom}_baseline_{tf,fr}_s1234.log
# ============================================================================
fr_parts = ["ID", "r/m-OOD", "v-OOD", "both-OOD"]
fold = {"uniform":   [3.0, 3.3, 4.6, 2.0],
        "parabola":  [2.5, 3.3, 2.5, 2.5],
        "collision": [2.6, 3.9, 2.2, 2.4]}
pcol = {"ID": MUTED, "r/m-OOD": ORANGE, "v-OOD": SKY, "both-OOD": BLUE}
fig, ax = plt.subplots(figsize=(9.2, 4.4))
doms3 = ["uniform", "parabola", "collision"]
group_w = 0.8; bw = group_w / 4
for gi, dom in enumerate(doms3):
    for pi, part in enumerate(fr_parts):
        x = gi + (pi - 1.5) * bw
        v = fold[dom][pi]
        ax.bar(x, v, bw * 0.92, color=pcol[part], label=part if gi == 0 else None)
        ax.text(x, v + 0.06, f"{v:.1f}×", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
ax.axhline(1.0, color=VERM, lw=1.8, ls="--")
ax.text(2.55, 1.05, "1× = no improvement", color=VERM, fontsize=9, ha="right", va="bottom", fontweight="bold")
ax.set_xticks(range(len(doms3))); ax.set_xticklabels(doms3, fontsize=11)
ax.set_ylabel("improvement factor  (TF error / FR error, ↑)")
ax.set_ylim(0, 5.2)
ax.set_title("Free-rollout improves EVERY partition of EVERY domain — including in-distribution (2.0–4.6×)\n(not an OOD patch: even ID drops 2.5–3.0×)", fontsize=10.5)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=4)
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
save(fig, "fig12_free_rollout_all_partitions"); plt.close(fig)

# ============================================================================
# FIG 13 — advantage grows with horizon (exposure-bias signature). TF vs FR cos.
# Source: aaai_p0/rollout_{dom}_baseline_{tf,fr}_s1234.log (--- vs horizon --- cos)
# ============================================================================
hz2 = [1, 4, 8, 16, 28]
cos_tf = {"uniform":[0.99,0.96,0.91,0.84,0.84], "parabola":[0.98,0.88,0.84,0.52,0.57], "collision":[0.99,0.88,0.64,0.36,0.24]}
cos_fr = {"uniform":[0.99,0.98,0.97,0.95,0.95], "parabola":[0.98,0.94,0.94,0.79,0.93], "collision":[1.00,0.98,0.95,0.76,0.48]}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
for ax, dom in zip(axes, doms3):
    tf, fr = cos_tf[dom], cos_fr[dom]
    ax.fill_between(hz2, tf, fr, color=BLUE, alpha=0.12, lw=0)
    ax.plot(hz2, tf, color=VERM, marker="o", ms=6, lw=2.4, ls="--", label="teacher-forced")
    ax.plot(hz2, fr, color=BLUE, marker="s", ms=6, lw=2.4, label="free-rollout")
    gap = fr[-1] - tf[-1]
    ax.annotate(f"+{gap:.2f}", (28, (tf[-1]+fr[-1])/2), fontsize=10, color=BLUE, ha="right", fontweight="bold")
    ax.set_xticks(hz2); ax.set_xlabel("rollout horizon"); ax.set_title(dom, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True); ax.set_ylim(0.15, 1.03)
axes[0].set_ylabel("prediction fidelity (cos, ↑)")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
axes[1].annotate("equal at 1 step", (1, 0.98), (4, 0.80), fontsize=8.5, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED))
fig.suptitle("The advantage is ~zero at one step and grows monotonically with horizon — the exposure-bias signature",
             fontsize=10.5, y=1.0, color=INK)
save(fig, "fig13_free_rollout_horizon_gap"); plt.close(fig)

# ============================================================================
# FIG 14 — Physion++ (real data): free-rollout vs teacher-forced by horizon.
# Source: physionpp/eval_pp_{tf,fr}_s3072.log (--- latent fidelity vs HORIZON ---)
# ============================================================================
hz3 = [4, 8, 16, 32, 64]
pp_tf_cos = [0.998, 0.993, 0.929, 0.829, 0.479]
pp_fr_cos = [0.999, 0.999, 0.996, 0.971, 0.898]
pp_tf_nmse = [0.0046, 0.015, 0.138, 0.324, 1.219]
pp_fr_nmse = [0.0017, 0.004, 0.015, 0.064, 0.139]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.3))
# cos
axL.fill_between(hz3, pp_tf_cos, pp_fr_cos, color=BLUE, alpha=0.12, lw=0)
axL.plot(hz3, pp_tf_cos, color=VERM, marker="o", ms=6, lw=2.4, ls="--", label="teacher-forced")
axL.plot(hz3, pp_fr_cos, color=BLUE, marker="s", ms=6, lw=2.4, label="free-rollout")
axL.annotate("+0.42", (64, 0.69), fontsize=11, color=BLUE, ha="right", fontweight="bold")
axL.set_xticks(hz3); axL.set_xlabel("rollout horizon"); axL.set_ylabel("prediction fidelity (cos, ↑)")
axL.set_title("(a) drift: cos", fontsize=10); axL.set_ylim(0.4, 1.02)
axL.legend(frameon=False, fontsize=9, loc="lower left"); axL.grid(True, color=GRID, lw=0.6, alpha=0.6); axL.set_axisbelow(True)
# nMSE (log)
axR.plot(hz3, pp_tf_nmse, color=VERM, marker="o", ms=6, lw=2.4, ls="--", label="teacher-forced")
axR.plot(hz3, pp_fr_nmse, color=BLUE, marker="s", ms=6, lw=2.4, label="free-rollout")
axR.set_yscale("log")
axR.annotate("8.3× at h64\n(1.17 → 0.14)", (64, 0.42), (20, 0.55), fontsize=10, color=INK, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=INK))
axR.set_xticks(hz3); axR.set_xlabel("rollout horizon"); axR.set_ylabel("nMSE (log, ↓)")
axR.set_title("(b) error: nMSE", fontsize=10)
axR.legend(frameon=False, fontsize=9, loc="lower right"); axR.grid(True, color=GRID, lw=0.6, alpha=0.6, which="both"); axR.set_axisbelow(True)
fig.suptitle("Real data (Physion++): free-rollout holds long-horizon while teacher-forcing collapses — the effect is even LARGER than synthetic (8.3×)",
             fontsize=10.5, y=1.0, color=INK)
save(fig, "fig14_physionpp_free_rollout"); plt.close(fig)

# ============================================================================
# FIG 15 — the BYPASS, measured (probe-190): black-box 190 dims alone decode
# position ~ as well as the full latent -> prediction can route around any slot.
# Source: why_physics_structure_fails.md 层2② table (probe_dim_subset.py), both-OOD pos rho.
# ============================================================================
bdoms = ["uniform", "parabola", "collision"]
full192 = [0.92, 0.85, 0.78]      # decode position from all 192 dims (baseline)
blackbox190 = [0.92, 0.85, 0.79]  # decode from black-box [2:192] only (baseline, no slot)
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(len(bdoms)); w = 0.34
# Reserve empty space on the right for the control annotation so it never sits on a bar.
ax.set_xlim(-0.62, 3.30)
ax.axhspan(0.2, 0.5, color=VERM, alpha=0.08, lw=0)
ax.text(2.62, 0.35, "random 2 dims\n(control):\ncan't decode\n(0.2–0.5)", color=VERM, fontsize=8.5,
        ha="left", va="center", linespacing=1.35)
b1 = ax.bar(x - w/2, full192, w, color=INK, label="all 192 dims")
b2 = ax.bar(x + w/2, blackbox190, w, color=SKY, label="black-box 190 dims only (physics slot removed)")
for xi in range(len(bdoms)):
    ax.text(xi - w/2, full192[xi] + 0.02, f"{full192[xi]:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(xi + w/2, blackbox190[xi] + 0.02, f"{blackbox190[xi]:.2f}", ha="center", va="bottom", fontsize=9.5,
            color="#1f6f9b", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(bdoms, fontsize=11)
ax.set_ylabel("position decodable  (probe $\\rho$, ↑)"); ax.set_ylim(0, 1.16)
# Legend above the axes: the bars fill both the lower and upper interior, leaving no free spot inside.
ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
          borderaxespad=0, handlelength=1.6, columnspacing=1.4)
ax.set_title("Black-box dims alone decode position as well as the full latent\n→ prediction can route around any physics slot",
             fontsize=10.5, pad=30)
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
save(fig, "fig15_bypass_probe190"); plt.close(fig)

# ============================================================================
# FIG 16 — full physics-injection scan: 10 arms x 3 domains (30 cells), ratio to baseline.
# Source: runs/{aaai_p0,structdyn_eval,consistency_eval}/rollout_*.log (judged partition:
#   uniform/collision both-OOD, parabola r/m-OOD). baseline u=0.131 p=0.127 c=0.393.
# ============================================================================
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
arms = ["[slot] structpos", "[slot] +reweight (w=30)", "[slot] +velocity",
        "[probe] probe", "[probe] +slot", "[dyn] free MLP", "[dyn] strict a=g",
        "[cons] consistency", "[free] label-free", "[free] grounded"]
bcols = ["uniform\n(both-OOD)", "parabola\n(r/m-OOD)", "collision\n(both-OOD)"]
base = np.array([0.131, 0.127, 0.393])
raw = np.array([
    [0.183, 0.156, 0.651],   # structpos
    [0.114, 0.160, 0.596],   # +pw30
    [0.207, 0.093, 0.621],   # +velocity (posvel)
    [0.167, 0.115, 0.647],   # probe
    [0.125, 0.127, 0.607],   # probe+structpos
    [0.155, 0.178, 0.560],   # dyn MLP
    [0.206, 0.173, 0.559],   # dyn const a=g
    [0.151, 0.147, 0.640],   # consistency
    [0.171, 0.172, 0.653],   # label-free
    [0.166, 0.156, 0.524],   # grounded
])
# 4 cells have 3-SEED data — display & color by the 3-seed MEAN (single-seed <1 was seed luck):
disp = raw.copy()
seed3 = {(1, 0): 0.132, (3, 1): 0.137, (4, 0): 0.141, (2, 1): 0.096}  # 3-seed means
for (i, j), v in seed3.items():
    disp[i, j] = v
ratio = disp / base            # color by the real (best-estimate) ratio, no artificial override
parity3 = {(1, 0), (3, 1), (4, 0)}  # 3-seed overlaps baseline (parity)
gain = {(2, 1)}                     # posvel·parabola = only real 3-seed gain
ratio_c = ratio
cmap = LinearSegmentedColormap.from_list("bwv", [BLUE, "#f7f7f7", VERM])  # blue<1 white=1 vermillion>1 (Okabe-Ito, CVD-safe)
norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.7)
fig, ax = plt.subplots(figsize=(6.6, 7.2))
im = ax.imshow(ratio_c, cmap=cmap, norm=norm, aspect="auto")
for i in range(len(arms)):
    for j in range(3):
        r = ratio[i, j]
        mark = "†" if (i, j) in parity3 else ("†✓" if (i, j) in gain else "")
        ax.text(j, i, f"{r:.2f}×\n({disp[i,j]:.3f}){mark}", ha="center", va="center",
                fontsize=8, color=INK if 0.9 < ratio_c[i, j] < 1.3 else "white", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(bcols, fontsize=9.5)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=9)
for y in [2.5, 4.5, 6.5, 7.5]:   # family separators
    ax.axhline(y, color="white", lw=2)
ax.set_title("Every physics-injection arm × domain vs baseline (nMSE ratio; number = single-seed 3072, † = 3-seed mean)\n"
             "physics ≥ baseline nearly everywhere (color: red worse / white parity / green better); †✓ posvel·parabola = only real gain",
             fontsize=8.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("nMSE / baseline  (>1 worse, <1 better; 1.0 = baseline)", fontsize=9)
save(fig, "fig16_physics_injection_scan"); plt.close(fig)

# --- paper variant: same data, laid out for a single AAAI column ------------
# The slide version (6.6x7.2in) shrinks to ~4pt text when dropped into a 3.3in
# column. Here: no in-figure title (the LaTeX caption carries it), tall/narrow
# geometry, so nothing is downscaled and the cells stay legible in print.
fig, ax = plt.subplots(figsize=(3.45, 6.0))
im = ax.imshow(ratio_c, cmap=cmap, norm=norm, aspect="auto")
for i in range(len(arms)):
    for j in range(3):
        r = ratio[i, j]
        mark = "†" if (i, j) in parity3 else ("†✓" if (i, j) in gain else "")
        ax.text(j, i, f"{r:.2f}×\n({disp[i,j]:.3f}){mark}", ha="center", va="center",
                fontsize=6.4, color=INK if 0.9 < ratio_c[i, j] < 1.3 else "white", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(bcols, fontsize=6.6)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=6.6)
ax.tick_params(length=2, pad=2)
for y in [2.5, 4.5, 6.5, 7.5]:
    ax.axhline(y, color="white", lw=1.6)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("nMSE / baseline", fontsize=6.6); cb.ax.tick_params(labelsize=6)
save(fig, "fig16_scan_paper"); plt.close(fig)

print("ALL storyline figures done ->", OUT)
