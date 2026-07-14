"""Generate figures for the Physion++ report from eval logs (2026-07-12).

Reads /data1/.../runs/physionpp/eval_<tag>.log, parses the by-horizon nMSE
(first h= block = overall horizon), and draws 3 figures into reports/physion/figures/.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/data1/likun-share/junjxu/runs/physionpp"
OUT = "/home/likun-share/junjxu/wm/reports/physion/figures"
os.makedirs(OUT, exist_ok=True)

PAT = re.compile(r"^\s*h=\s*(\d+)\s+n=\s*\d+\s+cos=([+\-\d.]+)\s+nMSE=([\d.]+)")


def parse_horizon(tag):
    """First 7 h= lines = overall horizon curve (before scene/group blocks)."""
    hs, nmse = [], []
    path = f"{LOG}/eval_{tag}.log"
    if not os.path.exists(path):
        print(f"  MISSING {path}")
        return [], []
    for line in open(path):
        m = PAT.match(line)
        if m:
            h = int(m.group(1))
            if h in hs:
                break  # entered scene/group block (repeats h=)
            hs.append(h)
            nmse.append(float(m.group(3)))
        if len(hs) >= 7:
            break
    return hs, nmse


# ---- Figure A: FR vs TF headline ----
plt.figure(figsize=(6.2, 4.6))
for s, c in [("3072", "#1f4e79"), ("1234", "#2e75b6"), ("42", "#5b9bd5")]:
    h, n = parse_horizon(f"pp_fr_s{s}")
    if h:
        plt.plot(h, n, "o-", color=c, label=f"FR (np8) seed{s}", lw=2)
for s, c in [("3072", "#843c0c"), ("1234", "#c55a11"), ("42", "#ed7d31")]:
    h, n = parse_horizon(f"pp_tf_s{s}")
    if h:
        plt.plot(h, n, "s--", color=c, label=f"TF (np1) seed{s}", lw=1.6, alpha=0.85)
plt.xlabel("rollout horizon"); plt.ylabel("latent nMSE  (lower = better)")
plt.yscale("log"); plt.xscale("log", base=2)
plt.title("free-rollout ≫ teacher-forcing  (Physion++, 3 seeds)")
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_fr_vs_tf.png", dpi=130); plt.close()

# ---- Figure B: num_preds longer-rollout ----
plt.figure(figsize=(6.2, 4.6))
for tag, lab, c in [
    ("pp_fr_e20", "baseline np8", "#999999"),
    ("pp_fr_np20_e20", "np20 (no scale)", "#7cb342"),
    ("pp_fr_np20sc_e20", "np20 + scale", "#2e75b6"),
    ("pp_fr_np28sc_e20", "np28 + scale", "#1f4e79"),
]:
    h, n = parse_horizon(tag)
    if h:
        plt.plot(h, n, "o-", label=lab, lw=2, color=c)
plt.xlabel("rollout horizon"); plt.ylabel("latent nMSE  (lower = better)")
plt.yscale("log"); plt.xscale("log", base=2)
plt.title("longer free-rollout (num_preds↑) → better long-horizon")
plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_num_preds.png", dpi=130); plt.close()

# ---- Figure C: init ablation + physical structure (h64 bar) ----
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
def h64_seeds(base):
    tags = [f"pp_init_{base}", f"pp_init_{base}_s1234", f"pp_init_{base}_s42"]
    vals = [parse_horizon(t)[1][-1] for t in tags if parse_horizon(t)[1]]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

init_lab = ["scratch", "cube (3D)", "pusht (2D)"]
init_ms = [h64_seeds(b) for b in ["scratch", "cube", "pusht"]]
init_m = [x[0] for x in init_ms]; init_s = [x[1] for x in init_ms]
axes[0].bar(init_lab, init_m, yerr=init_s, capsize=6, color=["#2e75b6", "#7cb342", "#c55a11"])
axes[0].set_title("init ablation  (h64 nMSE, np20sc, 3 seeds)"); axes[0].set_ylabel("h64 nMSE (↓)")
for i, (m, s) in enumerate(init_ms):
    axes[0].text(i, m + s, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8)

ps_tags = ["pp_fr_s3072", "pp_struct2", "pp_cons2", "pp_consacc2"]
ps_lab = ["pure FR", "+struct", "+cons", "+cons+accel"]
ps_h64 = [parse_horizon(t)[1][-1] if parse_horizon(t)[1] else 0 for t in ps_tags]
axes[1].bar(ps_lab, ps_h64, color=["#2e75b6", "#c55a11", "#ed7d31", "#f4b183"])
axes[1].set_title("physical structure hurts long-horizon  (h64 nMSE)"); axes[1].set_ylabel("h64 nMSE (↓)")
for i, v in enumerate(ps_h64):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_init_structure.png", dpi=130); plt.close()

print("figures saved to", OUT)
print("  init h64 (mean±std):", {l: (round(m, 4), round(s, 4)) for l, (m, s) in zip(init_lab, init_ms)})
print("  structure h64:", dict(zip(ps_lab, [round(x, 4) for x in ps_h64])))
