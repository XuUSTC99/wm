"""Figure 3 -- decodable, used, and still drifting. THREE small multiples, one
per domain (uniform / parabola / collision), single-column row.

Each panel: the shaded band is that domain's position decodability (probe rho on
the two coordinates pos0/pos1, both-OOD, from REAL frame embeddings) -- flat with
horizon, the "presence" ceiling. Two curves fall away from it: teacher forcing
(sienna) and free rollout (blue). The gap to the band grows with dynamics
complexity across the three panels; free rollout recovers much of it with no
physics added. Splitting the old single-axes version into three panels removes
the six-line pile-up and lets each domain's band sit at its own level.

Data provenance
---------------
Rollout cosine: storyline_figures.py FIG 13 (h = 1/4/8/16/28); collision matches
the earlier single-domain figure exactly.
Probe band: both-OOD REAL-embedding position rho, per coordinate (pos0, pos1),
from the per-domain table in detail/why_physics_structure_fails.md.

NOTE (why no Physion/Physion++ 4th panel): the matching experiment does not
exist there. Physion++ tells its long-horizon story in nMSE (bounded), has only
aggregate/per-scene cosine (not a per-horizon curve), and its probe number is
baseline *velocity* decodability (0.44) under the probe-injection variant -- not
a flat, high position-decodability band. A 4th panel would require new runs.
"""
import pathlib
import re
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_style
from fig_style import BLUE, SIENNA, INK, MUTED

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig_style.apply(base=8.0)

BAND = "#C7CBD1"   # neutral grey-blue: the probe band must not read as data

H = [1, 4, 8, 16, 28]
DOMS = ["uniform", "parabola", "collision"]
HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parents[3] / "raw_data" / "runs"


def _log_path(dom, mode, seed):
    if seed == 3072:
        name = "uniform_motion" if dom == "uniform" else dom
        return RUNS / "structdyn_eval" / f"rollout_{name}_baseline_{mode}_id1k.log"
    name = "uniform" if dom == "uniform" else dom
    return RUNS / "aaai_p0" / f"rollout_{name}_baseline_{mode}_s{seed}.log"


def _horizon_cosines(path):
    values = {}
    in_section = False
    for line in path.read_text().splitlines():
        if "latent fidelity vs horizon" in line:
            in_section = True
            continue
        if in_section and not line.strip():
            break
        if in_section and (match := re.match(
                r"\s*h=\s*(\d+).*?cos=([+\-\d.]+)", line)):
            values[int(match.group(1))] = float(match.group(2))
    missing = set(H) - values.keys()
    if missing:
        raise ValueError(f"missing horizons {sorted(missing)} in {path}")
    return [values[h] for h in H]


def _seed_curves(mode):
    return {dom: np.asarray([
        _horizon_cosines(_log_path(dom, mode, seed))
        for seed in (3072, 1234, 42)
    ]) for dom in DOMS}


TF_SEEDS = _seed_curves("tf")
FR_SEEDS = _seed_curves("fr")
# per-domain band = the two position coordinates' decodability rho (pos0, pos1)
PROBE = {"uniform": (0.862, 0.955), "parabola": (0.831, 0.893),
         "collision": (0.796, 0.889)}

fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.68), sharey=True)

for ax, dom in zip(axes, DOMS):
    lo, hi = PROBE[dom]
    tf_mean = TF_SEEDS[dom].mean(axis=0)
    fr_mean = FR_SEEDS[dom].mean(axis=0)
    ax.axhspan(lo, hi, color=BAND, alpha=0.65, lw=0, zorder=1)
    ax.plot(H, tf_mean, color=SIENNA, marker="s", ms=2.6, lw=1.4,
            mfc=SIENNA, mec=SIENNA, mew=0, zorder=4)
    ax.plot(H, fr_mean, color=BLUE, marker="^", ms=2.9, lw=1.4,
            mfc=BLUE, mec=BLUE, mew=0, zorder=3)
    ax.set_title(dom, fontsize=8.5, pad=2.5)
    ax.set_xticks([1, 8, 28])
    ax.set_xlim(0.4, 29)
    ax.set_ylim(0.10, 1.04)
    ax.tick_params(labelsize=7.5, length=2.4, width=0.6)
    ax.grid(True, axis="y", color="#EAE8E4", lw=0.6)
    ax.set_axisbelow(True)
    fig_style.despine(ax)

axes[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
fig.supxlabel("rollout horizon (steps)", fontsize=8.0, y=0.14)

key = [Line2D([], [], color=SIENNA, marker="s", ms=3, lw=1.6,
              mfc=SIENNA, mec=SIENNA, mew=0, label="teacher forcing"),
       Line2D([], [], color=BLUE, marker="^", ms=3.3, lw=1.6,
              mfc=BLUE, mec=BLUE, mew=0, label="free rollout"),
       Patch(facecolor=BAND, alpha=0.65, label="real-frame probe $\\rho$")]
fig.legend(handles=key, fontsize=8.0, loc="lower center", ncol=3,
           columnspacing=0.8, handlelength=1.4, handletextpad=0.4,
           borderpad=0.0, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(pad=0.3, rect=(0, 0.10, 1, 1))
for out in [HERE / "fig1_thesis_presence_not_use",
            HERE.parent / "paper" / "figures" / "fig1_thesis_presence_not_use"]:
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png")
    print("wrote", out)
print("  per-domain bands:", {d: PROBE[d] for d in DOMS})
print("  TF@h28 mean±std:", ", ".join(
    f"{d} {TF_SEEDS[d][:, -1].mean():.3f}±{TF_SEEDS[d][:, -1].std(ddof=1):.3f}"
    for d in DOMS))
print("  FR@h28 mean±std:", ", ".join(
    f"{d} {FR_SEEDS[d][:, -1].mean():.3f}±{FR_SEEDS[d][:, -1].std(ddof=1):.3f}"
    for d in DOMS))
