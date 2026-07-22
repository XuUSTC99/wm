"""Per-dimension Jacobian sensitivity: slot dims vs black-box dims, all cells.

Unlike every interventional measure here, this picks no direction and designs no
intervention, so all 36 cells are usable -- the dose ladder left only one.
"""
import os, re
import numpy as np

R = "/data1/likun-share/junjxu/runs/causal_route/jacobian"
SEEDS = [3072, 1234, 42]

def read(dom, model, arm, seed):
    p = f"{R}/jac_{dom}_{model}_{arm}_s{seed}.log"
    if not os.path.exists(p): return None
    out = {}
    for line in open(p, errors="ignore"):
        m = re.search(r"slot \[0:2\]\s+mean \|dpos/dz\| per dim = ([\d.]+)", line)
        if m: out["slot"] = float(m.group(1))
        m = re.search(r"black box\s+mean \|dpos/dz\| per dim = ([\d.]+)", line)
        if m: out["bb"] = float(m.group(1))
        m = re.search(r"RATIO slot/bb per dim = ([\d.]+)", line)
        if m: out["ratio"] = float(m.group(1))
        m = re.search(r"slot share ([\d.]+)", line)
        if m: out["share"] = float(m.group(1))
        m = re.search(r"slot dims rank \[(\d+), (\d+)\] of (\d+)", line)
        if m: out["rank"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out if "ratio" in out else None

def agg(v):
    return f"{np.mean(v):.2f}±{np.std(v,ddof=1):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if v else "—")

print("Per-dimension sensitivity of the predicted position to each input latent dim.")
print("ratio = slot per-dim / black-box per-dim.  share = slot's fraction of TOTAL sensitivity.")
print("rank  = where the two slot dims sit among all 192 dims, most sensitive first.\n")
for model in ("lewm", "dino"):
    print("=" * 78); print(model.upper()); print("=" * 78)
    print(f"{'domain':<16}{'arm':<11}{'slot/dim':>10}{'bb/dim':>10}{'ratio':>12}{'slot share':>12}{'slot ranks':>16}")
    for dom in ("uniform_motion", "parabola", "collision"):
        for arm in ("baseline", "structpos"):
            rs_, ss, bs, sh, rk = [], [], [], [], []
            for s in SEEDS:
                d = read(dom, model, arm, s)
                if not d: continue
                rs_.append(d["ratio"]); ss.append(d["slot"]); bs.append(d["bb"])
                if "share" in d: sh.append(d["share"])
                if "rank" in d: rk.append(d["rank"][:2])
            if not rs_: continue
            rkm = f"{np.mean([r[0] for r in rk]):.0f},{np.mean([r[1] for r in rk]):.0f}" if rk else "—"
            print(f"{dom:<16}{arm:<11}{np.mean(ss):>10.4f}{np.mean(bs):>10.4f}"
                  f"{agg(rs_):>12}{(f'{np.mean(sh):.3f}' if sh else '—'):>12}{rkm:>16}")
    print()
