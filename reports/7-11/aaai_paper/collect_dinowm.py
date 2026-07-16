"""Collect the dinowm (frozen DINOv2 + AR predictor) cross-model matrix.

Decision partitions follow the paper's rule: uniform/collision -> both-OOD,
parabola -> r/m-OOD (both-OOD nMSE has the divide-by-zero blowup there);
physion -> h64. Prints tables ready to paste into the report.
"""
import os
import re

import numpy as np

LOG = "/data1/likun-share/junjxu/runs/dinowm"
PART = {"um": "both-OOD", "par": "r/m-OOD", "col": "both-OOD"}
DOMNAME = {"um": "uniform", "par": "parabola(r/m)", "col": "collision"}


def nmse(tag, part):
    f = f"{LOG}/rollout_{tag}.log"
    if not os.path.exists(f):
        return None
    pat = re.compile(rf"^\s*{re.escape(part)}\s+n=\s*\d+\s+cos=[+\-\d.]+\s+nMSE=([\d.]+)")
    for line in open(f):
        m = pat.match(line)
        if m:
            return float(m.group(1))
    return None


def h64(tag):
    f = f"{LOG}/rollout_{tag}.log"
    if not os.path.exists(f):
        return None
    for line in open(f):
        m = re.match(r"^\s*h=\s*64\s+n=\s*\d+\s+cos=([+\-\d.]+)\s+nMSE=([\d.]+)", line)
        if m:
            return float(m.group(2))
    return None


def ms(vals):
    v = [x for x in vals if x is not None]
    return (np.mean(v), np.std(v), len(v)) if v else (None, None, 0)


print("=" * 72)
print("A) HEADLINE: free-rollout vs teacher-forcing  (dinowm, 3 seeds)")
print("=" * 72)
print(f"{'domain':<16} {'TF mean±std':<22} {'FR mean±std':<22} {'ratio':<8}")
fr_base = {}
for d in ["um", "par", "col"]:
    tf = ms([nmse(f"dinowm_{d}_tf_s{s}", PART[d]) for s in [3072, 1234, 42]])
    fr = ms([nmse(f"dinowm_{d}_fr_s{s}", PART[d]) for s in [3072, 1234, 42]])
    fr_base[d] = fr[0]
    if tf[0] and fr[0]:
        print(f"{DOMNAME[d]:<16} {tf[0]:.3f}±{tf[1]:.3f} (n={tf[2]})   "
              f"{fr[0]:.3f}±{fr[1]:.3f} (n={fr[2]})   {tf[0]/fr[0]:.2f}x")
    else:
        print(f"{DOMNAME[d]:<16} TF={tf} FR={fr}")

print()
print("=" * 72)
print("B) PHYSICS INJECTION vs pure FR baseline  (ratio >1 = WORSE than baseline)")
print("=" * 72)
ARMS = [
    ("structpos_pw30", "[slot] pos + pw30", True),
    ("probeF2", "[probe] deep-sup f2", True),
    ("cons", "[cons] consistency", True),
    ("posvel_pw30", "[slot] pos+vel pw30", False),
    ("structpos_plain", "[slot] pos, no pw", False),
    ("grounded_const", "[dyn] slot+kinematic", False),
]
print(f"{'arm':<24} {'uniform':<18} {'parabola':<18} {'collision':<18}")
for key, label, seeded in ARMS:
    row = f"{label:<24}"
    for d in ["um", "par", "col"]:
        tags = [f"dinowm_{d}_{key}"]
        if seeded:
            tags += [f"dinowm_{d}_{key}_s1234", f"dinowm_{d}_{key}_s42"]
        m, s, n = ms([nmse(t, PART[d]) for t in tags])
        if m is None:
            row += f"{'--':<18}"
        else:
            r = m / fr_base[d] if fr_base.get(d) else float("nan")
            flag = "WORSE" if r > 1.05 else ("better" if r < 0.95 else "~same")
            row += f"{m:.3f} ({r:.2f}x {flag})".ljust(18)
    print(row)

print()
print("=" * 72)
print("C) HORIZON-COMPLEXITY MATCHING (C2 double dissociation)")
print("=" * 72)
for d, extra in [("col", ["np16", "np20"]), ("um", ["np16"]), ("par", ["np16"])]:
    base = fr_base.get(d)
    for e in extra:
        v = nmse(f"dinowm_{d}_{e}", PART[d])
        if v and base:
            r = v / base
            verdict = "helps" if r < 0.95 else ("HURTS" if r > 1.05 else "~same")
            print(f"  {DOMNAME[d]:<16} np8(base)={base:.3f} -> {e}={v:.3f}  ({r:.2f}x {verdict})")

print()
print("=" * 72)
print("D) LBR curve: pos_weight sweep on uniform (mechanism, cross-model)")
print("=" * 72)
base = fr_base.get("um")
for pw in ["pw1", "pw10", "pw30", "pw100", "pw300"]:
    tags = [f"dinowm_um_structpos_{pw}"]
    if pw == "pw30":
        tags += ["dinowm_um_structpos_pw30_s1234", "dinowm_um_structpos_pw30_s42"]
    m, s, n = ms([nmse(t, "both-OOD") for t in tags])
    if m:
        print(f"  pos_weight={pw[2:]:<4} both-OOD={m:.3f}  ({m/base:.2f}x baseline)  n={n}")

print()
print("=" * 72)
print("E) PHYSION (real data) x dinowm: FR vs TF, h64 nMSE")
print("=" * 72)
tf = ms([h64(f"dinowm_pp_tf_s{s}") for s in [3072, 1234, 42]])
fr = ms([h64(f"dinowm_pp_fr_s{s}") for s in [3072, 1234, 42]])
if tf[0] and fr[0]:
    print(f"  TF: {tf[0]:.3f}±{tf[1]:.3f} (n={tf[2]})   FR: {fr[0]:.3f}±{fr[1]:.3f} (n={fr[2]})   "
          f"ratio {tf[0]/fr[0]:.2f}x")
else:
    print(f"  TF={tf}  FR={fr}")
