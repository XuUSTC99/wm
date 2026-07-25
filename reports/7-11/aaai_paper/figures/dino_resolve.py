"""Per-seed log resolution for the frozen-DINOv2 injection scan (fig 6).

Mirrors scan_resolve.py's API for the dinowm runs so the two scan figures can
share one drawing routine and one verdict rule (seed-range separation against
the baseline's seed range, not a ratio threshold).

Naming quirk in this batch: a seed-3072 arm log carries NO suffix
(rollout_dinowm_um_structpos_pw30.log) while the baseline carries _s3072
(rollout_dinowm_um_fr_s3072.log). grounded_const and grounded2_const are the
same configuration launched twice (the "2" is literally a second launch), so
the merged a=g row pools both names -- 4 draws where both exist.
"""
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
LOG = _HERE.parents[3] / "raw_data" / "runs" / "dinowm"

DOMS = ["um", "par", "col"]
PARTITION = {"um": "both-OOD", "par": "r/m-OOD", "col": "both-OOD"}

# (key(s), display label) -- 9 rows; the a=g row pools its duplicate launch.
ROWS = [
    (["structpos_plain"], "[slot] structpos"),
    (["structpos_pw30"], "[slot] +reweight\n(w=30)"),
    (["posvel_pw30"], "[slot] +velocity"),
    (["probeF2"], "[probe] probe"),
    (["probe_structpos_pw30"], "[probe] +slot"),
    (["dyn_mlp"], "[dyn] free MLP"),
    (["grounded_const", "grounded2_const"], "[dyn] strict a=g\n([free] grounded)"),
    (["cons"], "[cons] consistency"),
    (["labelfree_const"], "[free] label-free"),
]


def _nmse(tag, part):
    p = LOG / f"rollout_{tag}.log"
    if not p.exists():
        return None
    pat = re.compile(r"^\s*" + re.escape(part) + r"\s+n=\s*\d+\s+cos=[+\-\d.]+\s+nMSE=([\d.]+)")
    for line in p.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            return float(m.group(1))
    return None


def draws(keys, dom):
    """All available per-seed nMSE values for a row (pooled over duplicate names)."""
    part = PARTITION[dom]
    out = []
    for key in keys:
        for tag in (f"dinowm_{dom}_{key}", f"dinowm_{dom}_{key}_s1234", f"dinowm_{dom}_{key}_s42"):
            v = _nmse(tag, part)
            if v is not None:
                out.append(v)
    return out


def baseline_draws(dom):
    part = PARTITION[dom]
    return [v for s in (3072, 1234, 42)
            if (v := _nmse(f"dinowm_{dom}_fr_s{s}", part)) is not None]


if __name__ == "__main__":
    import numpy as np
    w = n = b = 0
    for keys, lab in ROWS:
        cells = []
        for dom in DOMS:
            d, bd = draws(keys, dom), baseline_draws(dom)
            r = np.mean(d) / np.mean(bd)
            if min(d) > max(bd):
                v = "worse"; w += 1
            elif max(d) < min(bd):
                v = "BETTER"; b += 1
            else:
                v = "noise"; n += 1
            cells.append(f"{dom}:{r:.2f}x/{v}/n{len(d)}")
        print(f"{lab.split(chr(10))[0]:<26} {'  '.join(cells)}")
    print(f"\nverdicts: {w} worse / {n} within-noise / {b} better  ({w+n} of {w+n+b} at or below)")
