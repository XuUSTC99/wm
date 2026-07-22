"""Summarise the causal follow-up sweeps.

A  -- structdyn yardstick: does g land near 1 once the steering vector is
      written straight into the slot, and is it sensitive to delta?
B  -- parabola delta ladder per seed: does the across-seed spread shrink as
      delta shrinks (an out-of-linear-range artifact) or persist (real)?
B2 -- the same ladder on uniform, which is stable, so that a delta-dependence
      on parabola can be told apart from one the method has everywhere.

Reads /data1/likun-share/junjxu/runs/causal_route/followup/.
"""
import os
import re
import numpy as np

L = "/data1/likun-share/junjxu/runs/causal_route/followup"
FOCUS = {"uniform_motion": 0, "parabola": 1, "collision": 0}


def read(name):
    """-> dict with g_slot/g_bb/g_joint/g_rand on read-bb, both coords, + probe R2"""
    p = f"{L}/{name}.log"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, errors="ignore"):
        m = re.search(r"probe R2 on train\]\s+full=([\d.]+)\s+slot=([\d.]+)\s+bb=([\d.]+)", line)
        if m:
            out["r2_slot"] = float(m.group(2))
        m = re.match(r"\s+(slot|blackbox|joint|rand-bb)\S*\s+.*read-bb=\(([-+][\d.]+),([-+][\d.]+)\)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out if "slot" in out else None


def g(name, key, coord):
    d = read(name)
    if d is None or key not in d:
        return None
    return d[key][coord]


print("=" * 84)
print("A. structdyn yardstick -- position can only reach prediction through the slot,")
print("   so g should be ~1 if the measurement is calibrated.")
print("=" * 84)
print(f"{'domain':<16}{'construction':<10}{'delta':<8}{'g_slot':>12}{'g_bb':>12}{'g_joint':>12}")
for dom in ("uniform_motion", "parabola", "collision"):
    c = FOCUS[dom]
    for tag, lab, dl in (("direct_d02", "direct", "0.2"),
                         ("direct_d10", "direct", "1.0"),
                         ("pinv_d02", "pinv", "0.2")):
        n = f"A_{dom}_structdyn_{tag}"
        gs, gb, gj = g(n, "slot", c), g(n, "blackbox", c), g(n, "joint", c)
        if gs is None:
            continue
        f = lambda v: f"{v:+.3f}" if v is not None else "—"
        print(f"{dom:<16}{lab:<10}{dl:<8}{f(gs):>12}{f(gb):>12}{f(gj):>12}")

print()
print("=" * 84)
print("B. parabola delta ladder (focus coord1, the gravity axis) -- per seed")
print("=" * 84)
LADDER = [("d005", "0.05"), ("d02", "0.2"), ("d05", "0.5"), ("d10", "1.0")]
for grp, dom, pref in (("B", "parabola", "B_parabola_structpos"),
                       ("B2", "uniform_motion", "B2_uniform_structpos")):
    c = FOCUS[dom]
    print(f"\n--- {dom} (coord{c}) ---")
    print(f"{'delta':<8}{'s3072':>10}{'s1234':>10}{'s42':>10}{'mean':>12}{'std':>10}{'std/|mean|':>12}")
    for tag, dl in LADDER:
        vals = []
        for s in (3072, 1234, 42):
            vals.append(g(f"{pref}_s{s}_{tag}", "slot", c))
        got = [v for v in vals if v is not None]
        if not got:
            continue
        m, sd = np.mean(got), (np.std(got, ddof=1) if len(got) > 1 else float("nan"))
        rel = abs(sd / m) if m else float("nan")
        cells = "".join(f"{v:+.3f}".rjust(10) if v is not None else "—".rjust(10) for v in vals)
        print(f"{dl:<8}{cells}{m:>12.3f}{sd:>10.3f}{rel:>12.2f}")

print()
print("Reading the last column: std/|mean| is the across-seed spread relative to the")
print("effect. If it falls as delta falls, the instability was an out-of-range")
print("artifact. If it stays high at the smallest delta, it is real.")
