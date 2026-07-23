"""Three-seed Physion++ injection table (paper Table 5).

Table 5 was single-seed (3072). This pools the seed-1234/42 top-ups launched
2026-07-22 and applies the same verdict rule the two scan figures use: a cell
counts as worse only if every seed exceeds every baseline seed.

Baseline caveat this exists to respect: the three-seed FR baseline already in
the repo (pp_fr_s*) trains at batch 32, while every arm in Table 5 trains at
64. Mixing them would put a numerator and a denominator under different
settings, so FR was re-run at 64 (pp_fr_bs64_s*) and only those are used here.
The seed-3072 draw for each arm comes from the original runs (pp_fr, pp_struct,
pp_cons, pp_consacc), which are already batch 64.
"""
import os
import re
import statistics as st

LOG = "/data1/likun-share/junjxu/runs/physionpp"

# scenarios shown in Table 5
SCEN = ["friction_platform", "mass_collision", "mass_dominoes", "mass_waterpush"]

# arm -> {seed: eval-log tag}
ARMS = {
    "free rollout": {3072: "pp_fr_e20", 1234: "pp_fr_bs64_s1234", 42: "pp_fr_bs64_s42"},
    "+slot": {3072: "pp_struct_e20", 1234: "pp_struct_s1234", 42: "pp_struct_s42"},
    "+cons.": {3072: "pp_cons_e20", 1234: "pp_cons_s1234", 42: "pp_cons_s42"},
    "+cons.+acc.": {3072: "pp_consacc_e20", 1234: "pp_consacc_s1234", 42: "pp_consacc_s42"},
}


def per_scene(tag):
    """-> {scenario: nMSE}; tries the eval_ prefix then the rollout_ prefix."""
    for name in (f"{LOG}/eval_{tag}.log", f"{LOG}/rollout_{tag}.log"):
        if not os.path.exists(name):
            continue
        out = {}
        for line in open(name, errors="ignore"):
            m = re.match(r"\s+([a-z]+_[a-z]+)\s+n=\s*\d+\s+cos=[+-][\d.]+\s+nMSE=([\d.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
        if out:
            return out
    return {}


def draws(arm, scen):
    vals = []
    for seed, tag in ARMS[arm].items():
        d = per_scene(tag)
        if scen in d:
            vals.append(d[scen])
    return vals


def fmt(v):
    if not v:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.3f}(n1)"
    return f"{st.mean(v):.3f}±{st.stdev(v):.3f}"


if __name__ == "__main__":
    print("Physion++ per-scenario rollout nMSE, three-seed mean±std\n")
    hdr = f"{'scenario':<20}" + "".join(f"{a:>18}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    complete = True
    for s in SCEN:
        cells = [draws(a, s) for a in ARMS]
        if any(len(c) < 3 for c in cells):
            complete = False
        print(f"{s:<20}" + "".join(fmt(c).rjust(18) for c in cells))

    print()
    base = "free rollout"
    for a in ARMS:
        if a == base:
            continue
        worse = sep = 0
        for s in SCEN:
            d, b = draws(a, s), draws(base, s)
            if len(d) < 3 or len(b) < 3:
                continue
            worse += 1
            if min(d) > max(b):
                sep += 1
        print(f"  {a:<14} worse in every seed on {sep}/{worse} scenarios")

    if not complete:
        print("\n[incomplete] some arms still lack three seeds -- reruns still training")
