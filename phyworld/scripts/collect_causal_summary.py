"""Collect every causal-route intervention log into one table.

Reads /data1/likun-share/junjxu/runs/causal_route/{full,wave2,v2,v4,v5}/ and
prints steer / patch / amnesic results per domain x seed x backbone x arm.

Both coordinates are reported. Coordinate choice is not cosmetic here: gravity
acts on the vertical axis, so for parabola the informative coordinate is
coord1, while for uniform motion and collision the motion under test is
horizontal (coord0). Reading one fixed coordinate across all three domains is
what made parabola look unstable in the single-seed pilot.

Usage:  python collect_causal_summary.py [--md]
"""
import argparse
import os
import re
import numpy as np

R = "/data1/likun-share/junjxu/runs/causal_route"
SUBS = ("full", "wave2", "v5", "v4", "v2")
DOMS = ["uniform_motion", "parabola", "collision"]
SEEDS = [3072, 1234, 42]
# the coordinate whose motion the intervention is meant to move
FOCUS = {"uniform_motion": 0, "parabola": 1, "collision": 0}


def find(kind, dom, model, arm, seed):
    for sub in SUBS:
        p = f"{R}/{sub}/{kind}_{dom}_{model}_{arm}_s{seed}.log"
        if os.path.exists(p):
            return p
    return None


def probe_r2(path):
    """[probe R2 on train] full=.. slot=.. bb=.."""
    for line in open(path, errors="ignore"):
        m = re.search(r"probe R2 on train\]\s+full=([\d.]+)\s+slot=([\d.]+)\s+bb=([\d.]+)", line)
        if m:
            return dict(full=float(m.group(1)), slot=float(m.group(2)), bb=float(m.group(3)))
    return None


def steer(path):
    """gains on the read-bb column (the load-bearing readout), both coords."""
    out = {}
    for line in open(path, errors="ignore"):
        m = re.match(r"\s+(slot|blackbox|joint|rand-bb)\S*\s+.*read-bb=\(([-+][\d.]+),([-+][\d.]+)\)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def patch(path):
    """donor-slot follow fraction on read-bb, both coords."""
    out, in_donor, read_bb = {}, False, False
    for line in open(path, errors="ignore"):
        if re.match(r"\s+donor-slot", line):
            in_donor, read_bb = True, False
        elif re.match(r"\s+(zero|donor-bb|donor-rand)", line):
            in_donor = False
        if in_donor and "read-bb" in line:
            read_bb = True
        elif in_donor and re.search(r"read-(full|slot)", line):
            read_bb = False
        if in_donor and read_bb:
            m = re.search(r"coord(\d): .*follow-fraction\(donor\) = ([\d.]+)", line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def amnesic(path):
    """rank needed to erase position from the black box, and selectivity."""
    out = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"INLP removed rank (\d+) of (\d+) to drive position R2 ([\d.]+) -> ([\d.]+)", line)
        if m:
            out.update(rank=int(m.group(1)), of=int(m.group(2)),
                       r2_before=float(m.group(3)), r2_after=float(m.group(4)),
                       capped="HIT ITERATION CAP" in line)
        m = re.search(r"bb: position removed \(INLP\)\s+nMSE=([\d.]+)", line)
        if m:
            out["nmse_pos"] = float(m.group(1))
        m = re.search(r"bb: random rank-\d+ \(control\)\s+nMSE=([\d.]+)", line)
        if m:
            out["nmse_rand"] = float(m.group(1))
    return out


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:+.3f}(n=1)"
    return f"{np.mean(v):+.3f}±{np.std(v, ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown tables")
    a = ap.parse_args()
    B = "| " if a.md else "  "
    S = " | " if a.md else "  "

    for model in ("lewm", "dino"):
        print(f"\n{'='*88}\n{model.upper()}\n{'='*88}")

        # ---- validity gate: does the slot actually carry position? ----
        print("\n[gate] probe R2 (train) -- slot must carry position for steering to be readable")
        if a.md:
            print("| domain | arm | slot R2 | bb R2 | full R2 |")
            print("|---|---|---|---|---|")
        for dom in DOMS:
            for arm in ("baseline", "structpos"):
                rs = []
                for s in SEEDS:
                    p = find("steer", dom, model, arm, s)
                    if p:
                        r = probe_r2(p)
                        if r:
                            rs.append(r)
                if not rs:
                    continue
                f = lambda k: np.mean([r[k] for r in rs])
                print(f"{B}{dom:<15}{S}{arm:<10}{S}{f('slot'):.3f}{S}{f('bb'):.3f}{S}{f('full'):.3f}"
                      + (" |" if a.md else "") + f"   (n={len(rs)})")

        # ---- steering ----
        print("\n[steer] causal gain g on read-bb; focus coord marked *")
        if a.md:
            print("| domain | arm | g_slot (focus) | g_bb (focus) | random ctrl | additivity |")
            print("|---|---|---|---|---|---|")
        for dom in DOMS:
            c = FOCUS[dom]
            for arm in ("baseline", "structpos"):
                gs, gb, gr, gj = [], [], [], []
                for s in SEEDS:
                    p = find("steer", dom, model, arm, s)
                    if not p:
                        continue
                    d = steer(p)
                    gs.append(d.get("slot", (None, None))[c])
                    gb.append(d.get("blackbox", (None, None))[c])
                    gr.append(d.get("rand-bb", (None, None))[c])
                    gj.append(d.get("joint", (None, None))[c])
                if not any(x is not None for x in gs):
                    continue
                sm = np.mean([x for x in gs if x is not None])
                bm = np.mean([x for x in gb if x is not None])
                jm = np.mean([x for x in gj if x is not None]) if any(gj) else float("nan")
                add = f"{sm+bm:+.2f} vs {jm:+.2f}"
                print(f"{B}{dom:<15}{S}{arm:<10}{S}{agg(gs):>15}{S}{agg(gb):>15}{S}{agg(gr):>15}{S}{add}"
                      + (" |" if a.md else "") + f"  coord{c}*")

        # ---- per-seed detail for steering (where the instability lives) ----
        print("\n[steer per-seed] structpos, focus coord -- g_slot by seed")
        for dom in DOMS:
            c = FOCUS[dom]
            row = []
            for s in SEEDS:
                p = find("steer", dom, model, "structpos", s)
                v = steer(p).get("slot", (None, None))[c] if p else None
                row.append(f"s{s}={v:+.3f}" if v is not None else f"s{s}=—")
            # also the other coordinate, to show the coordinate choice matters
            other = 1 - c
            row2 = []
            for s in SEEDS:
                p = find("steer", dom, model, "structpos", s)
                v = steer(p).get("slot", (None, None))[other] if p else None
                row2.append(f"{v:+.3f}" if v is not None else "—")
            print(f"  {dom:<15} coord{c}*: {'  '.join(row)}")
            print(f"  {'':<15} coord{other} : {'  '.join(row2)}")

        # ---- counterfactual patch ----
        print("\n[patch] donor-slot follow-fraction on read-bb (1.0 = prediction fully follows donor)")
        if a.md:
            print("| domain | arm | follow (focus) | follow (other) |")
            print("|---|---|---|---|")
        for dom in DOMS:
            c = FOCUS[dom]
            for arm in ("baseline", "structpos"):
                ff, fo = [], []
                for s in SEEDS:
                    p = find("patch", dom, model, arm, s)
                    if not p:
                        continue
                    d = patch(p)
                    ff.append(d.get(c))
                    fo.append(d.get(1 - c))
                if not any(x is not None for x in ff):
                    continue
                print(f"{B}{dom:<15}{S}{arm:<10}{S}{agg(ff):>15}{S}{agg(fo):>15}"
                      + (" |" if a.md else ""))

        # ---- amnesic ----
        print("\n[amnesic] rank needed to erase position from the 190-d black box")
        if a.md:
            print("| domain | arm | rank/190 | R2 before->after | capped | nMSE pos-removed | nMSE random ctrl |")
            print("|---|---|---|---|---|---|---|")
        for dom in DOMS:
            for arm in ("baseline", "structpos"):
                recs = []
                for s in SEEDS:
                    p = find("amnesic", dom, model, arm, s)
                    if p:
                        d = amnesic(p)
                        if d:
                            recs.append(d)
                if not recs:
                    continue
                rk = np.mean([r["rank"] for r in recs])
                of = recs[0].get("of", 190)
                b4 = np.mean([r["r2_before"] for r in recs])
                af = np.mean([r["r2_after"] for r in recs])
                cap = sum(r.get("capped", False) for r in recs)
                np_ = np.mean([r["nmse_pos"] for r in recs if "nmse_pos" in r]) if any("nmse_pos" in r for r in recs) else float("nan")
                nr = np.mean([r["nmse_rand"] for r in recs if "nmse_rand" in r]) if any("nmse_rand" in r for r in recs) else float("nan")
                print(f"{B}{dom:<15}{S}{arm:<10}{S}{rk:.0f}/{of}{S}{b4:.3f}->{af:.3f}{S}"
                      f"capped {cap}/{len(recs)}{S}{np_:.3f}{S}{nr:.3f}" + (" |" if a.md else ""))


if __name__ == "__main__":
    main()
