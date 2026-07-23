"""Aggregate causal_route_eval logs into seed-averaged tables.

Reads every log under the given directories, parses the three intervention
families, and prints markdown tables with mean +/- sample std over seeds.
Coordinate 0 is the physically moving coordinate in every domain
(uniform/parabola: the ball's x; collision: ball 1's x).
"""
import argparse, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PAIR = r"\(([-+0-9.]+),([-+0-9.]+)\)"


def parse(path):
    txt = path.read_text()
    out = {"file": path.name}
    m = re.search(r"tag=(\S+)\s+mode=(\S+)\s+clamp=(\S+)", txt)
    if not m:
        return None
    out["tag"], out["mode"], out["clamp"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"--domain\s+(\S+)", txt) or re.search(r"causal_route/\w+/\w+_(\w+?)_", str(path))
    # domain is encoded in the filename: <mode>_<domain>_<tag>.log
    parts = path.stem.split("_")
    for d in ("uniform_motion", "parabola", "collision"):
        if d in path.stem:
            out["domain"] = d
            break
    else:
        out["domain"] = "?"
    m = re.search(r"probe R2 on train\]\s+full=([\d.]+)\s+slot=([\d.]+)\s+bb=([\d.]+)", txt)
    if m:
        out["r2_full"], out["r2_slot"], out["r2_bb"] = map(float, m.groups())
    m = re.search(r"clean latent nMSE.*?:\s*([\d.]+)", txt)
    if m:
        out["clean_nmse"] = float(m.group(1))
    m = re.search(r"clean decoded rho from PRED latents.*?:\s*([-+\d.]+)\s*/\s*([-+\d.]+)", txt)
    if m:
        out["clean_rho"] = [float(m.group(1)), float(m.group(2))]

    # ---- steering summary ----
    for name, key in [("slot", "g_slot"), ("blackbox", "g_bb"),
                      ("joint", "g_joint"), (r"rand-bb\(norm-matched\)", "g_rand")]:
        m = re.search(rf"^\s+{name}\s+read-full={PAIR}\s+read-bb={PAIR}\s+read-slot={PAIR}",
                      txt, re.M)
        if m:
            g = [float(x) for x in m.groups()]
            out[key + "_full"], out[key + "_bb"], out[key + "_slot"] = g[0:2], g[2:4], g[4:6]

    # ---- patch ----
    blocks = re.split(r"^  (zero|donor-slot|donor-rand2|freeze|donor-bb)\s+nMSE=([\d.]+)",
                      txt, flags=re.M)
    for i in range(1, len(blocks) - 2, 3):
        kind, nm, body = blocks[i], float(blocks[i + 1]), blocks[i + 2]
        b = re.search(r"read-bb\s+rho vs OWN pos = ([-+\d.]+)/([-+\d.]+)", body)
        if b:
            out[f"patch_{kind}_rho"] = [float(b.group(1)), float(b.group(2))]
        out[f"patch_{kind}_nmse"] = nm
        seg = body.split("read-bb")[1] if "read-bb" in body else ""
        f0 = re.search(r"coord0:.*?follow-fraction\(donor\) = ([\d.]+)", seg)
        if f0:
            out[f"patch_{kind}_follow0"] = float(f0.group(1))

    # ---- amnesic ----
    m = re.search(r"INLP removed rank (\d+) of (\d+) to drive position R2 "
                  r"([\d.]+) -> ([\d.]+)", txt)
    if m:
        out["inlp_rank"], out["inlp_of"] = int(m.group(1)), int(m.group(2))
        out["inlp_r2_0"], out["inlp_r2_1"] = float(m.group(3)), float(m.group(4))
    for label, key in [(r"bb: position removed \(INLP\)", "am_pos"),
                       (r"bb: random rank-\d+ \(control\)", "am_rand"),
                       (r"slot: position removed", "am_slot")]:
        m = re.search(rf"{label}\s+nMSE=([\d.]+)\s*\n\s+read-full\s+rho=([-+\d.]+)/([-+\d.]+)"
                      rf"\s+\(clean [^)]*\)\s*\n\s+read-bb\s+rho=([-+\d.]+)/([-+\d.]+)", txt)
        if m:
            out[key + "_nmse"] = float(m.group(1))
            out[key + "_rho_bb"] = [float(m.group(4)), float(m.group(5))]

    # ---- subset table on rolled-out latents ----
    for name in ["all[192]", "slot[0:2]", "blackbox[2:192]", "rand-2", "rand-10"]:
        m = re.search(rf"^\s+{re.escape(name)}\s+([-+\d.]+)\s+([-+\d.]+)\s+\|", txt, re.M)
        if m:
            out[f"sub_{name}_real"] = float(m.group(1))
            out[f"sub_{name}_pred"] = float(m.group(2))
    return out


def agg(rows, key, coord=0):
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        vals.append(v[coord] if isinstance(v, list) else v)
    if not vals:
        return None
    return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0), len(vals)


def fmt(a, prec=3):
    if a is None:
        return "--"
    m, s, n = a
    return f"{m:+.{prec}f}" + (f"±{s:.{prec}f}" if n > 1 else "") + (f" ({n})" if n != 3 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        for p in sorted(Path(d).glob("*.log")):
            try:
                r = parse(p)
            except Exception as e:                       # noqa: BLE001
                print(f"[warn] {p.name}: {e}", file=sys.stderr); continue
            if r:
                rows.append(r)
    print(f"parsed {len(rows)} logs\n")

    # group by (domain, arm) where arm = tag minus the seed suffix
    def arm(t):
        return re.sub(r"_s\d+$", "", t)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["domain"], arm(r["tag"]), r["mode"], r["clamp"])].append(r)

    doms = ["uniform_motion", "parabola", "collision"]
    arms = ["lewm_baseline", "lewm_structpos", "dino_baseline", "dino_structpos",
            "lewm_structdyn"]

    print("## 1. Steering: causal gain on the model's own black-box state (coord 0)\n")
    print("| arm | domain | slot R2 | g_slot | g_bb | g_rand (control) | "
          "g_slot+g_bb | g_joint | seeds |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        for d in doms:
            rs = grp.get((d, a, "steer", "every-step"), []) + \
                 grp.get((d, a, "steer_small", "every-step"), [])
            if not rs:
                continue
            gs, gb = agg(rs, "g_slot_bb"), agg(rs, "g_bb_bb")
            add = (gs[0] + gb[0]) if gs and gb else None
            print(f"| {a} | {d} | {fmt(agg(rs,'r2_slot'),3)} | {fmt(gs)} | {fmt(gb)} | "
                  f"{fmt(agg(rs,'g_rand_bb'))} | "
                  f"{('%+.3f' % add) if add is not None else '--'} | "
                  f"{fmt(agg(rs,'g_joint_bb'))} | {len(rs)} |")

    # A raw g_bb is not interpretable: min-norm steering of a redundant 190-d code is
    # diluted, so even a model where the black box is the ONLY route reads low. The
    # baseline arm IS such a model (its dims[0:2] carry nothing), so its g_bb is the
    # method's full-scale reading for a fully load-bearing black box. Normalise by it.
    print("\n### 1b. Black-box gain normalised by the baseline arm (its full-scale reading)\n")
    print("| domain | g_bb baseline (= full scale) | g_bb structpos | ratio |")
    print("|---|---|---|---|")
    for d in doms:
        b = agg(grp.get((d, "lewm_baseline", "steer", "every-step"), []), "g_bb_bb")
        s = agg(grp.get((d, "lewm_structpos", "steer", "every-step"), []), "g_bb_bb")
        if not (b and s):
            continue
        print(f"| {d} | {fmt(b)} | {fmt(s)} | "
              f"{s[0]/b[0]:.2f}x |" if abs(b[0]) > 1e-6 else f"| {d} | {fmt(b)} | {fmt(s)} | -- |")

    print("\n## 2. Counterfactual patching, read on the black box (coord 0)\n")
    print("| arm | domain | clamp | clean rho | freeze slot | donor slot | "
          "donor rand-2 (control) | follow(donor slot) | follow(donor rand2) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        for d in doms:
            for cl in ("every-step", "ctx-only"):
                rs = grp.get((d, a, "patch", cl), [])
                if not rs:
                    continue
                print(f"| {a} | {d} | {cl} | {fmt(agg(rs,'clean_rho'))} | "
                      f"{fmt(agg(rs,'patch_freeze_rho'))} | "
                      f"{fmt(agg(rs,'patch_donor-slot_rho'))} | "
                      f"{fmt(agg(rs,'patch_donor-rand2_rho'))} | "
                      f"{fmt(agg(rs,'patch_donor-slot_follow0'))} | "
                      f"{fmt(agg(rs,'patch_donor-rand2_follow0'))} |")

    print("\n## 3. Amnesic projection: how redundant is the black-box position code\n")
    print("| arm | domain | rank to erase / 190 | R2 before->after | "
          "rho after pos-removal | rho after random rank-matched | rho after slot removal |")
    print("|---|---|---|---|---|---|---|")
    for a in arms:
        for d in doms:
            rs = grp.get((d, a, "amnesic", "every-step"), [])
            if not rs:
                continue
            r0, r1 = agg(rs, "inlp_r2_0"), agg(rs, "inlp_r2_1")
            print(f"| {a} | {d} | {fmt(agg(rs,'inlp_rank'),1)} | "
                  f"{r0[0]:.3f}->{r1[0]:.3f} | {fmt(agg(rs,'am_pos_rho_bb'))} | "
                  f"{fmt(agg(rs,'am_rand_rho_bb'))} | {fmt(agg(rs,'am_slot_rho_bb'))} |")

    print("\n## 4. Bypass table on ROLLED-OUT latents (rho, coord-mean)\n")
    print("| arm | domain | all 192 REAL | all 192 PRED | black box PRED | "
          "slot PRED | rand-2 PRED |")
    print("|---|---|---|---|---|---|---|")
    for a in arms:
        for d in doms:
            rs = grp.get((d, a, "subset", "every-step"), [])
            if not rs:
                continue
            print(f"| {a} | {d} | {fmt(agg(rs,'sub_all[192]_real'))} | "
                  f"{fmt(agg(rs,'sub_all[192]_pred'))} | "
                  f"{fmt(agg(rs,'sub_blackbox[2:192]_pred'))} | "
                  f"{fmt(agg(rs,'sub_slot[0:2]_pred'))} | "
                  f"{fmt(agg(rs,'sub_rand-2_pred'))} |")


if __name__ == "__main__":
    main()
