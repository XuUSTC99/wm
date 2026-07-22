"""Black-box dose-response: how many black-box dimensions equal the 2-d slot?

For each model we patch k black-box dimensions from a donor trajectory (slot
left at its own value) and measure how far the rollout follows the donor. The
same run also patches the 2-d slot, so the two channels are measured on one
scale, in one model, against one donor.

The headline number is the crossing point: the k at which moving k black-box
dimensions buys as much follow-fraction as moving the 2 slot dimensions does.
It is reported by linear interpolation on the measured ladder, and as a bound
when the curve never reaches the slot's level.

Read on read-bb -- the black-box readout. Reading a channel back after clamping
it is partly trivial (the predictor copies its input forward); the cross terms
are the informative ones.
"""
import os
import re
import sys
import numpy as np

# every-step by default; pass "ctx" to read the context-only ladder, which asks
# whether the counterfactual is CARRIED rather than whether the channel is read
# at steady state.
_MODE = sys.argv[1] if len(sys.argv) > 1 else "step"
R = ("/data1/likun-share/junjxu/runs/causal_route/bbdose_ctx" if _MODE == "ctx"
     else "/data1/likun-share/junjxu/runs/causal_route/bbdose")
_PREFIX = "ctx" if _MODE == "ctx" else "bbdose"
DOSES = [2, 10, 40, 100, 190]
FOCUS = {"uniform_motion": 0, "parabola": 1, "collision": 0}
SEEDS = [3072, 1234, 42]


def parse(path):
    """arm -> {coord: follow-fraction on read-bb, 'nmse': ...}"""
    out, arm, sect = {}, None, None
    for line in open(path, errors="ignore"):
        m = re.match(r"\s+(zero|donor-slot|donor-rand2|freeze|donor-bb|donor-bbk\d+)\s+nMSE=([\d.]+)", line)
        if m:
            arm = m.group(1)
            out.setdefault(arm, {})["nmse"] = float(m.group(2))
            sect = None
            continue
        if arm and re.search(r"read-(full|bb|slot)", line):
            sect = re.search(r"read-(full|bb|slot)", line).group(1)
            continue
        if arm and sect == "bb":
            m = re.search(r"coord(\d): .*follow-fraction\(donor\) = ([\d.]+)", line)
            if m:
                out[arm][int(m.group(1))] = float(m.group(2))
    return out


def load(dom, model, armname, seed):
    p = f"{R}/{_PREFIX}_{dom}_{model}_{armname}_s{seed}.log"
    return parse(p) if os.path.exists(p) else None


def crossing(ks, ys, target):
    """smallest k whose follow-fraction reaches target, linearly interpolated"""
    if target is None or not ys:
        return None
    for i, (k, y) in enumerate(zip(ks, ys)):
        if y >= target:
            if i == 0:
                return float(k)
            k0, y0 = ks[i - 1], ys[i - 1]
            if y == y0:
                return float(k)
            return float(k0 + (k - k0) * (target - y0) / (y - y0))
    return None  # never reaches it


print(f"[{'ctx-only (is the counterfactual carried?)' if _MODE=='ctx' else 'every-step (steady-state sensitivity)'}]")
print("Follow-fraction on read-bb, focus coordinate, three-seed mean")
print("slot = patching the 2-d slot;  k = patching k black-box dims (slot untouched)\n")

for model in ("lewm", "dino"):
    print("=" * 92)
    print(f"{model.upper()}")
    print("=" * 92)
    hdr = f"{'domain':<16}{'arm':<11}{'slot(2d)':>10}" + "".join(f"{'k='+str(k):>9}" for k in DOSES) + f"{'slot≈k dims':>14}"
    print(hdr)
    for dom in ("uniform_motion", "parabola", "collision"):
        c = FOCUS[dom]
        for armname in ("baseline", "structpos"):
            slot_v, dose_v = [], {k: [] for k in DOSES}
            n = 0
            for s in SEEDS:
                d = load(dom, model, armname, s)
                if d is None:
                    continue
                n += 1
                if "donor-slot" in d and c in d["donor-slot"]:
                    slot_v.append(d["donor-slot"][c])
                for k in DOSES:
                    key = f"donor-bbk{k}"
                    if key in d and c in d[key]:
                        dose_v[k].append(d[key][c])
            if n == 0:
                continue
            sm = np.mean(slot_v) if slot_v else None
            ys = [np.mean(dose_v[k]) if dose_v[k] else None for k in DOSES]
            ks_ok = [k for k, y in zip(DOSES, ys) if y is not None]
            ys_ok = [y for y in ys if y is not None]
            x = crossing(ks_ok, ys_ok, sm)
            cells = "".join(f"{y:+.3f}".rjust(9) if y is not None else "—".rjust(9) for y in ys)
            xs = f"{x:.0f} dims" if x is not None else (">190" if sm is not None else "—")
            print(f"{dom:<16}{armname:<11}{(f'{sm:+.3f}' if sm is not None else '—'):>10}{cells}{xs:>14}  n={n}")
    print()

print("=" * 92)
print("VALIDITY GATE -- a structpos crossing means nothing unless its baseline passes both:")
print("  (1) negative control: patching the 2 slot dims of a model that HAS no slot must")
print("      do ~nothing, so baseline slot(2d) should sit near 0.")
print("  (2) positive control: the black box is the only route in a baseline model, so")
print("      replacing all of it must make the rollout follow the donor, k=190 -> ~1.")
print("=" * 92)
print(f"{'backbone/domain':<26}{'baseline slot':>14}{'baseline k=190':>16}   verdict")
for model in ("lewm", "dino"):
    for dom in ("uniform_motion", "parabola", "collision"):
        c = FOCUS[dom]
        sl, k190 = [], []
        for s in SEEDS:
            d = load(dom, model, "baseline", s)
            if not d:
                continue
            if "donor-slot" in d and c in d["donor-slot"]:
                sl.append(d["donor-slot"][c])
            key = f"donor-bbk190"
            if key in d and c in d[key]:
                k190.append(d[key][c])
        if not sl or not k190:
            continue
        a, b = np.mean(sl), np.mean(k190)
        ok1, ok2 = a < 0.15, b > 0.85
        v = "USABLE" if (ok1 and ok2) else (
            "unusable: " + ", ".join(
                x for x in (("neg ctrl dirty" if not ok1 else ""),
                            ("pos ctrl fails" if not ok2 else "")) if x))
        print(f"{model+'/'+dom:<26}{a:>14.3f}{b:>16.3f}   {v}")
print()
print("Reading the result: 'slot≈k dims' is how much of the 190-d black box has to move")
print("to match what moving the 2-d slot does -- but only in rows whose baseline is USABLE.")
