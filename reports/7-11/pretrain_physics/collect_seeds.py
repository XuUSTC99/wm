#!/usr/bin/env python3
"""Wait for the 12 seed runs to land, then rebuild paper Table 2 with 3 seeds.

Writes SEED_RESULTS.md next to the logs, so the summary exists even if the
launching session is gone. Judged partition per domain matches the paper:
uniform/collision -> both-OOD, parabola -> r/m-OOD (its both-OOD aggregate is
invalidated by the nMSE denominator blow-up).
"""
import re, time, statistics as st
from pathlib import Path

LOG = Path("/data1/likun-share/junjxu/runs/pretrain_physics")
OUT = LOG / "SEED_RESULTS.md"
DOMS = [("um", "uniform", "both-OOD"), ("par", "parabola", "r/m-OOD"), ("col", "collision", "both-OOD")]
SEED0 = {  # existing seed-3072 values (verified against the archived logs)
    ("um", "off"): 0.1915, ("um", "on"): 0.7500,
    ("par", "off"): 0.3430, ("par", "on"): 0.3747,
    ("col", "off"): 0.5378, ("col", "on"): 0.6351,
}
NEW = [(d, p, s) for d, _, _ in DOMS for p in ("off", "on") for s in (1234, 42)]


def read(short, phys, seed, part):
    f = LOG / f"rollout_pp2_{short}_scratch_{phys}_s{seed}.log"
    if not f.exists():
        return None
    for line in f.read_text(errors="ignore").splitlines():
        if re.match(rf"\s*{re.escape(part)}\s", line) and "nMSE" in line:
            m = re.search(r"nMSE=([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def done():
    return all(read(d, p, s, dict((x[0], x[2]) for x in DOMS)[d]) is not None for d, p, s in NEW)


deadline = time.time() + 6 * 3600
while not done() and time.time() < deadline:
    time.sleep(120)

lines = ["# Table 2 seed replication (auto-generated)", ""]
lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | complete: {done()}")
lines.append("")
lines.append("| domain | phys | s3072 | s1234 | s42 | mean±std | n |")
lines.append("|---|---|---|---|---|---|---|")
cell = {}
for short, name, part in DOMS:
    for phys in ("off", "on"):
        vals = [SEED0[(short, phys)], read(short, phys, 1234, part), read(short, phys, 42, part)]
        got = [v for v in vals if v is not None]
        cell[(short, phys)] = got
        ms = f"**{st.mean(got):.3f}±{st.stdev(got):.3f}**" if len(got) > 1 else f"{got[0]:.3f}"
        fmt = lambda v: f"{v:.4f}" if v is not None else "—"
        lines.append(f"| {name} | {phys} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {ms} | {len(got)} |")

lines += ["", "## Δ (injection cost) per domain", "",
          "| domain | Δ mean | 95% CI (Welch) | verdict |", "|---|---|---|---|"]
for short, name, _ in DOMS:
    off, on = cell[(short, "off")], cell[(short, "on")]
    if len(off) > 1 and len(on) > 1:
        d = st.mean(on) - st.mean(off)
        se = (st.stdev(on) ** 2 / len(on) + st.stdev(off) ** 2 / len(off)) ** 0.5
        lo, hi = d - 2.78 * se, d + 2.78 * se   # t(0.975, df~4)
        v = "positive (injection hurts)" if lo > 0 else ("negative (injection helps!)" if hi < 0 else "**overlaps zero — parity**")
        lines.append(f"| {name} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {v} |")
    else:
        lines.append(f"| {name} | (incomplete) | — | — |")

OUT.write_text("\n".join(lines) + "\n")
print(OUT)
