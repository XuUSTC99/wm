#!/usr/bin/env python3
"""Extract λ_probe × frames sweep results across 3 domains.

Reads (in order — first match wins per (dom,w,f)):
  /data1/likun-share/junjxu/runs/sweep_three_domains_logs/rollout_<dom>_w<W>_f<F>.log
  /data1/likun-share/junjxu/runs/sweep_three_domains_extend_logs/rollout_<dom>_w<W>_f<F>.log

Weights: {0.1, 1.0, 10.0} from the original sweep + {30.0, 50.0} from the extend sweep.
Frames:  {1, 2, 4}.

Output:
  - Per-domain markdown tables: rows = weight, cols = frames
  - One table per (metric, partition) pair (default: vx K=4, vy K=4, cos@h16)
  - Best-(w,f) summary per metric

Usage:
  python extract_sweep_results.py                  # print to stdout
  python extract_sweep_results.py > out.md         # save to file
"""
import argparse
import math
import re
from pathlib import Path

# -----------------------------------------------------------------------------
# Multiple log dirs: original sweep (w=0.1/1.0/10.0) + extend sweep (w=30.0/50.0).
# Order matters: when a (dom,w,f) log exists in both, the first wins.
LOG_DIRS = [
    Path('/data1/likun-share/junjxu/runs/sweep_three_domains_logs'),
    Path('/data1/likun-share/junjxu/runs/sweep_three_domains_extend_logs'),
]
DOMAINS = ['parabola', 'uniform', 'collision']  # prefix used in log filenames
DOMAIN_LABEL = {
    'parabola': 'parabola',
    'uniform':  'uniform_motion',
    'collision': 'collision',
}
WEIGHTS = ['0.1', '1.0', '10.0', '30.0', '50.0']
WTAG = {'0.1': '0p1', '1.0': '1p0', '10.0': '10p0', '30.0': '30p0', '50.0': '50p0'}
FRAMES = ['1', '2', '4']
PART_ORDER = ['ID', 'r-OOD', 'r/m-OOD', 'v-OOD', 'both-OOD']

# -----------------------------------------------------------------------------
def parse_log(path: Path) -> dict:
    """Extract K=4 partition + cos-by-horizon + cos-by-partition from a rollout log."""
    if not path.exists():
        return {}
    text = path.read_text()
    out = {'K4_partition': {}, 'cos_by_horizon': {}, 'cos_by_partition': {}}

    blk = re.search(r"\[K=4\] probe applied to PREDICTED embs[^\n]*\n((?:.*PRED-K4.*\n)+)", text)
    if blk:
        for line in blk.group(1).split('\n'):
            m = re.search(r"PRED-K4\s+(\S+)\s+pos0", line)
            if not m:
                continue
            part = m.group(1)
            v0 = re.search(r"vel0ρ=([+\-]?(?:\d+\.\d+|nan))", line)
            v1 = re.search(r"vel1ρ=([+\-]?(?:\d+\.\d+|nan))", line)
            out['K4_partition'][part] = {
                'vel0': float('nan') if (not v0 or 'nan' in v0.group(1)) else float(v0.group(1)),
                'vel1': float('nan') if (not v1 or 'nan' in v1.group(1)) else float(v1.group(1)),
            }

    m = re.search(r"--- latent fidelity vs horizon \(test, aggregate\) ---\n((?:\s*h=\s*\d+.*\n)+)", text)
    if m:
        for line in m.group(1).split('\n'):
            mh = re.match(r"\s*h=\s*(\d+)\s+n=\s*\d+\s+cos=([+\-]?\d+\.\d+)", line)
            if mh:
                out['cos_by_horizon'][int(mh.group(1))] = float(mh.group(2))

    m = re.search(r"--- latent fidelity \(pred vs real emb\)[^\n]*\n((?:[^\n]+cos=[^\n]+\n)+)", text)
    if m:
        for line in m.group(1).split('\n'):
            mp = re.match(r"\s+(\S+)\s+n=\s*\d+\s+cos=([+\-]?\d+\.\d+)", line)
            if mp:
                out['cos_by_partition'][mp.group(1)] = float(mp.group(2))
    return out

# -----------------------------------------------------------------------------
def find_log(dom: str, w: str, f: str) -> Path | None:
    """Search LOG_DIRS in order for the matching rollout log."""
    fn = f"rollout_{dom}_w{WTAG[w]}_f{f}.log"
    for d in LOG_DIRS:
        p = d / fn
        if p.exists():
            return p
    return None

def gather_all() -> dict:
    """Return results[dom][weight][frames] = parsed log dict (empty if log missing)."""
    results = {}
    for dom in DOMAINS:
        results[dom] = {}
        for w in WEIGHTS:
            results[dom][w] = {}
            for f in FRAMES:
                p = find_log(dom, w, f)
                results[dom][w][f] = parse_log(p) if p else {}
    return results

def fmt(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        if math.isnan(v):
            return 'nan'
        return f"{v:+.3f}"
    return str(v)

def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

# -----------------------------------------------------------------------------
def render_metric_table(results, dom, metric_key, metric_label):
    """Render a weight × frames grid table for one (domain, metric)."""
    lines = []
    lines.append(f"\n#### {DOMAIN_LABEL[dom]} — {metric_label}")
    lines.append(f"| weight \\ frames | f=1 | f=2 | f=4 |")
    lines.append(f"|---|---|---|---|")
    best = None
    for w in WEIGHTS:
        row = [f"**w={w}**"]
        for f in FRAMES:
            val = metric_key(results[dom][w][f])
            row.append(fmt(val))
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                if best is None or val > best[0]:
                    best = (val, w, f)
        lines.append("| " + " | ".join(row) + " |")
    if best:
        lines.append(f"\n→ best: **w={best[1]}, f={best[2]}** at ρ={best[0]:+.3f}")
    return "\n".join(lines)

def render_partition_table(results, dom, get_value, title):
    """Render a weight × frames grid for ALL partitions of ONE quantity (table per partition)."""
    lines = [f"\n### {DOMAIN_LABEL[dom]} — {title}"]
    parts_present = set()
    for w in WEIGHTS:
        for f in FRAMES:
            r = results[dom][w][f]
            parts_present |= set(r.get('K4_partition', {}).keys())
    parts = [p for p in PART_ORDER if p in parts_present]
    for part in parts:
        lines.append(f"\n**{part}**")
        lines.append(f"| weight \\ frames | f=1 | f=2 | f=4 |")
        lines.append(f"|---|---|---|---|")
        for w in WEIGHTS:
            row = [f"**w={w}**"]
            for f in FRAMES:
                v = get_value(results[dom][w][f], part)
                row.append(fmt(v))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='emit detailed per-partition tables (4 partitions × 4 quantities per domain)')
    args = ap.parse_args()

    results = gather_all()

    # Sanity: which logs are missing?
    missing = []
    for dom in DOMAINS:
        for w in WEIGHTS:
            for f in FRAMES:
                if not results[dom][w][f]:
                    missing.append(f"{dom}_w{w}_f{f}")

    total = len(DOMAINS) * len(WEIGHTS) * len(FRAMES)
    found = total - len(missing)
    print("# λ_probe × frames sweep — three-domain results")
    print(f"\n**Date**: 2026-06-05  **Configs**: {total} = {len(DOMAINS)} domains × {len(WEIGHTS)} weights × {len(FRAMES)} frames")
    print("**Log dirs**:")
    for d in LOG_DIRS:
        n = sum(1 for f in d.glob("rollout_*.log")) if d.exists() else 0
        print(f"- `{d}` ({n} rollout logs)")
    if missing:
        print(f"\n⚠️ **Missing {len(missing)}/{total} logs**: {', '.join(missing[:8])}{'...' if len(missing)>8 else ''}")
    else:
        print(f"\n✅ All {total} logs parsed")

    # -------- Summary headline tables (per-domain quick view) --------
    print("\n---\n\n## 1. Quick-view tables (K=4 vx · vy · long-cos)\n")

    for dom in DOMAINS:
        print(f"\n### {DOMAIN_LABEL[dom]}")
        # vx ID
        print(render_metric_table(results, dom,
              lambda r: get(r, 'K4_partition', 'ID', 'vel0'),
              "**vx (vel0) K=4, ID**"))
        # vx v-OOD
        print(render_metric_table(results, dom,
              lambda r: get(r, 'K4_partition', 'v-OOD', 'vel0'),
              "**vx (vel0) K=4, v-OOD**"))
        # vy ID (NaN for uniform)
        print(render_metric_table(results, dom,
              lambda r: get(r, 'K4_partition', 'ID', 'vel1'),
              "**vy (vel1) K=4, ID**"))
        # cos @ h=16
        print(render_metric_table(results, dom,
              lambda r: get(r, 'cos_by_horizon', 16),
              "**latent cos @ h=16**"))
        # cos by partition both-OOD
        print(render_metric_table(results, dom,
              lambda r: get(r, 'cos_by_partition', 'both-OOD'),
              "**cos by partition: both-OOD**"))

    # -------- Best-(w,f) summary across all metrics --------
    print("\n---\n\n## 2. Best (w, f) per (domain, metric)\n")
    metrics = [
        ('vx K=4 ID',    lambda r: get(r, 'K4_partition', 'ID', 'vel0')),
        ('vx K=4 v-OOD', lambda r: get(r, 'K4_partition', 'v-OOD', 'vel0')),
        ('vy K=4 ID',    lambda r: get(r, 'K4_partition', 'ID', 'vel1')),
        ('vy K=4 v-OOD', lambda r: get(r, 'K4_partition', 'v-OOD', 'vel1')),
        ('cos h=4',      lambda r: get(r, 'cos_by_horizon', 4)),
        ('cos h=16',     lambda r: get(r, 'cos_by_horizon', 16)),
        ('cos both-OOD', lambda r: get(r, 'cos_by_partition', 'both-OOD')),
    ]
    print("| domain | metric | best (w, f) | ρ |")
    print("|---|---|---|---|")
    for dom in DOMAINS:
        for mlabel, mfn in metrics:
            best = None
            for w in WEIGHTS:
                for f in FRAMES:
                    v = mfn(results[dom][w][f])
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    if best is None or v > best[0]:
                        best = (v, w, f)
            if best is None:
                print(f"| {DOMAIN_LABEL[dom]} | {mlabel} | — | — |")
            else:
                print(f"| {DOMAIN_LABEL[dom]} | {mlabel} | w={best[1]}, f={best[2]} | {best[0]:+.3f} |")

    # -------- Full per-partition detail (optional) --------
    if args.full:
        print("\n---\n\n## 3. Full per-partition tables (K=4)\n")
        for dom in DOMAINS:
            print(render_partition_table(results, dom,
                  lambda r, p: get(r, 'K4_partition', p, 'vel0'),
                  "K=4 vx (vel0) ρ — all partitions"))
            print(render_partition_table(results, dom,
                  lambda r, p: get(r, 'K4_partition', p, 'vel1'),
                  "K=4 vy (vel1) ρ — all partitions"))

if __name__ == '__main__':
    main()
