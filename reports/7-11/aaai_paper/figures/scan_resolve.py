"""Resolve every cell of the 30-cell injection scan to its per-seed rollout logs.

The seed-3072 runs, the earlier seed top-ups, and the 2026-07-22 batch all use
different naming conventions and live in three different trees, so the mapping
is written out explicitly rather than globbed. Import `resolve()` from the
figure script; run this file directly to see coverage.

Partition per domain matches what the figure reads: uniform/collision take
both-OOD, parabola takes r/m-OOD (its both-OOD nMSE has a degenerate
denominator -- h28 shows nMSE=109424 alongside cos=0.9301).

Audit note (2026-07-22): [dyn] strict a=g and [free] grounded were run under
two names but are the SAME config (their config.yaml files differ only in
output_model_name/subdir), so both rows resolve to the same pool of runs --
which is why the pool can hold four draws rather than three.
"""
import re
import pathlib

RUNS = pathlib.Path("/data1/likun-share/junjxu/runs")
SEARCH_DIRS = [
    RUNS / "scan_seeds",        # 2026-07-22 seed top-up
    RUNS / "aaai_p0",
    RUNS / "structdyn_eval",
    RUNS / "consistency_eval",
    pathlib.Path("/home/likun-share/junjxu/wm/raw_data/runs"),
]

ARMS = ["[slot] structpos", "[slot] +reweight (w=30)", "[slot] +velocity",
        "[probe] probe", "[probe] +slot", "[dyn] free MLP", "[dyn] strict a=g",
        "[cons] consistency", "[free] label-free", "[free] grounded"]
DOMS = ["uniform", "parabola", "collision"]
PARTITION = {"uniform": "both-OOD", "parabola": "r/m-OOD", "collision": "both-OOD"}

# cell -> {seed: run-name}. Seed 3072 is the original draw; 1234/42 are top-ups.
# "ag" runs serve rows 6 and 9 alike (see audit note above).
_AG = {"uniform":   {3072: ["uniform_piwm_const_fr_id1k", "uniform_grounded_const_id1k"],
                     1234: ["sc_um_ag_s1234"], 42: ["sc_um_ag_s42"]},
       "parabola":  {3072: ["parabola_piwm_const_fr_id1k", "parabola_grounded_const_id1k"],
                     1234: ["sc_par_ag_s1234"], 42: ["sc_par_ag_s42"]},
       "collision": {3072: ["collision_piwm_const_fr_id1k", "collision_grounded_const_id1k"],
                     1234: ["sc_col_ag_s1234"], 42: ["sc_col_ag_s42"]}}

CELLS = {
    ("[slot] structpos", "uniform"): {
        3072: ["uniform_motion_structpos_fr_id1k"],
        1234: ["uniform_motion_structpos_fr_pw1_s1234_id1k"],   # pw=1 IS plain structpos
        42:   ["uniform_motion_structpos_fr_pw1_s42_id1k"]},
    ("[slot] structpos", "parabola"): {
        3072: ["parabola_structpos_fr_id1k"],
        1234: ["sc_par_structpos_s1234"], 42: ["sc_par_structpos_s42"]},
    ("[slot] structpos", "collision"): {
        3072: ["collision_structpos_fr_id1k"],
        1234: ["sc_col_structpos_s1234"], 42: ["sc_col_structpos_s42"]},

    ("[slot] +reweight (w=30)", "uniform"): {
        3072: ["uniform_motion_structpos_fr_pw30_id1k"],
        1234: ["uniform_structpos_fr_pw30_s1234"], 42: ["uniform_structpos_fr_pw30_s42"]},
    ("[slot] +reweight (w=30)", "parabola"): {
        3072: ["parabola_structpos_fr_pw30_id1k"],
        1234: ["sc_par_reweight_s1234"], 42: ["sc_par_reweight_s42"]},
    ("[slot] +reweight (w=30)", "collision"): {
        3072: ["collision_structpos_fr_pw30_id1k"],
        1234: ["sc_col_reweight_s1234"], 42: ["sc_col_reweight_s42"]},

    ("[slot] +velocity", "uniform"): {
        3072: ["uniform_structposvel_pw30_fr"],
        1234: ["sc_um_velocity_s1234"], 42: ["sc_um_velocity_s42"]},
    ("[slot] +velocity", "parabola"): {
        3072: ["parabola_structposvel_pw30_fr"],
        1234: ["parabola_structposvel_pw30_s1234"], 42: ["parabola_structposvel_pw30_s42"]},
    ("[slot] +velocity", "collision"): {
        3072: ["collision_structposvel_pw30_fr"],
        1234: ["sc_col_velocity_s1234"], 42: ["sc_col_velocity_s42"]},

    ("[probe] probe", "uniform"): {
        3072: ["uniform_probeF2_fr"],
        1234: ["sc_um_probe_s1234"], 42: ["sc_um_probe_s42"]},
    ("[probe] probe", "parabola"): {
        3072: ["parabola_probeF2_fr"],
        1234: ["parabola_probeF2_fr_s1234"], 42: ["parabola_probeF2_fr_s42"]},
    ("[probe] probe", "collision"): {
        3072: ["collision_probeF2_fr"],
        1234: ["sc_col_probe_s1234"], 42: ["sc_col_probe_s42"]},

    ("[probe] +slot", "uniform"): {
        3072: ["uniform_probeF2_structpos_pw30_fr"],
        1234: ["uniform_probeF2_structpos_pw30_fr_s1234"],
        42:   ["uniform_probeF2_structpos_pw30_fr_s42"]},
    ("[probe] +slot", "parabola"): {
        3072: ["parabola_probeF2_structpos_pw30_fr"],
        1234: ["sc_par_probeslot_s1234"], 42: ["sc_par_probeslot_s42"]},
    ("[probe] +slot", "collision"): {
        3072: ["collision_probeF2_structpos_pw30_fr"],
        1234: ["sc_col_probeslot_s1234"], 42: ["sc_col_probeslot_s42"]},

    ("[dyn] free MLP", "uniform"): {
        3072: ["uniform_dyn_mlp_fr_id1k"],
        1234: ["sc_um_dynmlp_s1234"], 42: ["sc_um_dynmlp_s42"]},
    ("[dyn] free MLP", "parabola"): {
        3072: ["parabola_dyn_mlp_fr_id1k"],
        1234: ["sc_par_dynmlp_s1234"], 42: ["sc_par_dynmlp_s42"]},
    ("[dyn] free MLP", "collision"): {
        3072: ["collision_structdyn_fr_id1k"],
        1234: ["sc_col_dynmlp_s1234"], 42: ["sc_col_dynmlp_s42"]},

    # [cons] standardised 2026-07-22 on the plain config (cons=1.0, accel=0,
    # pw=1, bs=64). The old uniform (pw30) and parabola (accel 1.0, bs 32) runs
    # are deliberately NOT reused -- they were different variants.
    ("[cons] consistency", "uniform"): {
        3072: ["sc_um_cons_s3072"], 1234: ["sc_um_cons_s1234"], 42: ["sc_um_cons_s42"]},
    ("[cons] consistency", "parabola"): {
        3072: ["sc_par_cons_s3072"], 1234: ["sc_par_cons_s1234"], 42: ["sc_par_cons_s42"]},
    ("[cons] consistency", "collision"): {
        3072: ["collision_structpos_cons1p0_id1k"],
        1234: ["sc_col_cons_s1234"], 42: ["sc_col_cons_s42"]},

    ("[free] label-free", "uniform"): {
        3072: ["uniform_labelfree_const_id1k"],
        1234: ["sc_um_labelfree_s1234"], 42: ["sc_um_labelfree_s42"]},
    ("[free] label-free", "parabola"): {
        3072: ["parabola_labelfree_const_id1k"],
        1234: ["sc_par_labelfree_s1234"], 42: ["sc_par_labelfree_s42"]},
    ("[free] label-free", "collision"): {
        3072: ["collision_labelfree_const_id1k"],
        1234: ["sc_col_labelfree_s1234"], 42: ["sc_col_labelfree_s42"]},
}
for d in DOMS:
    CELLS[("[dyn] strict a=g", d)] = _AG[d]
    CELLS[("[free] grounded", d)] = _AG[d]

BASELINE = {
    "uniform":   {3072: ["uniform_motion_baseline_fr_id1k"],
                  1234: ["uniform_baseline_fr_s1234"], 42: ["uniform_baseline_fr_s42"]},
    "parabola":  {3072: ["parabola_baseline_fr_id1k"],
                  1234: ["parabola_baseline_fr_s1234"], 42: ["parabola_baseline_fr_s42"]},
    "collision": {3072: ["collision_baseline_fr_id1k"],
                  1234: ["collision_baseline_fr_s1234"], 42: ["collision_baseline_fr_s42"]},
}


def find_log(name):
    for d in SEARCH_DIRS:
        p = d / f"rollout_{name}.log"
        if p.exists():
            return p
    return None


def read_nmse(name, domain):
    """nMSE on the partition the figure reads, or None if the run has not landed."""
    p = find_log(name)
    if p is None:
        return None
    pat = re.compile(r"^\s+" + re.escape(PARTITION[domain]) + r"\s+n=\s*\d+\s+cos=[+-][\d.]+\s+nMSE=([\d.]+)")
    for line in p.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            return float(m.group(1))
    return None


def resolve(spec, domain):
    """spec: {seed: [run names]} -> list of nMSE values that have landed."""
    vals = []
    for seed in sorted(spec):
        for name in spec[seed]:
            v = read_nmse(name, domain)
            if v is not None:
                vals.append(v)
    return vals


if __name__ == "__main__":
    print(f"{'cell':<40} {'n':>3}  values")
    print("-" * 78)
    incomplete = 0
    for dom in DOMS:
        vals = resolve(BASELINE[dom], dom)
        mark = "" if len(vals) >= 3 else "   <-- INCOMPLETE"
        print(f"{'BASELINE ' + dom:<40} {len(vals):>3}  "
              f"{' '.join(f'{v:.3f}' for v in vals)}{mark}")
    print("-" * 78)
    for arm in ARMS:
        for dom in DOMS:
            vals = resolve(CELLS[(arm, dom)], dom)
            if len(vals) < 3:
                incomplete += 1
                mark = "   <-- INCOMPLETE"
            else:
                mark = ""
            print(f"{arm + ' / ' + dom:<40} {len(vals):>3}  "
                  f"{' '.join(f'{v:.3f}' for v in vals)}{mark}")
    print("-" * 78)
    print(f"{30 - incomplete}/30 cells at n>=3")
