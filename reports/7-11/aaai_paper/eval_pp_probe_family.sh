#!/bin/bash
# Wait for the six probe-family runs to finish training, then evaluate each and
# write a comparison against the arms already in tab:pp.
#
# Kept separate from training (as run_pp_seeds.sh does) so evaluation does not
# compete with training for the cards. Invocation copied from the sibling queue:
# rollout_eval_physionpp.py --ckpt ... --device cuda:0 --tag ..., which emits the
# per-scenario nMSE block tab:pp is built from.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/physionpp
QLOG=$LOG/probe_eval.log
NAMES=(pp_probe_s3072 pp_probe_s1234 pp_probe_s42 pp_probeslot_s3072 pp_probeslot_s1234 pp_probeslot_s42)

echo "=== probe-family eval watcher START $(date) ===" > "$QLOG"
DEADLINE=$(( $(date +%s) + 12*3600 ))
for n in "${NAMES[@]}"; do
  # wait for this run's final checkpoint
  while [ ! -f "$S/$n/${n}_epoch_20_object.ckpt" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 120; done
  if [ ! -f "$S/$n/${n}_epoch_20_object.ckpt" ]; then
    echo "[$(date +%H:%M)] TIMEOUT waiting for $n" >> "$QLOG"; continue
  fi
  echo "[$(date +%H:%M)] EVAL $n" >> "$QLOG"
  cd "$ROOT" || exit 2
  CUDA_VISIBLE_DEVICES=0 STABLEWM_HOME=$S "$LEWM/.venv/bin/python" \
    phyworld/scripts/physion/rollout_eval_physionpp.py \
      --ckpt "$S/$n/${n}_epoch_20_object.ckpt" --device cuda:0 --tag "$n" \
      > "$LOG/rollout_${n}.log" 2>&1
  echo "[$(date +%H:%M)] EVAL done $n ec=$?" >> "$QLOG"
done

# --- summarise against the arms already in tab:pp -------------------------
"$LEWM/.venv/bin/python" - <<'PY' >> "$QLOG" 2>&1
import re, os, statistics as st
LOG="/data1/likun-share/junjxu/runs/physionpp"
SCEN=["friction_platform","mass_collision","mass_dominoes","mass_waterpush"]
def per_scene(tag):
    f=f"{LOG}/rollout_{tag}.log"
    if not os.path.exists(f): return {}
    out={}
    for l in open(f,errors="ignore"):
        m=re.match(r"\s+([a-z]+_[a-z]+)\s+n=\s*\d+\s+cos=[+-][\d.]+\s+nMSE=([\d.]+)", l)
        if m: out[m.group(1)]=float(m.group(2))
    return out
rows=[("FR (existing)","eval_pp_fr_e20"),("+slot (existing)","eval_pp_struct_e20"),
      ("+cons (existing)","eval_pp_cons_e20")]
print("\n=== per-scenario rollout nMSE (lower is better) ===")
hdr=f"{'arm':22}" + "".join(f"{s[:14]:>16}" for s in SCEN); print(hdr)
for lbl,tag in rows:
    d=per_scene(tag.replace("eval_","")) or per_scene(tag)
    print(f"{lbl:22}" + "".join(f"{d.get(s,float('nan')):>16.3f}" for s in SCEN))
for arm in ["pp_probe","pp_probeslot"]:
    vals={s:[] for s in SCEN}
    for sd in (3072,1234,42):
        d=per_scene(f"{arm}_s{sd}")
        for s in SCEN:
            if s in d: vals[s].append(d[s])
    lbl = ("probe (NEW)" if arm=="pp_probe" else "probe+slot (NEW)")
    line=""
    for s in SCEN:
        v=vals[s]
        line += f"{st.mean(v):>10.3f}±{st.stdev(v):.3f}" if len(v)>1 else (f"{v[0]:>16.3f}" if v else f"{'—':>16}")
    print(f"{lbl:22}{line}")
print("\n(3-seed mean±std for the new arms; existing arms are the single-seed values in tab:pp)")
PY
echo "=== probe-family eval ALL DONE $(date) ===" >> "$QLOG"
