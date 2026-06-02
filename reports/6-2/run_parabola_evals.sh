#!/bin/bash
set -u
ROOT=/home/qlib/am/wm; LOG=$ROOT/reports/6-2/logs; SWM=/home/qlib/.stable_worldmodel
PY=le-wm/.venv/bin/python
ev () { ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$1 $PY phyworld/scripts/rollout_eval_id1k.py \
   --domain parabola --ckpt "$2" --tag "$3" --max-trajs 500 ) > "$LOG/rollout_parabola_$3.log" 2>&1
   echo "[$(date +%H:%M:%S)] parabola_$3 done (exit $?)" >> "$LOG/parabola_evals.log"; }
echo "=== PARABOLA EVALS START $(date) ===" > "$LOG/parabola_evals.log"
( ev 0 "$SWM/parabola_paperinit_id1k/lewm_parabola_paperinit_id1k_epoch_20_object.ckpt"   baseline
  ev 0 "$SWM/parabola_piwm_posvel_id1k/lewm_parabola_piwm_posvel_id1k_epoch_20_object.ckpt" posvel ) &
( ev 1 "$SWM/parabola_piwm_probe_id1k/lewm_parabola_piwm_probe_id1k_epoch_20_object.ckpt"  posonly
  ev 1 "$SWM/parabola_piwm_mf4_id1k/lewm_parabola_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4 ) &
wait
echo "=== PARABOLA EVALS DONE $(date) ===" >> "$LOG/parabola_evals.log"
