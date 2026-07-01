#!/bin/bash
set -u
ROOT=/home/likun-share/junjxu/wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/6-2_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
PY=le-wm/.venv/bin/python
mkdir -p "$LOG"
ev () { ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$1 \
   STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
   $PY phyworld/scripts/rollout_eval_id1k.py \
   --domain parabola --ckpt "$2" --tag "$3" --max-trajs 500 ) > "$LOG/rollout_parabola_$3.log" 2>&1
   echo "[$(date +%H:%M:%S)] parabola_$3 done (exit $?)" >> "$LOG/parabola_evals.log"; }
echo "=== PARABOLA EVALS START $(date) ===" > "$LOG/parabola_evals.log"
( ev 0 "$SWM/parabola_paperinit_id1k/lewm_parabola_paperinit_id1k_epoch_20_object.ckpt"   baseline
  ev 0 "$SWM/parabola_piwm_posvel_id1k/lewm_parabola_piwm_posvel_id1k_epoch_20_object.ckpt" posvel ) &
( ev 1 "$SWM/parabola_piwm_probe_id1k/lewm_parabola_piwm_probe_id1k_epoch_20_object.ckpt"  posonly
  ev 1 "$SWM/parabola_piwm_mf4_id1k/lewm_parabola_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4 ) &
wait
echo "=== PARABOLA EVALS DONE $(date) ===" >> "$LOG/parabola_evals.log"
