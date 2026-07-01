#!/bin/bash
# Eval-only re-run (training already done; phase2 evals failed on a wrong python
# path). Correct path = le-wm/.venv/bin/python from ROOT. GPU0=uniform, GPU2=collision.
set -u
ROOT=/home/likun-share/junjxu/wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/6-2_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
PY=le-wm/.venv/bin/python
mkdir -p "$LOG"

ev () {  # gpu domain ckpt tag out
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$1 \
      STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
      $PY phyworld/scripts/rollout_eval_id1k.py \
      --domain $2 --ckpt "$3" --tag "$4" --max-trajs 500 ) > "$LOG/rollout_$5.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $5 (exit $?)" >> "$LOG/evals.log"
}

echo "=== EVALS START $(date) ===" > "$LOG/evals.log"
( ev 0 uniform_motion "$SWM/uniform_paperinit_id1k/lewm_uniform_paperinit_id1k_epoch_20_object.ckpt" baseline uniform_motion_baseline
  ev 0 uniform_motion "$SWM/uniform_piwm_probe_id1k/uniform_piwm_probe_id1k_epoch_20_object.ckpt"   pos-only uniform_motion_posonly
  ev 0 uniform_motion "$SWM/uniform_piwm_posvel_id1k/uniform_piwm_posvel_id1k_epoch_20_object.ckpt" pos+vel  uniform_motion_posvel
  ev 0 uniform_motion "$SWM/uniform_piwm_mf4_id1k/uniform_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4      uniform_motion_mf4 ) &
( ev 2 collision "$SWM/collision_paperinit_id1k/lewm_collision_paperinit_id1k_epoch_20_object.ckpt" baseline collision_baseline
  ev 2 collision "$SWM/collision_piwm_probe_id1k/collision_piwm_probe_id1k_epoch_20_object.ckpt"   pos-only collision_posonly
  ev 2 collision "$SWM/collision_piwm_posvel_id1k/collision_piwm_posvel_id1k_epoch_20_object.ckpt" pos+vel  collision_posvel
  ev 2 collision "$SWM/collision_piwm_mf4_id1k/collision_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4      collision_mf4 ) &
wait
echo "=== EVALS DONE $(date) ===" >> "$LOG/evals.log"
( cd "$ROOT" && STABLEWM_HOME=$STABLEWM_HOME $PY reports/6-2/summarize.py ) >> "$LOG/evals.log" 2>&1
echo "=== SUMMARY WRITTEN $(date) ===" >> "$LOG/evals.log"
