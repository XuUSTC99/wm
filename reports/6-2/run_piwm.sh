#!/bin/bash
# PIWM deep-supervision experiments on uniform_motion + collision, following the
# parabola protocol (reports/5-26/piwm_deepsup_results.md).
# 3 probe arms per domain (pos-only / pos+vel / mf4) vs existing baseline.
# GPU 0 = uniform_motion, GPU 3 = collision (run in parallel).
# Fully detached: survives the launching shell / Claude Code being closed.
set -u
ROOT=/home/qlib/am/wm
LEWM=$ROOT/le-wm
LOG=$ROOT/reports/6-2/logs
SWM=/home/qlib/.stable_worldmodel
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"
# Path masking: invoke the venv python via its RELATIVE path after `cd` (not the
# absolute path, not `activate` — which resolves to the PATH-shadowing qlib_env,
# not `exec -a` — which breaks python's prefix detection). The process cmdline
# then shows `.venv/bin/python -u train.py ...`, hiding /home/qlib on this shared host.

train_arm () {  # gpu domain datacfg name probe_args...
  local gpu=$1 datacfg=$3 name=$4; shift 4; local pargs="$@"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$LOG/orchestrator.log"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    .venv/bin/python -u train.py data=$datacfg output_model_name=$name subdir=$name \
      wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.enabled=true loss.probe.weight=1.0 $pargs \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$LOG/orchestrator.log"
}

eval_arm () {  # gpu domain ckpt tag outname
  local gpu=$1 dom=$2 ckpt=$3 tag=$4 out=$5
  echo "[eval  $(date +%H:%M:%S)] $out" >> "$LOG/orchestrator.log"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
      le-wm/.venv/bin/python phyworld/scripts/rollout_eval_id1k.py \
      --domain $dom --ckpt "$ckpt" --tag "$tag" --max-trajs 500 ) \
      > "$LOG/rollout_${out}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $out (exit $?)" >> "$LOG/orchestrator.log"
}

run_domain () {  # gpu domain datacfg velcol baseline_ckpt prefix
  local gpu=$1 dom=$2 datacfg=$3 vel=$4 base=$5 pre=$6
  # --- train 3 probe arms (sequential on this GPU) ---
  train_arm $gpu $dom $datacfg ${pre}_piwm_probe_id1k  loss.probe.target=proprio loss.probe.frames=1
  train_arm $gpu $dom $datacfg ${pre}_piwm_posvel_id1k "loss.probe.target=[proprio,$vel]" loss.probe.frames=1
  train_arm $gpu $dom $datacfg ${pre}_piwm_mf4_id1k    "loss.probe.target=[proprio,$vel]" loss.probe.frames=4
  # --- eval baseline + 3 arms ---
  eval_arm $gpu $dom "$base" baseline ${dom}_baseline
  eval_arm $gpu $dom "$SWM/${pre}_piwm_probe_id1k/lewm_${pre}_piwm_probe_id1k_epoch_20_object.ckpt"   pos-only ${dom}_posonly
  eval_arm $gpu $dom "$SWM/${pre}_piwm_posvel_id1k/lewm_${pre}_piwm_posvel_id1k_epoch_20_object.ckpt" pos+vel  ${dom}_posvel
  eval_arm $gpu $dom "$SWM/${pre}_piwm_mf4_id1k/lewm_${pre}_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4      ${dom}_mf4
  echo "[DOMAIN DONE $(date +%H:%M:%S)] $dom" >> "$LOG/orchestrator.log"
}

echo "=== START $(date) ===" >> "$LOG/orchestrator.log"

run_domain 0 uniform_motion phyworld_uniform_motion_id1k action \
  "$SWM/uniform_paperinit_id1k/lewm_uniform_paperinit_id1k_epoch_20_object.ckpt" uniform &
run_domain 3 collision phyworld_collision_id1k_st state \
  "$SWM/collision_paperinit_id1k/lewm_collision_paperinit_id1k_epoch_20_object.ckpt" collision &
wait

echo "=== ALL TRAIN+EVAL DONE $(date) ===" >> "$LOG/orchestrator.log"

# --- summarize logs -> markdown report ---
( cd "$ROOT" && le-wm/.venv/bin/python reports/6-2/summarize.py ) >> "$LOG/orchestrator.log" 2>&1
echo "=== SUMMARY WRITTEN $(date) ===" >> "$LOG/orchestrator.log"
