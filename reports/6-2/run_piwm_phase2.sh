#!/bin/bash
# Phase-2 takeover: once arm-1 (pos-only) finishes for BOTH domains, kill the old
# 2-GPU orchestrator and run the remaining 4 arms across the two FREE FAST GPUs
# (0 + 2), abandoning the shared/slow GPU3. Preserves arm-1 progress; no kill of
# in-flight arm-1. Fully detached.
set -u
ROOT=/home/qlib/am/wm
LEWM=$ROOT/le-wm
LOG=$ROOT/reports/6-2/logs
SWM=/home/qlib/.stable_worldmodel
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"
P2=$LOG/phase2.log
# NOTE: 6-2 arms use output_model_name WITHOUT a lewm_ prefix, so ckpt files are
# "<name>_epoch_20_object.ckpt" (not "lewm_<name>_..."). Baselines DO have lewm_.
ckpt20 () { ls "$SWM/$1/${1}_epoch_20_object.ckpt" 2>/dev/null; }

echo "=== PHASE2 START $(date) — waiting for arm-1 (pos-only) to finish ===" >> "$P2"
# 1) wait until BOTH arm-1 ckpts (epoch_20) exist
until [ -n "$(ckpt20 uniform_piwm_probe_id1k)" ] && [ -n "$(ckpt20 collision_piwm_probe_id1k)" ]; do
  sleep 20
done
echo "[$(date +%H:%M:%S)] arm-1 done for both domains; taking over" >> "$P2"

# 2) kill the old orchestrator + any in-flight arm-2 it just started (preserve arm-1 ckpts on disk).
# pkill uses ERE: alternation is `|` (NOT `\|`, which would match a literal pipe and kill nothing).
# `run_piwm\.sh$` anchors so we don't accidentally match run_piwm_phase2.sh.
pkill -9 -f "run_piwm\.sh" 2>/dev/null
pkill -9 -f "train\.py.*_piwm_(posvel|mf4)_" 2>/dev/null
# clear any partial arm-2/3 dirs the old orchestrator may have begun
rm -rf "$SWM"/{uniform,collision}_piwm_posvel_id1k "$SWM"/{uniform,collision}_piwm_mf4_id1k 2>/dev/null
sleep 5
echo "[$(date +%H:%M:%S)] old orchestrator killed; launching 4 arms on GPU0+GPU2" >> "$P2"

train_arm () {  # gpu datacfg name probe_args...
  local gpu=$1 datacfg=$2 name=$3; shift 3; local pargs="$@"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$P2"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    .venv/bin/python -u train.py data=$datacfg output_model_name=$name subdir=$name \
      wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.enabled=true loss.probe.weight=1.0 $pargs \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$P2"
}
eval_arm () {  # gpu domain ckpt tag out
  local gpu=$1 dom=$2 ckpt=$3 tag=$4 out=$5
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python phyworld/scripts/rollout_eval_id1k.py \
      --domain $dom --ckpt "$ckpt" --tag "$tag" --max-trajs 500 ) > "$LOG/rollout_${out}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $out" >> "$P2"
}

# 3) remaining 4 arms: GPU0 = uniform {posvel,mf4}, GPU2 = collision {posvel,mf4}
( train_arm 0 phyworld_uniform_motion_id1k uniform_piwm_posvel_id1k "loss.probe.target=[proprio,action]" loss.probe.frames=1
  train_arm 0 phyworld_uniform_motion_id1k uniform_piwm_mf4_id1k    "loss.probe.target=[proprio,action]" loss.probe.frames=4 ) &
( train_arm 2 phyworld_collision_id1k_st  collision_piwm_posvel_id1k "loss.probe.target=[proprio,state]" loss.probe.frames=1
  train_arm 2 phyworld_collision_id1k_st  collision_piwm_mf4_id1k    "loss.probe.target=[proprio,state]" loss.probe.frames=4 ) &
wait
echo "[$(date +%H:%M:%S)] all 4 arms trained" >> "$P2"

# 4) eval baseline + 3 arms per domain (GPU0 uniform, GPU2 collision)
( eval_arm 0 uniform_motion "$SWM/uniform_paperinit_id1k/lewm_uniform_paperinit_id1k_epoch_20_object.ckpt" baseline uniform_motion_baseline
  eval_arm 0 uniform_motion "$SWM/uniform_piwm_probe_id1k/uniform_piwm_probe_id1k_epoch_20_object.ckpt"  pos-only uniform_motion_posonly
  eval_arm 0 uniform_motion "$SWM/uniform_piwm_posvel_id1k/uniform_piwm_posvel_id1k_epoch_20_object.ckpt" pos+vel uniform_motion_posvel
  eval_arm 0 uniform_motion "$SWM/uniform_piwm_mf4_id1k/uniform_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4     uniform_motion_mf4 ) &
( eval_arm 2 collision "$SWM/collision_paperinit_id1k/lewm_collision_paperinit_id1k_epoch_20_object.ckpt" baseline collision_baseline
  eval_arm 2 collision "$SWM/collision_piwm_probe_id1k/collision_piwm_probe_id1k_epoch_20_object.ckpt"  pos-only collision_posonly
  eval_arm 2 collision "$SWM/collision_piwm_posvel_id1k/collision_piwm_posvel_id1k_epoch_20_object.ckpt" pos+vel collision_posvel
  eval_arm 2 collision "$SWM/collision_piwm_mf4_id1k/collision_piwm_mf4_id1k_epoch_20_object.ckpt"       mf4     collision_mf4 ) &
wait
echo "[$(date +%H:%M:%S)] all evals done" >> "$P2"

( cd "$ROOT" && le-wm/.venv/bin/python reports/6-2/summarize.py ) >> "$P2" 2>&1
echo "=== PHASE2 DONE $(date) ===" >> "$P2"
