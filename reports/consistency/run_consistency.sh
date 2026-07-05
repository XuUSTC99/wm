#!/bin/bash
# Dynamics-consistency loss experiment on uniform_motion (2026-07-05).
#
# Hypothesis: prior physics mechanisms (structured slot / kinematics head) only
# constrain "what the state IS", not "how it EVOLVES", so they don't fix
# long-horizon drift. The NEW consistency loss constrains the PREDICTED slot's
# velocity (finite-diff displacement) to match true proprio velocity over the
# rollout horizon -> a SOFT physics-equation constraint on evolution.
#
# Base = verified best: free_rollout + structured(1.0) + pos_weight(30), from
# pusht init (matches run_structdyn.sh). Only consistency.{weight,accel_weight} vary.
#
# Usage: ./run_consistency.sh <GPU> <NAME> <cons_weight> <accel_weight>
set -u
GPU=${1:-2}
NAME=${2:-uniform_cons}
CONS=${3:-1.0}
ACCEL=${4:-0.0}
PW=${5:-30}         # pos_weight (承重强度)
DATA=${6:-phyworld_uniform_motion_id1k}
DOM=${7:-uniform_motion}

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/consistency_eval
mkdir -p "$LOG"

echo "=== $NAME START $(date) GPU$GPU cons=$CONS accel=$ACCEL ==="
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  .venv/bin/python -u train.py data=$DATA \
    output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=False \
    trainer.max_epochs=20 \
    wm.free_rollout=true wm.num_preds=8 \
    loss.probe.weight=0.0 loss.structured.weight=1.0 loss.pos_weight=$PW \
    loss.consistency.weight=$CONS loss.consistency.accel_weight=$ACCEL \
    dynamics.enabled=false \
    +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
TRAIN_EC=$?
echo "=== train done ec=$TRAIN_EC $(date) ==="
[ $TRAIN_EC -ne 0 ] && { echo "TRAIN FAILED -> $LOG/train_${NAME}.log"; tail -25 "$LOG/train_${NAME}.log"; exit 1; }

CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
echo "=== eval ckpt=$CKPT ==="
[ -z "$CKPT" ] && { echo "NO CKPT"; exit 1; }
cd "$ROOT" || exit 2
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
    --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
echo "=== ALL DONE $(date) ec=$? -> $LOG/rollout_${NAME}.log ==="
