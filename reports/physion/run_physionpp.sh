#!/bin/bash
# Train lewm ON Physion++ real data (2026-07-06).
# Base: free-rollout (num_preds=8). Vary structured/consistency/pos_weight to test
# whether phyworld's positive methods make OOD + long-horizon good on REAL data
# (now possible because Physion++ gives real proprio = target object position).
#
# Usage: ./run_physionpp.sh <GPU> <NAME> <structured_w> <consistency_w> <pos_weight> [extra]
set -u
GPU=${1:-0}; NAME=${2:-pp}; SW=${3:-0.0}; CW=${4:-0.0}; PW=${5:-1.0}; EXTRA=${6:-}
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/physionpp; mkdir -p "$LOG"

echo "=== $NAME START $(date) GPU$GPU sw=$SW cw=$CW pw=$PW ==="
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u train.py data=physionpp \
  wandb.enabled=false tensorboard.enabled=false \
  wm.free_rollout=true wm.num_preds=8 \
  loss.structured.weight=$SW loss.consistency.weight=$CW loss.pos_weight=$PW $EXTRA \
  output_model_name=$NAME subdir=$NAME trainer.max_epochs=20 \
  +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
EC=$?
echo "=== $NAME DONE $(date) ec=$EC ==="
[ $EC -ne 0 ] && tail -15 "$LOG/train_${NAME}.log"
