#!/bin/bash
# structpos + 2nd-order dynamics on uniform_motion, train -> auto latent eval.
# Goal: introduce a kinematics equation on the position slot and check long-horizon
# rollout drift (h=16/28) + OOD vs the structpos baseline.
#
# Parametrized so ablations (const-velocity, +action, collision) reuse it.
# Detached run (setsid). Single GPU (multi-GPU DDP errors on the unused probe_head).
#
# Usage: ./run_structdyn.sh <GPU> <NAME> "<extra hydra overrides>"
#   e.g. ./run_structdyn.sh 2 uniform_motion_structcv_id1k "dynamics.learnable_accel=false"
# Defaults reproduce the first learnable-accel run on GPU 1.
set -u
GPU=${1:-1}
NAME=${2:-uniform_motion_structdyn_id1k}
EXTRA=${3:-}

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/structdyn_eval
mkdir -p "$LOG"

echo "=== $NAME START $(date) GPU$GPU extra='$EXTRA' ==="

# ---- train ----
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  .venv/bin/python -u train.py data=${DATA:-phyworld_uniform_motion_id1k} \
    output_model_name=$NAME subdir=$NAME wandb.enabled=False trainer.max_epochs=20 \
    loss.probe.weight=0.0 loss.structured.weight=${SW:-1.0} \
    loss.structured.target=proprio loss.structured.start_dim=0 \
    dynamics.enabled=${DYN:-true} $EXTRA \
    +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
TRAIN_EC=$?
echo "=== train done ec=$TRAIN_EC $(date) ==="
[ $TRAIN_EC -ne 0 ] && { echo "TRAIN FAILED -> $LOG/train_${NAME}.log"; exit 1; }

# ---- eval: latent rollout (uses dynamics head via torch.load object dump) ----
CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
echo "=== eval ckpt=$CKPT ==="
[ -z "$CKPT" ] && { echo "NO CKPT FOUND"; exit 1; }
cd "$ROOT" || exit 2
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
    --domain ${DOM:-uniform_motion} --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
echo "=== eval done ec=$? $(date) ==="
echo "=== ALL DONE $(date) -> logs in $LOG ==="
