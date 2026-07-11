#!/bin/bash
# Pretrain-vs-posttrain physics injection. Trains LeWM either FROM SCRATCH (no init)
# or from pusht ckpt (post-training), with physics (structured+const dynamics) on/off,
# then latent-rollout evals. See EXPERIMENT_PLAN.md.
# Usage: INIT=scratch|pusht PHYS=on|off ./run_pretrain.sh <GPU> <NAME> <DATA> <DOM> [EPOCHS]
set -u
GPU=$1; NAME=$2; DATA=$3; DOM=$4; EP=${5:-60}
INIT=${INIT:-scratch}; PHYS=${PHYS:-off}

ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/pretrain_physics; mkdir -p "$LOG"

# init flag: pusht -> load ckpt (post-training); scratch -> nothing (from scratch)
INIT_FLAG=""
[ "$INIT" = "pusht" ] && INIT_FLAG="+init_from_ckpt=$STABLEWM_HOME/lewm_paper_pusht/weights.pt"
# physics: on -> structured slot + const(PIWM) dynamics; off -> pure free-rollout
if [ "$PHYS" = "on" ]; then
  PHYS_FLAG="loss.structured.weight=1.0 loss.structured.target=proprio dynamics.enabled=true dynamics.accel_form=const"
else
  PHYS_FLAG="loss.structured.weight=0.0 dynamics.enabled=false"
fi

echo "=== $NAME START $(date) GPU$GPU init=$INIT phys=$PHYS ep=$EP ==="
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  .venv/bin/python -u train.py data=$DATA \
    output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
    trainer.max_epochs=$EP optimizer.lr=${LR:-5e-5} \
    loss.probe.weight=0.0 $PHYS_FLAG $INIT_FLAG \
    > "$LOG/train_${NAME}.log" 2>&1
EC=$?
echo "=== train done ec=$EC $(date) ==="
[ $EC -ne 0 ] && { echo "TRAIN FAIL -> $LOG/train_${NAME}.log"; tail -15 "$LOG/train_${NAME}.log"; exit 1; }

CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "NO CKPT"; exit 1; }
cd "$ROOT" || exit 2
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
    --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
echo "=== ALL DONE ec=$? $(date) ==="
