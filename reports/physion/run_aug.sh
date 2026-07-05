#!/bin/bash
# Appearance-aug transfer experiment (2026-07-05).
# Goal: make phyworld methods transfer to real-looking Physion by killing the
# encoder's dependence on synthetic appearance. Train collision + free-rollout +
# per-trial brightness/contrast jitter(strength), then eval BOTH:
#   (a) Physion transfer (eval_physion_suite, OCP) — did we cross random 0.607?
#   (b) phyworld collision OOD (rollout_eval, r/m-OOD) — did aug help the softspot?
#
# Usage: ./run_aug.sh <GPU> <NAME> <aug_strength>
set -u
GPU=${1:-3}; NAME=${2:-col_aug}; AUG=${3:-0.4}
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/aug_eval; mkdir -p "$LOG"

echo "=== $NAME START $(date) GPU$GPU aug=$AUG ==="
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 .venv/bin/python -u train.py \
  data=phyworld_collision_id1k_st wandb.enabled=false tensorboard.enabled=false \
  wm.free_rollout=true wm.num_preds=8 aug.appearance=$AUG \
  output_model_name=$NAME subdir=$NAME trainer.max_epochs=20 \
  +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
[ $? -ne 0 ] && { echo "$NAME TRAIN FAIL"; tail -15 "$LOG/train_${NAME}.log"; exit 1; }
CK=$(ls -t $STABLEWM_HOME/$NAME/*_object.ckpt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "$NAME NO CKPT"; exit 1; }

cd "$ROOT"
echo "=== $NAME Physion eval ==="
CUDA_VISIBLE_DEVICES=$GPU "$LEWM/.venv/bin/python" phyworld/scripts/physion/eval_physion_suite.py \
  --ckpt "$CK" --tag "$NAME" --device cuda:0 2>&1 | grep -E 'mean AUC|readout|n=800'
echo "=== $NAME phyworld collision OOD ==="
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
  --domain collision --ckpt "$CK" --tag "$NAME" --max-trajs 400 > "$LOG/rollout_${NAME}.log" 2>&1
sed -n '/by partition/,/vs horizon/p' "$LOG/rollout_${NAME}.log" | grep -E 'ID |OOD '
echo "=== $NAME DONE $(date) ==="
