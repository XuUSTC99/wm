#!/bin/bash
# Seed replication for the pretrain-vs-posttrain 2x2 (paper Table 2 / tab:scratch).
# Same as run_pretrain.sh but (a) takes a SEED and (b) pins the per-domain epoch
# count to whatever the original pp2_* run used, so the new seeds are comparable:
#   parabola -> 120 epochs, uniform_motion/collision -> 60   (verified in
#   raw_data/configs/pp2_*_scratch_*.yaml; the appendix's "60 uniformly" is wrong)
# LR is 2e-5 for every from-scratch cell, matching the pp2 reruns.
# Usage: SEED=1234 PHYS=on ./run_pretrain_seed.sh <GPU> <DOM>
#   DOM in {uniform_motion, parabola, collision}
set -u
GPU=$1; DOM=$2
SEED=${SEED:?need SEED}; PHYS=${PHYS:-off}; INIT=scratch

case "$DOM" in
  parabola)       EP=120; DATA=phyworld_parabola_id1k ;;
  uniform_motion) EP=60;  DATA=phyworld_uniform_motion_id1k ;;
  collision)      EP=60;  DATA=phyworld_collision_id1k ;;
  *) echo "bad DOM $DOM"; exit 2 ;;
esac
case "$DOM" in uniform_motion) SHORT=um ;; parabola) SHORT=par ;; collision) SHORT=col ;; esac
NAME=pp2_${SHORT}_scratch_${PHYS}_s${SEED}

ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/pretrain_physics; mkdir -p "$LOG"

# physics on -> structured slot (w=1.0, target=proprio) + const-accel dynamics head;
# off -> pure free rollout.  Verified against pp2_par_scratch_{on,off}.yaml.
if [ "$PHYS" = "on" ]; then
  PHYS_FLAG="loss.structured.weight=1.0 loss.structured.target=proprio dynamics.enabled=true dynamics.accel_form=const"
else
  PHYS_FLAG="loss.structured.weight=0.0 dynamics.enabled=false"
fi

echo "=== $NAME START $(date) GPU$GPU dom=$DOM ep=$EP seed=$SEED phys=$PHYS ==="
cd "$LEWM" || exit 2
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  .venv/bin/python -u train.py data=$DATA \
    output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
    seed=$SEED trainer.max_epochs=$EP optimizer.lr=2e-5 \
    loss.probe.weight=0.0 $PHYS_FLAG \
    > "$LOG/train_${NAME}.log" 2>&1
EC=$?
echo "=== train done ec=$EC $(date) ==="
[ $EC -ne 0 ] && { echo "TRAIN FAIL -> $LOG/train_${NAME}.log"; tail -20 "$LOG/train_${NAME}.log"; exit 1; }

CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "NO CKPT for $NAME"; exit 1; }
cd "$ROOT" || exit 2
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
    --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
echo "=== ALL DONE $NAME ec=$? $(date) ==="
