#!/bin/bash
# Cross-backbone generalization queue (2026-07-16, AAAI deadline sprint).
# Second JEPA-family model: DINO-WM / V-JEPA2-AC style = FROZEN DINOv2-small
# (384-d CLS) + trainable projector(384->192 adapter) + same ARPredictor stack.
# Tests that the paper's two core findings generalize beyond LeWM's ViT-tiny:
#   Finding 2 (FR >> TF): 3 domains x 2 modes x 3 seeds       = 18 runs
#   Finding 1 (physics injection fails): 3 mechanisms x 3 dom =  9 runs
# All 27 runs fit concurrently (6.3GB each, ~35min/run) -> static GPU packing.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm
QLOG=$LOG/queue.log
mkdir -p "$LOG"

UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
TF="wm.free_rollout=false wm.num_preds=1 loader.batch_size=128"
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
CONS="$SLOT loss.consistency.weight=1.0"

# NAME|DATA|DOM|EXTRA  (all: frozen dinov2, no init ckpt, 20ep, np8 fr defaults)
Q=(
  # --- headline FR vs TF, 3 seeds x 3 domains ---
  "dinowm_um_tf_s3072|$UM|uniform_motion|$TF seed=3072"
  "dinowm_um_tf_s1234|$UM|uniform_motion|$TF seed=1234"
  "dinowm_um_tf_s42|$UM|uniform_motion|$TF seed=42"
  "dinowm_um_fr_s3072|$UM|uniform_motion|seed=3072"
  "dinowm_um_fr_s1234|$UM|uniform_motion|seed=1234"
  "dinowm_um_fr_s42|$UM|uniform_motion|seed=42"
  "dinowm_par_tf_s3072|$PAR|parabola|$TF seed=3072"
  "dinowm_par_tf_s1234|$PAR|parabola|$TF seed=1234"
  "dinowm_par_tf_s42|$PAR|parabola|$TF seed=42"
  "dinowm_par_fr_s3072|$PAR|parabola|seed=3072"
  "dinowm_par_fr_s1234|$PAR|parabola|seed=1234"
  "dinowm_par_fr_s42|$PAR|parabola|seed=42"
  "dinowm_col_tf_s3072|$COL|collision|$TF seed=3072"
  "dinowm_col_tf_s1234|$COL|collision|$TF seed=1234"
  "dinowm_col_tf_s42|$COL|collision|$TF seed=42"
  "dinowm_col_fr_s3072|$COL|collision|seed=3072"
  "dinowm_col_fr_s1234|$COL|collision|seed=1234"
  "dinowm_col_fr_s42|$COL|collision|seed=42"
  # --- physics injection arms (fr np8, seed default 3072) ---
  "dinowm_um_structpos_pw30|$UM|uniform_motion|$SLOT loss.pos_weight=30"
  "dinowm_par_structpos_pw30|$PAR|parabola|$SLOT loss.pos_weight=30"
  "dinowm_col_structpos_pw30|$COL|collision|$SLOT loss.pos_weight=30"
  "dinowm_um_probeF2|$UM|uniform_motion|$PROBE"
  "dinowm_par_probeF2|$PAR|parabola|$PROBE"
  "dinowm_col_probeF2|$COL|collision|$PROBE"
  "dinowm_um_cons|$UM|uniform_motion|$CONS"
  "dinowm_par_cons|$PAR|parabola|$CONS"
  "dinowm_col_cons|$COL|collision|$CONS"
)

run_one() {  # GPU NAME DATA DOM EXTRA — train then eval, same GPU
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g ($EXTRA)" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=3 loss.probe.weight=0.0 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1
  local EC=$?
  echo "[$(date +%H:%M)] train done $NAME ec=$EC" >> "$QLOG"
  [ $EC -ne 0 ] && { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}

echo "=== dinowm queue START $(date), ${#Q[@]} jobs, static packing over 8 GPUs ===" > "$QLOG"
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  if grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null; then
    echo "SKIP $NAME (already done)" >> "$QLOG"; i=$((i+1)); continue
  fi
  g=$((i % 8))
  run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" &
  i=$((i+1))
  sleep 3
done
wait
echo "=== dinowm queue ALL FINISHED $(date) ===" >> "$QLOG"
