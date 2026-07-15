#!/bin/bash
# dinowm queue 2: widen injection scan to match LeWM's arm families (2026-07-16).
# +9 arms: posvel_pw30 (the ONE arm with LeWM's parabola exception — cross-model test!),
# plain slot (no rescue weighting), grounded_const (slot + kinematic dynamics head).
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm
QLOG=$LOG/queue2.log
mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
Q=(
  "dinowm_um_posvel_pw30|$UM|uniform_motion|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  "dinowm_par_posvel_pw30|$PAR|parabola|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  "dinowm_col_posvel_pw30|$COL|collision|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  "dinowm_um_structpos_plain|$UM|uniform_motion|$SLOT"
  "dinowm_par_structpos_plain|$PAR|parabola|$SLOT"
  "dinowm_col_structpos_plain|$COL|collision|$SLOT"
  "dinowm_um_grounded_const|$UM|uniform_motion|$SLOT dynamics.enabled=true dynamics.accel_form=const"
  "dinowm_par_grounded_const|$PAR|parabola|$SLOT dynamics.enabled=true dynamics.accel_form=const"
  "dinowm_col_grounded_const|$COL|collision|$SLOT dynamics.enabled=true dynamics.accel_form=const"
)
run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=3 loss.probe.weight=0.0 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}
echo "=== dinowm queue2 START $(date), ${#Q[@]} jobs ===" > "$QLOG"
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8))
  run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" &
  i=$((i+1)); sleep 3
done
wait
echo "=== dinowm queue2 ALL FINISHED $(date) ===" >> "$QLOG"
