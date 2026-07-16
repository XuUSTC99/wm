#!/bin/bash
# queue9 (2026-07-16): complete dinowm's 10-arm x 3-domain scan so it mirrors
# LeWM's Fig-16 heatmap exactly. Missing 4 arms (configs verified against the
# LeWM runs' own config.yaml):
#   [probe]+structpos : probe.weight=1 + structured.weight=1 + pw30
#   [dyn] free MLP    : structured=1 + dynamics(mlp, learnable_accel)
#   [free] label-free : structured=0 + dynamics(const)  <- the no-label arm
#   [free] grounded   : structured=1 + dynamics(const)  <- its labeled control
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm; QLOG=$LOG/queue9.log; mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
DYNC="dynamics.enabled=true dynamics.accel_form=const dynamics.learnable_accel=true"
DYNM="dynamics.enabled=true dynamics.accel_form=mlp dynamics.learnable_accel=true"
Q=(
  "dinowm_um_probe_structpos_pw30|$UM|uniform_motion|$PROBE $SLOT loss.pos_weight=30"
  "dinowm_par_probe_structpos_pw30|$PAR|parabola|$PROBE $SLOT loss.pos_weight=30"
  "dinowm_col_probe_structpos_pw30|$COL|collision|$PROBE $SLOT loss.pos_weight=30"
  "dinowm_um_dyn_mlp|$UM|uniform_motion|$SLOT $DYNM"
  "dinowm_par_dyn_mlp|$PAR|parabola|$SLOT $DYNM"
  "dinowm_col_dyn_mlp|$COL|collision|$SLOT $DYNM"
  "dinowm_um_labelfree_const|$UM|uniform_motion|loss.structured.weight=0.0 $DYNC"
  "dinowm_par_labelfree_const|$PAR|parabola|loss.structured.weight=0.0 $DYNC"
  "dinowm_col_labelfree_const|$COL|collision|loss.structured.weight=0.0 $DYNC"
  "dinowm_um_grounded2_const|$UM|uniform_motion|$SLOT $DYNC"
  "dinowm_par_grounded2_const|$PAR|parabola|$SLOT $DYNC"
  "dinowm_col_grounded2_const|$COL|collision|$SLOT $DYNC"
)
run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=4 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME "$LEWM/.venv/bin/python" \
    phyworld/scripts/rollout_eval_id1k.py --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}
echo "=== queue9 START $(date), ${#Q[@]} jobs (complete the 30-cell scan) ===" > "$QLOG"
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8)); run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" & i=$((i+1)); sleep 3
done
wait
echo "=== queue9 ALL FINISHED $(date) ===" >> "$QLOG"
