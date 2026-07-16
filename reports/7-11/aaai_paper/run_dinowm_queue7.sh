#!/bin/bash
# queue7 (2026-07-16): fill the LeWM-vs-dinowm experiment gaps that matter.
#  (a) C3 augmentation — LeWM's strongest synthetic OOD lever (app0.5 halves nMSE).
#      On dinowm the encoder is FROZEN -> pixel-space aug cannot reshape the encoder.
#      Prediction: aug should LOSE most of its power. That's a mechanism claim worth testing:
#      it would show aug works *through* the encoder, complementing the injection story.
#  (b) C4 family completion: free accel MLP (dyn family, LeWM had it; dinowm only had const)
#  (c) probe+structpos combo (the one LeWM combo cell never run on dinowm)
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm; QLOG=$LOG/queue7.log; mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
Q=(
  # (a) C3 augmentation x 3 domains x 3 seeds  (LeWM: app0.5 uniform 0.131->0.068)
  "dinowm_um_app05_s3072|$UM|uniform_motion|aug.appearance=0.5 seed=3072"
  "dinowm_um_app05_s1234|$UM|uniform_motion|aug.appearance=0.5 seed=1234"
  "dinowm_um_app05_s42|$UM|uniform_motion|aug.appearance=0.5 seed=42"
  "dinowm_par_app05_s3072|$PAR|parabola|aug.appearance=0.5 seed=3072"
  "dinowm_par_app05_s1234|$PAR|parabola|aug.appearance=0.5 seed=1234"
  "dinowm_par_app05_s42|$PAR|parabola|aug.appearance=0.5 seed=42"
  "dinowm_col_scale05_s3072|$COL|collision|aug.scale=0.5 seed=3072"
  "dinowm_col_scale05_s1234|$COL|collision|aug.scale=0.5 seed=1234"
  "dinowm_col_scale05_s42|$COL|collision|aug.scale=0.5 seed=42"
  # (b) C4 dyn family: free accel MLP (LeWM had this arm; dinowm only ran const)
  "dinowm_um_dyn_mlp|$UM|uniform_motion|$SLOT dynamics.enabled=true dynamics.accel_form=mlp"
  "dinowm_par_dyn_mlp|$PAR|parabola|$SLOT dynamics.enabled=true dynamics.accel_form=mlp"
  "dinowm_col_dyn_mlp|$COL|collision|$SLOT dynamics.enabled=true dynamics.accel_form=mlp"
  # (c) probe+structpos combo (LeWM's 2x2 cell)
  "dinowm_um_probe_structpos_pw30|$UM|uniform_motion|$PROBE $SLOT loss.pos_weight=30"
  "dinowm_par_probe_structpos_pw30|$PAR|parabola|$PROBE $SLOT loss.pos_weight=30"
  "dinowm_col_probe_structpos_pw30|$COL|collision|$PROBE $SLOT loss.pos_weight=30"
)
run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=4 loss.probe.weight=0.0 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME "$LEWM/.venv/bin/python" \
    phyworld/scripts/rollout_eval_id1k.py --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}
echo "=== queue7 WAITING for queue6 to drain ($(date)) ===" > "$QLOG"
for i in $(seq 1 90); do
  n=$(ps -eo cmd 2>/dev/null | grep '[t]rain.py' | grep -c dinov2)
  [ "$n" -le 4 ] && { echo "[$(date +%H:%M)] drained (n=$n), launching ${#Q[@]} jobs" >> "$QLOG"; break; }
  sleep 60
done
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8)); run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" & i=$((i+1)); sleep 3
done
wait
echo "=== queue7 ALL FINISHED $(date) ===" >> "$QLOG"
