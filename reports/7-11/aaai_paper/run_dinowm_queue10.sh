#!/bin/bash
# queue10 (2026-07-16): seed-harden the 30-cell scan. LeWM's Fig-16 is mostly
# single-seed (marked in the report); dinowm's measured injection-arm std reaches
# 0.087 -- large enough to flip a cell. Add s1234/s42 to every arm that is still
# single-seed, so the cross-model heatmap is 3-seed throughout = strictly more
# solid than the LeWM figure it mirrors.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm; QLOG=$LOG/queue10.log; mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
DYNC="dynamics.enabled=true dynamics.accel_form=const dynamics.learnable_accel=true"
DYNM="dynamics.enabled=true dynamics.accel_form=mlp dynamics.learnable_accel=true"
declare -A DATA=( [um]=$UM [par]=$PAR [col]=$COL )
declare -A DOM=( [um]=uniform_motion [par]=parabola [col]=collision )
declare -A ARM=(
  [structpos_plain]="$SLOT"
  [posvel_pw30]="loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  [grounded_const]="$SLOT $DYNC"
  [probe_structpos_pw30]="$PROBE $SLOT loss.pos_weight=30"
  [dyn_mlp]="$SLOT $DYNM"
  [labelfree_const]="loss.structured.weight=0.0 $DYNC"
  [grounded2_const]="$SLOT $DYNC"
)
Q=()
for d in um par col; do
  for a in structpos_plain posvel_pw30 grounded_const probe_structpos_pw30 dyn_mlp labelfree_const; do
    for s in 1234 42; do
      Q+=("dinowm_${d}_${a}_s${s}|${DATA[$d]}|${DOM[$d]}|${ARM[$a]} seed=$s")
    done
  done
done
run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=2 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME "$LEWM/.venv/bin/python" \
    phyworld/scripts/rollout_eval_id1k.py --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
    > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}
echo "=== queue10 WAIT for queue8/9 to drain ($(date)) ===" > "$QLOG"
for i in $(seq 1 120); do
  n=$(ps -eo cmd 2>/dev/null | grep '[t]rain.py' | grep -c dinov2)
  [ "$n" -le 8 ] && { echo "[$(date +%H:%M)] drained (n=$n), launching ${#Q[@]} jobs" >> "$QLOG"; break; }
  sleep 60
done
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8)); run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" & i=$((i+1)); sleep 3
done
wait
echo "=== queue10 ALL FINISHED $(date) ===" >> "$QLOG"
