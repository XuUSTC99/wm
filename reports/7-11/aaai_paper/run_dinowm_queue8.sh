#!/bin/bash
# queue8 (2026-07-16) DECISIVE CONTROL for the pw300 anomaly.
# Facts so far: pw1 (pin position, no reweight) = no effect; pw300 (pin + 300x reweight)
# = significant OOD gain but ID *degrades* 30%; probe-190 shows the bypass is INTACT
# (blackbox rho 0.973). All of this says "capacity restriction", not "physics knowledge".
# Test: pin the slot to a SHUFFLED (physically meaningless) target at the same pw.
#   shuffle+pw300 helps too  -> gain is regularization; paper's claim stands.
#   shuffle+pw300 does not   -> position content matters; paper needs revision.
# Also pw300 with structured.weight=0.1 (weak pin, strong reweight) to separate
# "pinning" from "reweighting" once more.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm; QLOG=$LOG/queue8.log; mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
Q=()
for s in 3072 1234 42; do
  Q+=("dinowm_um_shufslot_pw300_s${s}|$UM|uniform_motion|$SLOT +loss.structured.shuffle_control=true loss.pos_weight=300 seed=$s")
done
for s in 3072 1234 42; do
  Q+=("dinowm_um_weakpin_pw300_s${s}|$UM|uniform_motion|loss.structured.weight=0.1 loss.structured.target=proprio loss.pos_weight=300 seed=$s")
done
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
echo "=== queue8 START $(date), ${#Q[@]} decisive-control jobs ===" > "$QLOG"
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8)); run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" & i=$((i+1)); sleep 3
done
wait
echo "=== queue8 ALL FINISHED $(date) ===" >> "$QLOG"
