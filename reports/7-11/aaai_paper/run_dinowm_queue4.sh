#!/bin/bash
# dinowm queue4 (2026-07-16): WAITS for the 59-run matrix to drain (CPU is the
# bottleneck, load 208/112 — launching now would slow everything down), then runs:
#   (a) physion x dinowm FR vs TF x3 seeds = strongest cross-model x cross-data evidence
#   (b) LBR curve fill-in (pw1/pw10/pw300) = cross-model mechanism curve (Fig 8 analog)
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm
QLOG=$LOG/queue4.log
mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
TFP="wm.free_rollout=false wm.num_preds=1 loader.batch_size=64"

# NAME|DATA|DOM|EXTRA   (DOM=physionpp -> physionpp eval; else id1k eval)
Q=(
  "dinowm_pp_fr_s3072|physionpp|physionpp|seed=3072 loader.batch_size=32"
  "dinowm_pp_fr_s1234|physionpp|physionpp|seed=1234 loader.batch_size=32"
  "dinowm_pp_fr_s42|physionpp|physionpp|seed=42 loader.batch_size=32"
  "dinowm_pp_tf_s3072|physionpp|physionpp|$TFP seed=3072"
  "dinowm_pp_tf_s1234|physionpp|physionpp|$TFP seed=1234"
  "dinowm_pp_tf_s42|physionpp|physionpp|$TFP seed=42"
  "dinowm_um_structpos_pw1|$UM|uniform_motion|$SLOT loss.pos_weight=1"
  "dinowm_um_structpos_pw10|$UM|uniform_motion|$SLOT loss.pos_weight=10"
  "dinowm_um_structpos_pw300|$UM|uniform_motion|$SLOT loss.pos_weight=300"
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
      trainer.max_epochs=20 num_workers=4 loss.probe.weight=0.0 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  if [ "$DOM" = "physionpp" ]; then
    CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME "$LEWM/.venv/bin/python" \
      phyworld/scripts/physion/rollout_eval_physionpp.py --ckpt "$CKPT" --device cuda:0 --tag "$NAME" \
      > "$LOG/rollout_${NAME}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME "$LEWM/.venv/bin/python" \
      phyworld/scripts/rollout_eval_id1k.py --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  fi
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}

echo "=== queue4 WAITING for matrix to drain (started $(date)) ===" > "$QLOG"
# wait until the 59-run matrix is mostly done: <=6 dinov2 train procs alive
for i in $(seq 1 720); do
  n=$(ps -eo cmd 2>/dev/null | grep '[t]rain.py' | grep -c dinov2)
  [ "$n" -le 6 ] && { echo "[$(date +%H:%M)] matrix drained (n=$n), launching" >> "$QLOG"; break; }
  sleep 60
done
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD|h= 64" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8))
  run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" &
  i=$((i+1)); sleep 5
done
wait
echo "=== queue4 ALL FINISHED $(date) ===" >> "$QLOG"
