#!/bin/bash
# Fill the genuinely-missing Table-2 cells (parabola/collision arms never run).
# 9 jobs, GPUs 0 and 3 only (others occupied). Same launcher pattern as run_p0_queue.sh.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT_CKPT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/aaai_p0
QLOG=$LOG/fill_queue.log
mkdir -p "$LOG"

PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"

# NAME|DATA|DOM|EXTRA   (all pusht-init, 20ep, free-rollout np8 defaults)
Q=(
  "parabola_probeF2_fr|$PAR|parabola|$PROBE"
  "collision_probeF2_fr|$COL|collision|$PROBE"
  "parabola_probeF2_structpos_pw30_fr|$PAR|parabola|$PROBE $SLOT loss.pos_weight=30"
  "collision_probeF2_structpos_pw30_fr|$COL|collision|$PROBE $SLOT loss.pos_weight=30"
  "parabola_structposvel_pw30_fr|$PAR|parabola|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  "collision_structposvel_pw30_fr|$COL|collision|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30"
  "parabola_structpos_fr_id1k|$PAR|parabola|$SLOT"
  "collision_structpos_fr_id1k|$COL|collision|$SLOT"
  "collision_grounded_const_id1k|$COL|collision|$SLOT dynamics.enabled=true dynamics.accel_form=const"
)

free_gpu() {  # only GPUs 0 and 3; no substantial (>=2GB) compute proc + >=30GB free
  for g in 0 3; do
    np=$(nvidia-smi -i $g --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>=2000' | grep -c .)
    [ "$np" -ne 0 ] && continue
    mfree=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    [ "${mfree:-0}" -ge 30000 ] && { echo $g; return; }
  done
  echo ""
}

run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 $EXTRA +init_from_ckpt=$INIT_CKPT \
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

echo "=== fill-cells queue START $(date), ${#Q[@]} jobs ===" > "$QLOG"
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  if grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null; then
    echo "SKIP $NAME (already done)" >> "$QLOG"; continue
  fi
  g=""
  while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && sleep 120; done
  run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" &
  sleep 45
done
wait
echo "=== fill-cells queue ALL FINISHED $(date) ===" >> "$QLOG"
