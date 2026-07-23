#!/bin/bash
# Three-seed the Physion++ injection table (2026-07-22).
#
# WHAT IS MISSING. tab:pp compares four arms -- free rollout, +slot, +consistency,
# +consistency+accel -- and all four are single-seed (3072). It is the last
# single-seed table in the paper.
#
# WHY FR IS RE-RUN TOO. There are already three seeds of a Physion++ free-rollout
# baseline (pp_fr_s3072/s1234/s42), but they train at batch size 32 while every
# arm in tab:pp trains at 64. Mixing them would repeat the mistake the fig-2 scan
# had for months: a numerator and a denominator measured under different
# settings. So FR is re-run at 64 alongside the injection arms, and the existing
# bs-32 triple stays where it is used (the FR-vs-TF comparison of fig:fr).
#
# Arm configs read back from each seed-3072 run's own saved config.yaml:
#   pp_fr        structured=0.0  consistency=0.0            pos_weight=1
#   pp_struct    structured=1.0  consistency=0.0            pos_weight=30
#   pp_cons      structured=1.0  consistency=1.0            pos_weight=30
#   pp_consacc   structured=1.0  consistency=1.0 accel=0.5  pos_weight=30
# All: free rollout, num_preds=8, batch 64, lr 5e-5, 20 epochs, PushT init.
#
# 8 runs, one per GPU. A single Physion++ run took ~3.2h wall clock when it had
# a card to itself, so this should land in roughly that time rather than 8x it.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
INIT=$S/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/physionpp
QLOG=$LOG/seeds_queue.log
mkdir -p "$LOG"
JOB=0

# NAME|SEED|structured_w|consistency_w|accel_w|pos_weight
Q=(
  "pp_fr_bs64_s1234|1234|0.0|0.0|0.0|1.0"
  "pp_fr_bs64_s42|42|0.0|0.0|0.0|1.0"
  "pp_struct_s1234|1234|1.0|0.0|0.0|30"
  "pp_struct_s42|42|1.0|0.0|0.0|30"
  "pp_cons_s1234|1234|1.0|1.0|0.0|30"
  "pp_cons_s42|42|1.0|1.0|0.0|30"
  "pp_consacc_s1234|1234|1.0|1.0|0.5|30"
  "pp_consacc_s42|42|1.0|1.0|0.5|30"
)

echo "=== physion++ seed queue START $(date), ${#Q[@]} jobs ===" > "$QLOG"
for item in "${Q[@]}"; do
  IFS='|' read NAME SEED SW CW AW PW <<< "$item"
  if [ -f "$S/$NAME/${NAME}_epoch_20_object.ckpt" ]; then
    echo "SKIP $NAME (ckpt exists)" >> "$QLOG"; continue
  fi
  g=$((JOB % 8)); JOB=$((JOB + 1))
  echo "[$(date +%H:%M)] START $NAME on GPU$g (seed=$SEED sw=$SW cw=$CW aw=$AW pw=$PW)" >> "$QLOG"
  (
    cd "$LEWM" || exit 2
    CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u train.py data=physionpp \
      wandb.enabled=false tensorboard.enabled=false \
      wm.free_rollout=true wm.num_preds=8 loader.batch_size=64 \
      loss.structured.weight=$SW loss.structured.target=proprio \
      loss.consistency.weight=$CW loss.consistency.accel_weight=$AW \
      loss.pos_weight=$PW seed=$SEED \
      output_model_name=$NAME subdir=$NAME trainer.max_epochs=20 \
      +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
    EC=$?
    echo "[$(date +%H:%M)] TRAIN done $NAME ec=$EC" >> "$QLOG"
    [ $EC -ne 0 ] && exit $EC
    # keep only the checkpoint we evaluate
    ls $S/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null \
      | grep -v "_epoch_20_object.ckpt" | xargs -r rm -f
  ) &
  sleep 5
done
wait
echo "=== physion++ seed queue ALL DONE $(date) ===" >> "$QLOG"
