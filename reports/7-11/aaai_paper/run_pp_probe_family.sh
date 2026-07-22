#!/bin/bash
# Physion++ probe family -- the P0 gap in NOTE_to_physion_session_gaps.md.
#
# WHAT IS MISSING. tab:pp corroborates "injection also fails on photorealistic
# simulation" with the slot and consistency families only. The probe family --
# deep supervision, the literature's strongest competitor for shared-latent
# injection -- has never been run on Physion++, so a reviewer can answer the
# whole corroboration with "maybe soft supervision works on photorealistic
# data". The paper currently concedes this in its limitations.
#
# COMPARABILITY. Protocol copied from run_pp_seeds.sh (the sibling queue in
# flight): data=physionpp, free rollout, num_preds 8, batch 64, lr 5e-5, 20
# epochs, PushT init. Batch 64 specifically -- the older pp_fr_s* triple trains
# at 32, and mixing scales is exactly the numerator/denominator mismatch that
# script was written to avoid. The probe recipe is phyworld's probeF2: weight
# 1.0, target [proprio, action], 2 stacked frames.
#
# Two arms x 3 seeds = 6 runs, co-located with the 8 runs already on the cards
# (~18.5 GiB each of 80 GiB, so memory is not the constraint).
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
INIT=$S/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/physionpp
QLOG=$LOG/probe_queue.log
mkdir -p "$LOG"
JOB=0

# NAME|SEED|structured_w|pos_weight       (probe terms are identical across all)
Q=(
  "pp_probe_s3072|3072|0.0|1.0"
  "pp_probe_s1234|1234|0.0|1.0"
  "pp_probe_s42|42|0.0|1.0"
  "pp_probeslot_s3072|3072|1.0|30"
  "pp_probeslot_s1234|1234|1.0|30"
  "pp_probeslot_s42|42|1.0|30"
)

echo "=== physion++ probe family START $(date), ${#Q[@]} jobs ===" > "$QLOG"
for item in "${Q[@]}"; do
  IFS='|' read NAME SEED SW PW <<< "$item"
  if [ -f "$S/$NAME/${NAME}_epoch_20_object.ckpt" ]; then
    echo "SKIP $NAME (ckpt exists)" >> "$QLOG"; continue
  fi
  g=$((JOB % 8)); JOB=$((JOB + 1))
  echo "[$(date +%H:%M)] START $NAME on GPU$g (seed=$SEED sw=$SW pw=$PW)" >> "$QLOG"
  (
    cd "$LEWM" || exit 2
    CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u train.py data=physionpp \
      wandb.enabled=false tensorboard.enabled=false \
      wm.free_rollout=true wm.num_preds=8 loader.batch_size=64 \
      loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2 \
      loss.structured.weight=$SW loss.structured.target=proprio \
      loss.consistency.weight=0.0 loss.consistency.accel_weight=0.0 \
      loss.pos_weight=$PW seed=$SEED \
      output_model_name=$NAME subdir=$NAME trainer.max_epochs=20 \
      +init_from_ckpt=$INIT > "$LOG/train_${NAME}.log" 2>&1
    EC=$?
    echo "[$(date +%H:%M)] TRAIN done $NAME ec=$EC" >> "$QLOG"
    [ $EC -ne 0 ] && exit $EC
    # disk is at ~90%: keep only the checkpoint we evaluate
    ls $S/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null \
      | grep -v "_epoch_20_object.ckpt" | xargs -r rm -f
  ) &
  sleep 5
done
wait
echo "=== physion++ probe family ALL DONE $(date) ===" >> "$QLOG"
