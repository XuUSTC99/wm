#!/bin/bash
# probe-190 bypass test on dinowm (Fig 15 cross-model analog) — THE mechanism evidence.
# Paper's mechanism: injection fails because position is redundantly encoded across the
# black-box 190 dims, so prediction routes around the slot (bypass).
# dinowm's pw300 DID work -> prediction: its bypass should be measurably WEAKER.
# Compares blackbox[2:192] vs all-192 vs slot[0:2] rho on: baseline FR / pw30 / pw300.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm; QLOG=$LOG/probe190.log; mkdir -p "$LOG"
# wait for training queues to drain so we don't fight for GPU
for i in $(seq 1 120); do
  n=$(ps -eo cmd 2>/dev/null | grep '[t]rain.py' | grep -c dinov2)
  [ "$n" -le 4 ] && break
  sleep 60
done
echo "=== probe190 START $(date) ===" > "$QLOG"
i=0
for spec in \
  "uniform_motion|dinowm_um_fr_s3072" \
  "uniform_motion|dinowm_um_structpos_pw30" \
  "uniform_motion|dinowm_um_structpos_pw300" \
  "collision|dinowm_col_fr_s3072" \
  "collision|dinowm_col_structpos_pw30" \
  "parabola|dinowm_par_fr_s3072" \
  "parabola|dinowm_par_structpos_pw30" ; do
  IFS='|' read DOM NAME <<< "$spec"
  CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "SKIP $NAME (no ckpt)" >> "$QLOG"; continue; }
  g=$((i % 8))
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/probe_dim_subset.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 400 \
      > "$LOG/probe190_${NAME}.log" 2>&1 &
  echo "[$(date +%H:%M)] probe190 $NAME on GPU$g" >> "$QLOG"
  i=$((i+1)); sleep 2
done
wait
echo "=== probe190 ALL FINISHED $(date) ===" >> "$QLOG"
