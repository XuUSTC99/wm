#!/bin/bash
# Evaluate the Physion++ seed-replication runs as their checkpoints land.
#
# run_pp_seeds.sh only trains -- this is the missing half. Same pattern as
# eval_pp_probe_family.sh: block on each run's epoch-20 checkpoint, then run
# the per-scenario rollout eval that tab:pp is built from.
#
# Eight runs share four GPUs here (two evals per card); each eval is short
# next to the ~4h training it waits on.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/physionpp
QLOG=$LOG/seeds_eval.log
DEADLINE=$(( $(date +%s) + 8*3600 ))   # give training 8h before giving up

NAMES=(pp_fr_bs64_s1234 pp_fr_bs64_s42
       pp_struct_s1234 pp_struct_s42
       pp_cons_s1234 pp_cons_s42
       pp_consacc_s1234 pp_consacc_s42)

echo "=== physion++ seed eval START $(date) ===" > "$QLOG"
i=0
for n in "${NAMES[@]}"; do
  (
    CK="$S/$n/${n}_epoch_20_object.ckpt"
    while [ ! -f "$CK" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 180; done
    if [ ! -f "$CK" ]; then echo "[$(date +%H:%M)] TIMEOUT $n" >> "$QLOG"; exit 1; fi
    # the trainer writes the file before it finishes flushing; give it a moment
    sleep 30
    g=$(( i % 4 ))
    echo "[$(date +%H:%M)] EVAL $n on GPU$g" >> "$QLOG"
    cd "$ROOT" || exit 2
    CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$S HF_HOME=$HF_HOME \
      "$LEWM/.venv/bin/python" phyworld/scripts/physion/rollout_eval_physionpp.py \
        --ckpt "$CK" --device cuda:0 --tag "$n" \
        > "$LOG/rollout_${n}.log" 2>&1
    echo "[$(date +%H:%M)] EVAL done $n ec=$?" >> "$QLOG"
  ) &
  i=$((i+1))
done
wait
echo "=== physion++ seed eval ALL DONE $(date) ===" >> "$QLOG"
