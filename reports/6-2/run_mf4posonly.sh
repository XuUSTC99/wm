#!/bin/bash
# Extra 5th arm: mf4-pos-only — frames=4, target=proprio (NO velocity supervision).
# Tests the hypothesis: "if encoder encodes clean per-frame position, then K=4 inference
# can recover velocity from frame-to-frame differences in the 4-frame window — without
# explicit velocity supervision".
#
# Comparison vs existing arms (already trained, A800 fixed-init, loaded=216):
#   pos-only (frames=1, target=proprio)             — single-frame pos supervision only
#   mf4      (frames=4, target=[proprio,vel])       — multi-frame pos + EXPLICIT vel
#   THIS:    (frames=4, target=proprio)             — multi-frame pos, NO explicit vel
#
# 3 jobs (one per domain), full parallel on 3 GPUs.
# Usage: ./run_mf4posonly.sh "0 1 2"     (default: 0 1 2)
set -u
GPUS=(${1:-0 1 2})

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/6-2_three_domains_logs    # same dir as A800 fixed-init re-run
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"
MASTER=$LOG/mf4posonly.log
echo "=== mf4-pos-only START $(date) on GPUs ${GPUS[*]} ===" > "$MASTER"

# (domain, datacfg) — vel_col irrelevant since target=proprio only
JOBS=(
  "parabola:phyworld_parabola_id1k"
  "uniform_motion:phyworld_uniform_motion_id1k"
  "collision:phyworld_collision_id1k_st"
)

run_job () {
  local gpu=$1 dom=$2 cfg=$3
  local name="${dom}_piwm_mf4posonly_id1k"
  local tag="mf4posonly"

  echo "[train $(date +%H:%M:%S)] GPU$gpu $name" >> "$MASTER"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$cfg \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=1.0 loss.probe.target=proprio loss.probe.frames=4 \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  local ec=$?
  echo "[train done $(date +%H:%M:%S)] $name (exit $ec)" >> "$MASTER"
  if [ $ec -ne 0 ]; then return $ec; fi

  local ckpt="$SWM/$name/${name}_epoch_20_object.ckpt"
  echo "[eval  $(date +%H:%M:%S)] GPU$gpu $name → $dom $tag" >> "$MASTER"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
      STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
      "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $dom --ckpt "$ckpt" --tag "$tag" --max-trajs 500 ) \
      > "$LOG/rollout_${dom}_${tag}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] ${dom}_${tag} (exit $?)" >> "$MASTER"
}

i=0
for j in "${JOBS[@]}"; do
  IFS=: read dom cfg <<< "$j"
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  run_job "$gpu" "$dom" "$cfg" &
  i=$((i+1))
done
wait
echo "=== mf4-pos-only ALL DONE $(date) ===" >> "$MASTER"
