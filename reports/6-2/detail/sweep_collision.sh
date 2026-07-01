#!/bin/bash
# λ_probe × frames sweep on collision. Grid: weight {0.1,1,10} × frames {1,2,4},
# target fixed = [proprio,state] (collision: action is accel, velocity lives in state).
# Includes pixels-in-RAM optimization: keys_to_cache=[pixels,action,proprio,state].
# Usage: sweep_collision.sh "0 1 2 3 4 5 6 7"
set -u
GPUS=(${1:-0})

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/sweep_collision_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"; SW=$LOG/sweep.log
echo "=== SWEEP START $(date) on GPUs ${GPUS[*]} ===" > "$SW"

CONFIGS=(); for w in 0.1 1.0 10.0; do for f in 1 2 4; do CONFIGS+=("$w:$f"); done; done

run_one () {  # gpu weight frames
  local gpu=$1 w=$2 f=$3
  local wtag=${w/./p}
  local name="collision_sw_w${wtag}_f${f}_id1k"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$SW"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=phyworld_collision_id1k_st \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w 'loss.probe.target=[proprio,state]' loss.probe.frames=$f \
      'data.dataset.keys_to_cache=[pixels,action,proprio,state]' \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$SW"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain collision --ckpt "$SWM/$name/${name}_epoch_20_object.ckpt" \
      --tag "w${w}f${f}" --max-trajs 500 ) > "$LOG/rollout_sw_w${wtag}_f${f}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $name" >> "$SW"
}

NG=${#GPUS[@]}
for ci in "${!CONFIGS[@]}"; do
  IFS=: read w f <<< "${CONFIGS[$ci]}"
  gpu=${GPUS[$(( ci % NG ))]}
  run_one "$gpu" "$w" "$f" &
done
wait
echo "=== SWEEP ALL DONE $(date) ===" >> "$SW"
