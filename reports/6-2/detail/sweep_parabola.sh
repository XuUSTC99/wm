#!/bin/bash
# λ_probe × frames sweep on parabola. Grid: weight {0.1,1,10} × frames {1,2,4},
# target fixed = [proprio,action]. weight is the only on/off knob (0 would = baseline).
# Usage: sweep_parabola.sh "0 1"   (space-separated GPU ids to round-robin over)
set -u
GPUS=(${1:-0})

# Code lives on system disk, but ALL outputs/data go to /data1
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/sweep_parabola_logs                       # logs on /data1
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel             # ckpts + data on /data1
export HF_HOME=$DATA_ROOT/.cache_huggingface                   # HF cache on /data1
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"; SW=$LOG/sweep.log
echo "=== SWEEP START $(date) on GPUs ${GPUS[*]} ===" > "$SW"
echo "  STABLEWM_HOME=$STABLEWM_HOME" >> "$SW"
echo "  LOG=$LOG" >> "$SW"

# build the 9 (weight,frames) configs
CONFIGS=(); for w in 0.1 1.0 10.0; do for f in 1 2 4; do CONFIGS+=("$w:$f"); done; done

run_one () {  # gpu weight frames
  local gpu=$1 w=$2 f=$3
  local wtag=${w/./p}                      # 0.1->0p1, 1.0->1p0, 10.0->10p0
  local name="parabola_sw_w${wtag}_f${f}_id1k"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$SW"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=phyworld_parabola_id1k \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w 'loss.probe.target=[proprio,action]' loss.probe.frames=$f \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$SW"
  # eval right after
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain parabola --ckpt "$SWM/$name/${name}_epoch_20_object.ckpt" \
      --tag "w${w}f${f}" --max-trajs 500 ) > "$LOG/rollout_sw_w${wtag}_f${f}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $name" >> "$SW"
}

# Launch ALL configs in parallel (overlap mode). With 9 configs and 8 GPUs,
# the first 8 jobs go 1-per-GPU and the 9th shares GPU 0 (A800-80GB has headroom
# for 2× LeWM ≈ 30GB used). Faster wall-clock than the round-robin queue.
NG=${#GPUS[@]}
for ci in "${!CONFIGS[@]}"; do
  IFS=: read w f <<< "${CONFIGS[$ci]}"
  gpu=${GPUS[$(( ci % NG ))]}
  run_one "$gpu" "$w" "$f" &
done
wait
echo "=== SWEEP ALL DONE $(date) ===" >> "$SW"
