#!/bin/bash
# λ_probe × frames sweep on parabola. Grid: weight {0.1,1,10} × frames {1,2,4},
# target fixed = [proprio,action]. weight is the only on/off knob (0 would = baseline).
# Usage: sweep_parabola.sh "0 1"   (space-separated GPU ids to round-robin over)
set -u
GPUS=(${1:-0})
ROOT=/home/qlib/am/wm; LEWM=$ROOT/le-wm; LOG=$ROOT/reports/6-2/logs; SWM=/home/qlib/.stable_worldmodel
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"; SW=$LOG/sweep.log
echo "=== SWEEP START $(date) on GPUs ${GPUS[*]} ===" > "$SW"

# build the 9 (weight,frames) configs
CONFIGS=(); for w in 0.1 1.0 10.0; do for f in 1 2 4; do CONFIGS+=("$w:$f"); done; done

run_one () {  # gpu weight frames
  local gpu=$1 w=$2 f=$3
  local wtag=${w/./p}                      # 0.1->0p1, 1.0->1p0, 10.0->10p0
  local name="parabola_sw_w${wtag}_f${f}_id1k"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$SW"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    .venv/bin/python -u train.py data=phyworld_parabola_id1k \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w 'loss.probe.target=[proprio,action]' loss.probe.frames=$f \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$SW"
  # eval right after
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python phyworld/scripts/rollout_eval_id1k.py \
      --domain parabola --ckpt "$SWM/$name/${name}_epoch_20_object.ckpt" \
      --tag "w${w}f${f}" --max-trajs 500 ) > "$LOG/rollout_sw_w${wtag}_f${f}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $name" >> "$SW"
}

# round-robin configs over GPUs; each GPU runs its queue sequentially, GPUs in parallel
NG=${#GPUS[@]}
for g in "${!GPUS[@]}"; do
  (
    for ci in "${!CONFIGS[@]}"; do
      if [ $(( ci % NG )) -eq $g ]; then
        IFS=: read w f <<< "${CONFIGS[$ci]}"
        run_one "${GPUS[$g]}" "$w" "$f"
      fi
    done
  ) &
done
wait
echo "=== SWEEP ALL DONE $(date) ===" >> "$SW"
