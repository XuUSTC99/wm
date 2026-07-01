#!/bin/bash
# Re-train the 9 ckpts that diagnostic_report.md's §1-§5 numbers depend on,
# using the FIXED init (train.py:_remap_old_vit_keys, loaded=216/216).
#
# Grid: 3 domains × 3 λ_probe ∈ {1, 5, 10} × frames=2
# target=[proprio, vel_col]   (vel_col: action for parabola/uniform, state for collision)
#
# Ckpt naming: <dom>_rerun_w<W>_f2_id1k   (distinct from broken-init <dom>_sw_*)
# Log dir:     /data1/likun-share/junjxu/runs/6-24_rerun_logs/
#
# GPU layout: 9 jobs round-robin over GPUs passed via $1 (default 3-7 because
# mf4-pos-only may still be on 0-2). 5 GPUs → 4 GPUs run 2 jobs, 1 GPU runs 1.
# Usage: ./rerun_diagnostic_inputs.sh "3 4 5 6 7"
set -u
GPUS=(${1:-3 4 5 6 7})
NG=${#GPUS[@]}

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/6-24_rerun_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"
MASTER=$LOG/orchestrator.log
echo "=== rerun-diag START $(date) on GPUs ${GPUS[*]} ===" > "$MASTER"

# Build 9 jobs: (dom, datacfg, vel_col, lambda)
JOBS=()
for dom_meta in "parabola:phyworld_parabola_id1k:action" \
                "uniform_motion:phyworld_uniform_motion_id1k:action" \
                "collision:phyworld_collision_id1k_st:state"; do
  IFS=: read dom cfg vel <<< "$dom_meta"
  for w in 1.0 5.0 10.0; do
    JOBS+=("$dom:$cfg:$vel:$w")
  done
done
echo "  Total jobs: ${#JOBS[@]}" >> "$MASTER"

run_one () {
  local gpu=$1 dom=$2 cfg=$3 vel=$4 w=$5
  local wtag=${w/./p}
  local name="${dom}_rerun_w${wtag}_f2_id1k"
  echo "[train $(date +%H:%M:%S)] GPU$gpu $name (λ=$w, target=[proprio,$vel])" >> "$MASTER"
  rm -rf "$SWM/$name"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$cfg \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w "loss.probe.target=[proprio,$vel]" loss.probe.frames=2 \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  local ec=$?
  echo "[train done $(date +%H:%M:%S)] $name (exit $ec)" >> "$MASTER"
  [ $ec -ne 0 ] && return $ec

  local ckpt="$SWM/$name/${name}_epoch_20_object.ckpt"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $dom --ckpt "$ckpt" --tag "rerun_w${wtag}" --max-trajs 500 ) \
      > "$LOG/rollout_${dom}_w${wtag}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] ${dom}_w${wtag}" >> "$MASTER"
}

for g in "${!GPUS[@]}"; do
  (
    for ji in "${!JOBS[@]}"; do
      if [ $(( ji % NG )) -eq $g ]; then
        IFS=: read dom cfg vel w <<< "${JOBS[$ji]}"
        run_one "${GPUS[$g]}" "$dom" "$cfg" "$vel" "$w"
      fi
    done
  ) &
done
wait
echo "=== rerun-diag ALL DONE $(date) ===" >> "$MASTER"
