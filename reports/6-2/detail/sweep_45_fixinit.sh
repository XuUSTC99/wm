#!/bin/bash
# 45-config λ_probe × frames sweep, FIXED init (loaded=216), memory-safe on few GPUs.
# 3 domains × 5 weights {0.1,1,10,30,50} × 3 frames {1,2,4} = 45 configs.
# Runs in WAVES capped at MAXPER jobs/GPU (default 4 → 56GB/80GB) so it is safe
# even when only a subset of GPUs is free. Round-robin assigns each wave's jobs
# across the given GPUs. All logs -> sweep_three_domains_logs (extract reads there).
# Usage: sweep_45_fixinit.sh "4 5 6 7" [MAXPER]
set -u
GPUS=(${1:-4 5 6 7})
MAXPER=${2:-4}
NG=${#GPUS[@]}
WAVE=$(( NG * MAXPER ))

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/sweep_three_domains_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"; SW=$LOG/sweep.log
echo "=== SWEEP45 START $(date) on GPUs ${GPUS[*]} (MAXPER=$MAXPER, wave=$WAVE) ===" > "$SW"

DOMAINS=(
  "parabola:phyworld_parabola_id1k:action:parabola"
  "uniform:phyworld_uniform_motion_id1k:action:uniform"
  "collision:phyworld_collision_id1k_st:state:collision"
)
CONFIGS=()
for dom_meta in "${DOMAINS[@]}"; do
  for w in 0.1 1.0 10.0 30.0 50.0; do
    for f in 1 2 4; do
      CONFIGS+=("$dom_meta:$w:$f")
    done
  done
done
echo "  Total configs: ${#CONFIGS[@]}" >> "$SW"

cache_list () { case "$1" in collision) echo '[pixels,action,proprio,state]';; *) echo '[pixels,action,proprio]';; esac; }
eval_dom ()  { case "$1" in parabola) echo parabola;; uniform) echo uniform_motion;; collision) echo collision;; esac; }

run_one () {  # gpu domain datacfg vel prefix weight frames
  local gpu=$1 dom=$2 cfg=$3 vel=$4 pre=$5 w=$6 f=$7
  local wtag=${w/./p}
  local name="${pre}_sw_w${wtag}_f${f}_id1k"
  local cache=$(cache_list "$dom")
  rm -rf "$SWM/$name"
  echo "[train $(date +%H:%M:%S)] $name on GPU$gpu" >> "$SW"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$cfg \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w "loss.probe.target=[proprio,$vel]" loss.probe.frames=$f \
      loader.num_workers=2 num_workers=2 \
      "data.dataset.keys_to_cache=$cache" \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  local rc=$?
  echo "[train done $(date +%H:%M:%S)] $name (exit $rc)" >> "$SW"
  [ $rc -ne 0 ] && return $rc
  local edom=$(eval_dom "$dom")
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $edom --ckpt "$SWM/$name/${name}_epoch_20_object.ckpt" \
      --tag "w${w}f${f}" --max-trajs 500 ) > "$LOG/rollout_${pre}_w${wtag}_f${f}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $name" >> "$SW"
}

N=${#CONFIGS[@]}
wave=0
for (( start=0; start<N; start+=WAVE )); do
  wave=$(( wave + 1 ))
  echo "--- WAVE $wave : configs $start..$(( start+WAVE-1 < N ? start+WAVE-1 : N-1 )) ---" >> "$SW"
  slot=0
  for (( ci=start; ci<start+WAVE && ci<N; ci++ )); do
    IFS=: read dom cfg vel pre w f <<< "${CONFIGS[$ci]}"
    gpu=${GPUS[$(( slot % NG ))]}
    run_one "$gpu" "$dom" "$cfg" "$vel" "$pre" "$w" "$f" &
    slot=$(( slot + 1 ))
  done
  wait
done
echo "=== SWEEP45 ALL DONE $(date) ===" >> "$SW"
