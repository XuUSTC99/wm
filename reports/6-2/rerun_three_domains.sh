#!/bin/bash
# Re-run piwm_three_domains.md experiments from scratch on the A500 box.
# 3 domains × 4 arms = 12 trainings + 12 evals (rollout K=4).
#
# Arms (probe.weight=0 ⇒ baseline; otherwise PIWM-style probe):
#   baseline   : no probe (paperinit)
#   pos-only   : probe.weight=1, target=proprio,                frames=1
#   pos+vel    : probe.weight=1, target=[proprio,<vel_col>],    frames=1
#   mf4        : probe.weight=1, target=[proprio,<vel_col>],    frames=4
# vel_col per domain: parabola/uniform_motion = action; collision = state.
#
# GPU layout: rotates 12 jobs across whatever GPUs are passed via $1.
# Usage: ./rerun_three_domains.sh "0 1 2 3 4 5 6 7"
#        ./rerun_three_domains.sh "0 1 2 3"
set -u
GPUS=(${1:-0 1 2 3 4 5 6 7})
NG=${#GPUS[@]}

ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/6-2_three_domains_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
SWM=$STABLEWM_HOME
INIT=$SWM/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"
MASTER=$LOG/orchestrator.log
echo "=== START $(date) on GPUs ${GPUS[*]} ===" > "$MASTER"
echo "  STABLEWM_HOME=$STABLEWM_HOME" >> "$MASTER"
echo "  LOG=$LOG" >> "$MASTER"

# Build the full job list: each entry = "domain:datacfg:vel_col:arm"
JOBS=()
for dom_cfg_vel in "parabola:phyworld_parabola_id1k:action" \
                  "uniform_motion:phyworld_uniform_motion_id1k:action" \
                  "collision:phyworld_collision_id1k_st:state"; do
  IFS=: read dom cfg vel <<< "$dom_cfg_vel"
  for arm in baseline posonly posvel mf4; do
    JOBS+=("$dom:$cfg:$vel:$arm")
  done
done
echo "  Total jobs: ${#JOBS[@]}" >> "$MASTER"

# Map (arm) → (probe args + output_model_name suffix)
arm_args () {
  local arm=$1 vel=$2 dom=$3
  case "$arm" in
    baseline) echo "${dom}_paperinit_id1k|loss.probe.weight=0.0";;
    posonly)  echo "${dom}_piwm_probe_id1k|loss.probe.weight=1.0 loss.probe.target=proprio loss.probe.frames=1";;
    posvel)   echo "${dom}_piwm_posvel_id1k|loss.probe.weight=1.0 loss.probe.target=[proprio,$vel] loss.probe.frames=1";;
    mf4)      echo "${dom}_piwm_mf4_id1k|loss.probe.weight=1.0 loss.probe.target=[proprio,$vel] loss.probe.frames=4";;
  esac
}

# Eval domain name (rollout_eval_id1k expects "uniform_motion", "collision", "parabola")
eval_dom () {
  case "$1" in
    uniform_motion) echo "uniform_motion";;
    collision) echo "collision";;
    parabola) echo "parabola";;
  esac
}

run_job () {  # gpu domain datacfg vel_col arm
  local gpu=$1 dom=$2 cfg=$3 vel=$4 arm=$5
  local meta=$(arm_args "$arm" "$vel" "$dom")
  local name="${meta%%|*}"
  local probe_args="${meta##*|}"
  local tag="$arm"

  echo "[train $(date +%H:%M:%S)] GPU$gpu $name (probe: $probe_args)" >> "$MASTER"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$cfg \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      $probe_args \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  local train_ec=$?
  echo "[train done $(date +%H:%M:%S)] $name (exit $train_ec)" >> "$MASTER"
  if [ $train_ec -ne 0 ]; then return $train_ec; fi

  # eval right after on the same GPU
  local ckpt="$SWM/$name/${name}_epoch_20_object.ckpt"
  local edom=$(eval_dom "$dom")
  echo "[eval  $(date +%H:%M:%S)] GPU$gpu $name → $edom $tag" >> "$MASTER"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu \
      STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
      "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $edom --ckpt "$ckpt" --tag "$tag" --max-trajs 500 ) \
      > "$LOG/rollout_${dom}_${tag}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] ${dom}_${tag} (exit $?)" >> "$MASTER"
}

# Round-robin jobs over GPUs; each GPU runs its queue sequentially, GPUs in parallel
for g in "${!GPUS[@]}"; do
  (
    for ji in "${!JOBS[@]}"; do
      if [ $(( ji % NG )) -eq $g ]; then
        IFS=: read dom cfg vel arm <<< "${JOBS[$ji]}"
        run_job "${GPUS[$g]}" "$dom" "$cfg" "$vel" "$arm"
      fi
    done
    echo "[GPU${GPUS[$g]} DONE $(date +%H:%M:%S)]" >> "$MASTER"
  ) &
done
wait
echo "=== ALL DONE $(date) ===" >> "$MASTER"
