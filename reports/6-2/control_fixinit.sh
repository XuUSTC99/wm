#!/bin/bash
# Minimal control: re-run parabola w∈{0.1,1.0,50} × f=2 with the FIXED init
# (_remap_old_vit_keys → encoder loads real pusht weights, loaded=216/216).
# Old broken-init ckpts kept as parabola_sw_w*_f2_id1k for comparison.
set -u
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
DATA_ROOT=/data1/likun-share/junjxu
LOG=$DATA_ROOT/runs/control_fixinit_logs
export STABLEWM_HOME=$DATA_ROOT/.stable_worldmodel
export HF_HOME=$DATA_ROOT/.cache_huggingface
INIT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
mkdir -p "$LOG"

# config:gpu  (w, gpu)
declare -A GPU=( [0p1]=0 [1p0]=1 [50p0]=5 )
declare -A WVAL=( [0p1]=0.1 [1p0]=1.0 [50p0]=50.0 )

train_one() {
  local wtag=$1 gpu=$2 w=${WVAL[$1]}
  local name=parabola_fixinit_w${wtag}_f2_id1k
  rm -rf "$STABLEWM_HOME/$name"
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=phyworld_parabola_id1k \
      output_model_name=$name subdir=$name wandb.enabled=False trainer.max_epochs=20 \
      loss.probe.weight=$w 'loss.probe.target=[proprio,action]' loss.probe.frames=2 \
      'data.dataset.keys_to_cache=[pixels,action,proprio]' \
      +init_from_ckpt=$INIT ) > "$LOG/train_${name}.log" 2>&1
  echo "[train done $(date +%H:%M:%S)] $name (exit $?)" >> "$LOG/control.log"

  # rollout eval (same protocol as sweep)
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    le-wm/.venv/bin/python phyworld/scripts/rollout_eval_id1k.py \
      --domain parabola \
      --ckpt $STABLEWM_HOME/$name/${name}_epoch_20_object.ckpt \
      --tag fixinit_w${wtag} --max-trajs 500 ) > "$LOG/rollout_parabola_w${wtag}.log" 2>&1
  echo "[eval done $(date +%H:%M:%S)] $name (exit $?)" >> "$LOG/control.log"
}

echo "=== CONTROL-FIXINIT START $(date) ===" > "$LOG/control.log"
for wtag in 0p1 1p0 50p0; do
  train_one "$wtag" "${GPU[$wtag]}" &
done
wait
echo "=== CONTROL-FIXINIT ALL DONE $(date) ===" >> "$LOG/control.log"
