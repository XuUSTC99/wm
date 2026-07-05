#!/bin/bash
# Pixel-rollout pipeline for one model: train a proj-space universal decoder
# (or reuse one) -> pixel rollout PSNR by horizon. The "right lens" (position-
# weighted) vs the diluted latent-aggregate metric.
# Usage: ./run_pixel.sh <GPU> <NAME> [DECODER_PT]
#   NAME       : model dir under STABLEWM_HOME (e.g. uniform_motion_structcv_fr_id1k)
#   DECODER_PT : optional existing decoder .pt -> skip decoder training
set -u
GPU=$1; NAME=$2; DEC=${3:-}

ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
DECDIR=/data1/likun-share/junjxu/runs/decoder_viz/universal_proj
LOG=/data1/likun-share/junjxu/runs/structdyn_eval
mkdir -p "$DECDIR" "$LOG"

CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
echo "=== pixel $NAME START $(date) GPU$GPU ckpt=$CKPT ==="
[ -z "$CKPT" ] && { echo "NO CKPT for $NAME"; exit 1; }

# ---- decoder (train proj-space universal decoder unless one is provided) ----
if [ -z "$DEC" ]; then
  DEC=$DECDIR/udecoder_${NAME}.pt
  cd "$LEWM/decode_viz" || exit 2
  CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" -u train_universal_decoder.py --domain ${DOM:-uniform_motion} \
      --ckpt "$CKPT" --emb-source proj --epochs 40 --tag ${NAME} --out "$DECDIR" \
      > "$LOG/decoder_${NAME}.log" 2>&1
  echo "=== decoder done ec=$? $(date) ==="
fi
[ ! -f "$DEC" ] && { echo "NO DECODER $DEC"; exit 1; }

# ---- pixel rollout by horizon ----
cd "$ROOT" || exit 2
CUDA_VISIBLE_DEVICES=$GPU STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
  "$LEWM/.venv/bin/python" phyworld/scripts/rollout_pixel_eval.py --domain ${DOM:-uniform_motion} \
    --ckpt "$CKPT" --decoder "$DEC" --tag "$NAME" --max-trajs 500 \
    --out "$LOG/pxroll_${NAME}.json" > "$LOG/pixel_${NAME}.log" 2>&1
echo "=== pixel rollout done ec=$? $(date) -> $LOG/pxroll_${NAME}.json ==="
