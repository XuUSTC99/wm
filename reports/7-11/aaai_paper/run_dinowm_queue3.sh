#!/bin/bash
# dinowm queue3 (2026-07-16 overnight): make cross-model scan MORE solid than LeWM's.
# +18 injection-arm seeds (core 3 families x 3 domains x s1234/s42) -> 3-seed injection scan
# +4 C2 horizon-matching arms (col np16/np20 expect help; um/par np16 expect hurt = double dissociation)
# +1 LBR pw100 on uniform (mechanism curve cross-model)
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
LOG=/data1/likun-share/junjxu/runs/dinowm
QLOG=$LOG/queue3.log
mkdir -p "$LOG"
UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
CONS="$SLOT loss.consistency.weight=1.0"
Q=(
  "dinowm_um_structpos_pw30_s1234|$UM|uniform_motion|$SLOT loss.pos_weight=30 seed=1234"
  "dinowm_um_structpos_pw30_s42|$UM|uniform_motion|$SLOT loss.pos_weight=30 seed=42"
  "dinowm_par_structpos_pw30_s1234|$PAR|parabola|$SLOT loss.pos_weight=30 seed=1234"
  "dinowm_par_structpos_pw30_s42|$PAR|parabola|$SLOT loss.pos_weight=30 seed=42"
  "dinowm_col_structpos_pw30_s1234|$COL|collision|$SLOT loss.pos_weight=30 seed=1234"
  "dinowm_col_structpos_pw30_s42|$COL|collision|$SLOT loss.pos_weight=30 seed=42"
  "dinowm_um_probeF2_s1234|$UM|uniform_motion|$PROBE seed=1234"
  "dinowm_um_probeF2_s42|$UM|uniform_motion|$PROBE seed=42"
  "dinowm_par_probeF2_s1234|$PAR|parabola|$PROBE seed=1234"
  "dinowm_par_probeF2_s42|$PAR|parabola|$PROBE seed=42"
  "dinowm_col_probeF2_s1234|$COL|collision|$PROBE seed=1234"
  "dinowm_col_probeF2_s42|$COL|collision|$PROBE seed=42"
  "dinowm_um_cons_s1234|$UM|uniform_motion|$CONS seed=1234"
  "dinowm_um_cons_s42|$UM|uniform_motion|$CONS seed=42"
  "dinowm_par_cons_s1234|$PAR|parabola|$CONS seed=1234"
  "dinowm_par_cons_s42|$PAR|parabola|$CONS seed=42"
  "dinowm_col_cons_s1234|$COL|collision|$CONS seed=1234"
  "dinowm_col_cons_s42|$COL|collision|$CONS seed=42"
  "dinowm_col_np20|$COL|collision|wm.num_preds=20 loader.batch_size=32"
  "dinowm_col_np16|$COL|collision|wm.num_preds=16 loader.batch_size=48"
  "dinowm_um_np16|$UM|uniform_motion|wm.num_preds=16 loader.batch_size=48"
  "dinowm_par_np16|$PAR|parabola|wm.num_preds=16 loader.batch_size=48"
  "dinowm_um_structpos_pw100|$UM|uniform_motion|$SLOT loss.pos_weight=100"
)
run_one() {
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      +encoder_type=dinov2 +freeze_encoder=true \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=2 loss.probe.weight=0.0 $EXTRA \
      > "$LOG/train_${NAME}.log" 2>&1 || { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}
echo "=== dinowm queue3 START $(date), ${#Q[@]} jobs ===" > "$QLOG"
i=0
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM EXTRA <<< "$item"
  grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null && { i=$((i+1)); continue; }
  g=$((i % 8))
  run_one "$g" "$NAME" "$DATA" "$DOM" "$EXTRA" &
  i=$((i+1)); sleep 3
done
wait
echo "=== dinowm queue3 ALL FINISHED $(date) ===" >> "$QLOG"
