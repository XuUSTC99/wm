#!/bin/bash
# AAAI-27 P0 experiment queue (see 04_todo_experiments.md).
# 20 jobs: 12 headline TF/FR seed arms + 2 LBR seed arms + 6 clean from-scratch 2x2 (lr 2e-5, 60ep).
# Own launcher (not run_structdyn.sh) because scratch runs must OMIT +init_from_ckpt.
# Good citizen: only GPUs 0-5, zero-compute-proc + >=30GB free required. Detached via setsid.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT_CKPT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/aaai_p0
QLOG=$LOG/p0_queue.log
mkdir -p "$LOG"

# NAME|DATA|DOM|INIT(pusht/scratch)|EPOCHS|EXTRA
Q=(
  # ---- P0-1 headline seeds: FR arms (defaults: fr=true np8 batch64) ----
  "uniform_baseline_fr_s1234|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|seed=1234"
  "collision_baseline_fr_s1234|phyworld_collision_id1k_st|collision|pusht|20|seed=1234"
  "parabola_baseline_fr_s1234|phyworld_parabola_id1k|parabola|pusht|20|seed=1234"
  "uniform_baseline_fr_s42|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|seed=42"
  "collision_baseline_fr_s42|phyworld_collision_id1k_st|collision|pusht|20|seed=42"
  "parabola_baseline_fr_s42|phyworld_parabola_id1k|parabola|pusht|20|seed=42"
  # ---- P0-1 headline seeds: TF arms (match old tf config: np1 batch128) ----
  "uniform_baseline_tf_s1234|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=1234"
  "collision_baseline_tf_s1234|phyworld_collision_id1k_st|collision|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=1234"
  "parabola_baseline_tf_s1234|phyworld_parabola_id1k|parabola|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=1234"
  "uniform_baseline_tf_s42|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=42"
  "collision_baseline_tf_s42|phyworld_collision_id1k_st|collision|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=42"
  "parabola_baseline_tf_s42|phyworld_parabola_id1k|parabola|pusht|20|wm.free_rollout=false wm.num_preds=1 loader.batch_size=128 seed=42"
  # ---- P0-4 LBR seeds (structpos + pos_weight 30) ----
  "uniform_structpos_fr_pw30_s1234|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|loss.structured.weight=1.0 loss.structured.target=proprio loss.pos_weight=30 seed=1234"
  "uniform_structpos_fr_pw30_s42|phyworld_uniform_motion_id1k|uniform_motion|pusht|20|loss.structured.weight=1.0 loss.structured.target=proprio loss.pos_weight=30 seed=42"
  # ---- P0-3 clean from-scratch 2x2 (lr 2e-5, 60ep; PHYS on = strict PIWM const) ----
  "pp2_par_scratch_off|phyworld_parabola_id1k|parabola|scratch|60|optimizer.lr=2e-5"
  "pp2_par_scratch_on|phyworld_parabola_id1k|parabola|scratch|60|optimizer.lr=2e-5 loss.structured.weight=1.0 loss.structured.target=proprio dynamics.enabled=true dynamics.accel_form=const"
  "pp2_um_scratch_off|phyworld_uniform_motion_id1k|uniform_motion|scratch|60|optimizer.lr=2e-5"
  "pp2_um_scratch_on|phyworld_uniform_motion_id1k|uniform_motion|scratch|60|optimizer.lr=2e-5 loss.structured.weight=1.0 loss.structured.target=proprio dynamics.enabled=true dynamics.accel_form=const"
  "pp2_col_scratch_off|phyworld_collision_id1k_st|collision|scratch|60|optimizer.lr=2e-5"
  "pp2_col_scratch_on|phyworld_collision_id1k_st|collision|scratch|60|optimizer.lr=2e-5 loss.structured.weight=1.0 loss.structured.target=proprio dynamics.enabled=true dynamics.accel_form=const"
)

free_gpu() {  # only 0-5; no SUBSTANTIAL compute proc (>=2GB) + >=30GB free.
  # (idle <2GB CUDA contexts — e.g. another user's stdin-python holding 420MiB on
  # every card — don't count as occupancy; hard zero-proc would deadlock forever)
  for g in 0 1 2 3 4 5; do
    np=$(nvidia-smi -i $g --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>=2000' | grep -c .)
    [ "$np" -ne 0 ] && continue
    mfree=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    [ "${mfree:-0}" -ge 30000 ] && { echo $g; return; }
  done
  echo ""
}

run_one() {  # GPU NAME DATA DOM INIT EPOCHS EXTRA
  local g=$1 NAME=$2 DATA=$3 DOM=$4 INIT=$5 EP=$6 EXTRA=$7
  local IF=""
  [ "$INIT" = "pusht" ] && IF="+init_from_ckpt=$INIT_CKPT"
  echo "[$(date +%H:%M)] START $NAME on GPU$g (init=$INIT ep=$EP extra='$EXTRA')" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=$EP loss.probe.weight=0.0 $EXTRA $IF \
      > "$LOG/train_${NAME}.log" 2>&1
  local EC=$?
  echo "[$(date +%H:%M)] train done $NAME ec=$EC" >> "$QLOG"
  [ $EC -ne 0 ] && { echo "  TRAIN FAIL $NAME" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}

echo "=== P0 queue START $(date), ${#Q[@]} jobs ===" > "$QLOG"
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM INIT EP EXTRA <<< "$item"
  # skip if already evaluated (resume-safe)
  if grep -q "ALL DONE" "$LOG/rollout_${NAME}.log" 2>/dev/null || grep -qE "both-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null; then
    echo "SKIP $NAME (already done)" >> "$QLOG"; continue
  fi
  g=""
  while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && sleep 120; done
  run_one "$g" "$NAME" "$DATA" "$DOM" "$INIT" "$EP" "$EXTRA" &
  sleep 45   # let the job claim GPU memory before probing for the next free GPU
done
wait
echo "=== P0 queue ALL FINISHED $(date) ===" >> "$QLOG"
