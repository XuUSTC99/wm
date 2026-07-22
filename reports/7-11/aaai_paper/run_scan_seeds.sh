#!/bin/bash
# ============================================================================
# Fig-2 (30-cell injection scan) multi-seed completion — 2026-07-22
#
# WHY: 26 of the 30 cells in fig16_scan_paper.pdf are single-seed (3072). Two
# cells that LOOKED like gains were already overturned once the extra seeds
# arrived, so single-seed cells are the figure's weakest point. This queue
# adds seeds 1234 + 42 to every cell that lacks them.
#
# CONFIG PROVENANCE: every EXTRA below was read back from the seed-3072 run's
# own saved config.yaml under $STABLEWM_HOME/<name>/config.yaml (audited
# 2026-07-22), NOT reconstructed from memory. Two audit findings:
#   (a) [dyn] strict a=g and [free] grounded are the SAME config (diff of the
#       two config.yaml files differs only in output_model_name/subdir). One
#       job therefore serves both heatmap rows -- see AG below.
#   (b) the [cons] row used a DIFFERENT variant per domain: uniform=pw30,
#       parabola=accel_weight1.0 + batch 32, collision=plain. This queue
#       standardises all three on the collision (plain) config, so the row
#       becomes comparable; uniform/parabola therefore need all 3 seeds.
#
# 48 jobs, 24 concurrent (3 per GPU x 8 GPUs), num_workers=4 to stay under
# 112 cores. Non-final checkpoints are deleted right after training (20 ckpts
# x 70M x 48 runs would be 67G of churn we do not need).
# ============================================================================
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm; LEWM=$ROOT/le-wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
INIT_CKPT=$STABLEWM_HOME/lewm_paper_pusht/weights.pt
LOG=/data1/likun-share/junjxu/runs/scan_seeds; QLOG=$LOG/queue.log; mkdir -p "$LOG"

UM=phyworld_uniform_motion_id1k; PAR=phyworld_parabola_id1k; COL=phyworld_collision_id1k_st
SLOT="loss.structured.weight=1.0 loss.structured.target=proprio"
PROBE="loss.probe.weight=1.0 loss.probe.target=[proprio,action] loss.probe.frames=2"
DYNC="dynamics.enabled=true dynamics.accel_form=const dynamics.learnable_accel=true"
DYNM="dynamics.enabled=true dynamics.accel_form=mlp dynamics.learnable_accel=true"
CONS="loss.consistency.weight=1.0 loss.consistency.accel_weight=0.0"

# NAME|DATA|DOM|EXTRA   (all pusht-init, 20 epochs, free-rollout np8 bs64)
Q=(
  # --- arm 1 [slot] structpos : all three domains single-seed ---
  "sc_um_structpos_s1234|$UM|uniform_motion|$SLOT seed=1234"
  "sc_um_structpos_s42|$UM|uniform_motion|$SLOT seed=42"
  "sc_par_structpos_s1234|$PAR|parabola|$SLOT seed=1234"
  "sc_par_structpos_s42|$PAR|parabola|$SLOT seed=42"
  "sc_col_structpos_s1234|$COL|collision|$SLOT seed=1234"
  "sc_col_structpos_s42|$COL|collision|$SLOT seed=42"
  # --- arm 2 [slot] +reweight w=30 : uniform already 3-seed ---
  "sc_par_reweight_s1234|$PAR|parabola|$SLOT loss.pos_weight=30 seed=1234"
  "sc_par_reweight_s42|$PAR|parabola|$SLOT loss.pos_weight=30 seed=42"
  "sc_col_reweight_s1234|$COL|collision|$SLOT loss.pos_weight=30 seed=1234"
  "sc_col_reweight_s42|$COL|collision|$SLOT loss.pos_weight=30 seed=42"
  # --- arm 3 [slot] +velocity : parabola already 3-seed (the one gain cell) ---
  "sc_um_velocity_s1234|$UM|uniform_motion|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30 seed=1234"
  "sc_um_velocity_s42|$UM|uniform_motion|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30 seed=42"
  "sc_col_velocity_s1234|$COL|collision|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30 seed=1234"
  "sc_col_velocity_s42|$COL|collision|loss.structured.weight=1.0 loss.structured.target=[proprio,action] loss.pos_weight=30 seed=42"
  # --- arm 4 [probe] probe : parabola already 3-seed ---
  "sc_um_probe_s1234|$UM|uniform_motion|$PROBE seed=1234"
  "sc_um_probe_s42|$UM|uniform_motion|$PROBE seed=42"
  "sc_col_probe_s1234|$COL|collision|$PROBE seed=1234"
  "sc_col_probe_s42|$COL|collision|$PROBE seed=42"
  # --- arm 5 [probe] +slot : uniform already 3-seed ---
  "sc_par_probeslot_s1234|$PAR|parabola|$PROBE $SLOT loss.pos_weight=30 seed=1234"
  "sc_par_probeslot_s42|$PAR|parabola|$PROBE $SLOT loss.pos_weight=30 seed=42"
  "sc_col_probeslot_s1234|$COL|collision|$PROBE $SLOT loss.pos_weight=30 seed=1234"
  "sc_col_probeslot_s42|$COL|collision|$PROBE $SLOT loss.pos_weight=30 seed=42"
  # --- arm 6 [dyn] free MLP : all three single-seed ---
  "sc_um_dynmlp_s1234|$UM|uniform_motion|$SLOT $DYNM seed=1234"
  "sc_um_dynmlp_s42|$UM|uniform_motion|$SLOT $DYNM seed=42"
  "sc_par_dynmlp_s1234|$PAR|parabola|$SLOT $DYNM seed=1234"
  "sc_par_dynmlp_s42|$PAR|parabola|$SLOT $DYNM seed=42"
  "sc_col_dynmlp_s1234|$COL|collision|$SLOT $DYNM seed=1234"
  "sc_col_dynmlp_s42|$COL|collision|$SLOT $DYNM seed=42"
  # --- arm 7 [dyn] strict a=g  ==  arm 10 [free] grounded (identical config;
  #     one job feeds BOTH heatmap rows -- audit finding (a) above) ---
  "sc_um_ag_s1234|$UM|uniform_motion|$SLOT $DYNC seed=1234"
  "sc_um_ag_s42|$UM|uniform_motion|$SLOT $DYNC seed=42"
  "sc_par_ag_s1234|$PAR|parabola|$SLOT $DYNC seed=1234"
  "sc_par_ag_s42|$PAR|parabola|$SLOT $DYNC seed=42"
  "sc_col_ag_s1234|$COL|collision|$SLOT $DYNC seed=1234"
  "sc_col_ag_s42|$COL|collision|$SLOT $DYNC seed=42"
  # --- arm 8 [cons] consistency, STANDARDISED (audit finding (b)):
  #     collision's plain config becomes the row's config for all domains.
  #     collision keeps its 3072 run and needs 2 seeds; um/par need all 3. ---
  "sc_um_cons_s3072|$UM|uniform_motion|$SLOT $CONS seed=3072"
  "sc_um_cons_s1234|$UM|uniform_motion|$SLOT $CONS seed=1234"
  "sc_um_cons_s42|$UM|uniform_motion|$SLOT $CONS seed=42"
  "sc_par_cons_s3072|$PAR|parabola|$SLOT $CONS seed=3072"
  "sc_par_cons_s1234|$PAR|parabola|$SLOT $CONS seed=1234"
  "sc_par_cons_s42|$PAR|parabola|$SLOT $CONS seed=42"
  "sc_col_cons_s1234|$COL|collision|$SLOT $CONS seed=1234"
  "sc_col_cons_s42|$COL|collision|$SLOT $CONS seed=42"
  # --- arm 9 [free] label-free : all three single-seed ---
  "sc_um_labelfree_s1234|$UM|uniform_motion|loss.structured.weight=0.0 $DYNC seed=1234"
  "sc_um_labelfree_s42|$UM|uniform_motion|loss.structured.weight=0.0 $DYNC seed=42"
  "sc_par_labelfree_s1234|$PAR|parabola|loss.structured.weight=0.0 $DYNC seed=1234"
  "sc_par_labelfree_s42|$PAR|parabola|loss.structured.weight=0.0 $DYNC seed=42"
  "sc_col_labelfree_s1234|$COL|collision|loss.structured.weight=0.0 $DYNC seed=1234"
  "sc_col_labelfree_s42|$COL|collision|loss.structured.weight=0.0 $DYNC seed=42"
)

run_one() {  # GPU NAME DATA DOM EXTRA
  local g=$1 NAME=$2 DATA=$3 DOM=$4 EXTRA=$5
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  cd "$LEWM" || return 1
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    .venv/bin/python -u train.py data=$DATA \
      output_model_name=$NAME subdir=$NAME wandb.enabled=False tensorboard.enabled=false \
      trainer.max_epochs=20 num_workers=4 $EXTRA +init_from_ckpt=$INIT_CKPT \
      > "$LOG/train_${NAME}.log" 2>&1
  local EC=$?
  [ $EC -ne 0 ] && { echo "  TRAIN FAIL $NAME ec=$EC" >> "$QLOG"; return 1; }
  local CKPT=$(ls -t $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "  NO CKPT $NAME" >> "$QLOG"; return 1; }
  # keep only the checkpoint we evaluate -- disk ran out once before
  ls $STABLEWM_HOME/$NAME/${NAME}_epoch_*_object.ckpt 2>/dev/null | grep -vF "$CKPT" | xargs -r rm -f
  cd "$ROOT" || return 1
  CUDA_VISIBLE_DEVICES=$g STABLEWM_HOME=$STABLEWM_HOME HF_HOME=$HF_HOME \
    "$LEWM/.venv/bin/python" phyworld/scripts/rollout_eval_id1k.py \
      --domain $DOM --ckpt "$CKPT" --tag "$NAME" --max-trajs 500 \
      > "$LOG/rollout_${NAME}.log" 2>&1
  echo "[$(date +%H:%M)] ALL DONE $NAME ec=$?" >> "$QLOG"
}

# 24 lanes; lane L owns GPU (L%8) and runs its jobs back-to-back.
LANES=24
echo "=== scan-seeds queue START $(date), ${#Q[@]} jobs, $LANES lanes ===" > "$QLOG"
for ((L=0; L<LANES; L++)); do
  (
    for ((i=L; i<${#Q[@]}; i+=LANES)); do
      IFS='|' read NAME DATA DOM EXTRA <<< "${Q[$i]}"
      if grep -qE "both-OOD|r/m-OOD" "$LOG/rollout_${NAME}.log" 2>/dev/null; then
        echo "SKIP $NAME (already done)" >> "$QLOG"; continue
      fi
      run_one $((L % 8)) "$NAME" "$DATA" "$DOM" "$EXTRA"
    done
  ) &
  sleep 4   # stagger so cards claim memory before the next lane probes
done
wait
echo "=== scan-seeds queue ALL FINISHED $(date) ===" >> "$QLOG"
