#!/bin/bash
# The dose ladder again, but with --clamp ctx-only  (2026-07-22)
#
# The every-step version (run_bbdose.sh) re-applies the patch at every rollout
# step, so it measures steady-state sensitivity: hold a channel at the donor's
# value and see where the readout ends up. That answers "is this channel read".
#
# ctx-only patches the context once and then lets the model evolve freely. It
# answers the strictly harder question: does the perturbation get CARRIED --
# does the predictor propagate a counterfactual state forward on its own? For a
# claim about what the rollout dynamics run through, that is the closer match,
# and a channel can pass the first test while failing this one.
#
# Same 36 cells. Inference only, ~0.7GB each; the every-step batch is already
# running and the machine has CPU headroom, so these launch alongside it.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/causal_route/bbdose_ctx
QLOG=$LOG/queue.log
mkdir -p "$LOG"
JOB=0

launch() {
  local NAME=$1 DOM=$2 CK=$3; shift 3
  local CKPT=$(ls $S/$CK/${CK}_epoch_20_object.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "MISSING $CK" >> "$QLOG"; return; }
  grep -q "\[done\]" "$LOG/${NAME}.log" 2>/dev/null && { echo "SKIP $NAME" >> "$QLOG"; return; }
  local g=$((JOB % 8)); JOB=$((JOB + 1))
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  (
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/le-wm/.venv/bin/python" "$ROOT/phyworld/scripts/causal_route_eval.py" \
      --domain "$DOM" --mode patch --max-trajs 150 --clamp ctx-only \
      --ckpt "$CKPT" --tag "$NAME" "$@" > "$LOG/${NAME}.log" 2>&1
    echo "[$(date +%H:%M)] DONE  $NAME ec=$?" >> "$QLOG"
  ) &
}

echo "=== bb dose ctx-only START $(date) ===" > "$QLOG"

launch ctx_uniform_motion_lewm_baseline_s3072  uniform_motion uniform_motion_baseline_fr_id1k
launch ctx_uniform_motion_lewm_baseline_s1234  uniform_motion uniform_baseline_fr_s1234
launch ctx_uniform_motion_lewm_baseline_s42    uniform_motion uniform_baseline_fr_s42
launch ctx_uniform_motion_lewm_structpos_s3072 uniform_motion uniform_motion_structpos_fr_pw30_id1k
launch ctx_uniform_motion_lewm_structpos_s1234 uniform_motion uniform_structpos_fr_pw30_s1234
launch ctx_uniform_motion_lewm_structpos_s42   uniform_motion uniform_structpos_fr_pw30_s42

launch ctx_parabola_lewm_baseline_s3072  parabola parabola_baseline_fr_id1k
launch ctx_parabola_lewm_baseline_s1234  parabola parabola_baseline_fr_s1234
launch ctx_parabola_lewm_baseline_s42    parabola parabola_baseline_fr_s42
launch ctx_parabola_lewm_structpos_s3072 parabola parabola_structpos_fr_pw30_id1k
launch ctx_parabola_lewm_structpos_s1234 parabola sc_par_reweight_s1234
launch ctx_parabola_lewm_structpos_s42   parabola sc_par_reweight_s42

launch ctx_collision_lewm_baseline_s3072  collision collision_baseline_fr_id1k
launch ctx_collision_lewm_baseline_s1234  collision collision_baseline_fr_s1234
launch ctx_collision_lewm_baseline_s42    collision collision_baseline_fr_s42
launch ctx_collision_lewm_structpos_s3072 collision collision_structpos_fr_pw30_id1k
launch ctx_collision_lewm_structpos_s1234 collision sc_col_reweight_s1234
launch ctx_collision_lewm_structpos_s42   collision sc_col_reweight_s42

launch ctx_uniform_motion_dino_baseline_s3072  uniform_motion dinowm_um_fr_s3072
launch ctx_uniform_motion_dino_baseline_s1234  uniform_motion dinowm_um_fr_s1234
launch ctx_uniform_motion_dino_baseline_s42    uniform_motion dinowm_um_fr_s42
launch ctx_uniform_motion_dino_structpos_s3072 uniform_motion dinowm_um_structpos_pw30
launch ctx_uniform_motion_dino_structpos_s1234 uniform_motion dinowm_um_structpos_pw30_s1234
launch ctx_uniform_motion_dino_structpos_s42   uniform_motion dinowm_um_structpos_pw30_s42

launch ctx_parabola_dino_baseline_s3072  parabola dinowm_par_fr_s3072
launch ctx_parabola_dino_baseline_s1234  parabola dinowm_par_fr_s1234
launch ctx_parabola_dino_baseline_s42    parabola dinowm_par_fr_s42
launch ctx_parabola_dino_structpos_s3072 parabola dinowm_par_structpos_pw30
launch ctx_parabola_dino_structpos_s1234 parabola dinowm_par_structpos_pw30_s1234
launch ctx_parabola_dino_structpos_s42   parabola dinowm_par_structpos_pw30_s42

launch ctx_collision_dino_baseline_s3072  collision dinowm_col_fr_s3072
launch ctx_collision_dino_baseline_s1234  collision dinowm_col_fr_s1234
launch ctx_collision_dino_baseline_s42    collision dinowm_col_fr_s42
launch ctx_collision_dino_structpos_s3072 collision dinowm_col_structpos_pw30
launch ctx_collision_dino_structpos_s1234 collision dinowm_col_structpos_pw30_s1234
launch ctx_collision_dino_structpos_s42   collision dinowm_col_structpos_pw30_s42

wait
echo "=== bb dose ctx-only ALL DONE $(date) ===" >> "$QLOG"
