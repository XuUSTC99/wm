#!/bin/bash
# Black-box dose-response: how many black-box dimensions must move before the
# rollout follows a donor as far as moving the 2-d slot does?  (2026-07-22)
#
# WHY THIS EXISTS. Proposition (b) of the mechanism -- "the black box bears the
# load" -- has been untestable so far. Every black-box intervention we had
# (steering g_bb, the donor-bb patch) moves the channel by a min-norm offset
# along the probe direction, which systematically under-moves a redundant,
# distributed code. The proof that this is a method artifact and not a fact
# about the channel: in a BASELINE model, where the black box is the only route
# position can take, the min-norm patch still yields a follow-fraction of 0.089,
# while replacing the black box outright yields 0.953. Same model, same channel,
# same donor -- an order of magnitude apart.
#
# The dose ladder (2/10/40/100/all dims, fixed subsets) turns that failure into
# a measurement. Smoke test on uniform baseline:
#     k=2 -> 0.058,  k=10 -> 0.099,  k=40 -> 0.701,  k=100 -> 0.872,  k=190 -> 0.953
# so the curve is steep and k=190 is a working positive control.
#
# This is NOT the dose-response already in the paper (fig8). That one sweeps the
# TRAINING loss weight w from 1 to 300 and asks whether a stronger injection
# helps. This one is inference-time on a trained model and asks how the two
# channels divide the load. Different axis, different question.
#
# 36 runs, all inference-only, ~0.7GB each -- everything launches at once.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/causal_route/bbdose
QLOG=$LOG/queue.log
mkdir -p "$LOG"
JOB=0

launch() {  # NAME DOMAIN CKPT_DIR [extra flags]
  local NAME=$1 DOM=$2 CK=$3; shift 3
  local CKPT=$(ls $S/$CK/${CK}_epoch_20_object.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "MISSING ckpt $CK" >> "$QLOG"; return; fi
  if grep -q "\[done\]" "$LOG/${NAME}.log" 2>/dev/null; then
    echo "SKIP $NAME" >> "$QLOG"; return; fi
  local g=$((JOB % 8)); JOB=$((JOB + 1))
  echo "[$(date +%H:%M)] START $NAME on GPU$g" >> "$QLOG"
  (
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/le-wm/.venv/bin/python" "$ROOT/phyworld/scripts/causal_route_eval.py" \
      --domain "$DOM" --mode patch --max-trajs 150 --ckpt "$CKPT" --tag "$NAME" "$@" \
      > "$LOG/${NAME}.log" 2>&1
    echo "[$(date +%H:%M)] DONE  $NAME ec=$?" >> "$QLOG"
  ) &
}

echo "=== bb dose-response START $(date) ===" > "$QLOG"

# ---------------- LeWM ----------------
launch bbdose_uniform_motion_lewm_baseline_s3072  uniform_motion uniform_motion_baseline_fr_id1k
launch bbdose_uniform_motion_lewm_baseline_s1234  uniform_motion uniform_baseline_fr_s1234
launch bbdose_uniform_motion_lewm_baseline_s42    uniform_motion uniform_baseline_fr_s42
launch bbdose_uniform_motion_lewm_structpos_s3072 uniform_motion uniform_motion_structpos_fr_pw30_id1k
launch bbdose_uniform_motion_lewm_structpos_s1234 uniform_motion uniform_structpos_fr_pw30_s1234
launch bbdose_uniform_motion_lewm_structpos_s42   uniform_motion uniform_structpos_fr_pw30_s42

launch bbdose_parabola_lewm_baseline_s3072  parabola parabola_baseline_fr_id1k
launch bbdose_parabola_lewm_baseline_s1234  parabola parabola_baseline_fr_s1234
launch bbdose_parabola_lewm_baseline_s42    parabola parabola_baseline_fr_s42
launch bbdose_parabola_lewm_structpos_s3072 parabola parabola_structpos_fr_pw30_id1k
launch bbdose_parabola_lewm_structpos_s1234 parabola sc_par_reweight_s1234
launch bbdose_parabola_lewm_structpos_s42   parabola sc_par_reweight_s42

launch bbdose_collision_lewm_baseline_s3072  collision collision_baseline_fr_id1k
launch bbdose_collision_lewm_baseline_s1234  collision collision_baseline_fr_s1234
launch bbdose_collision_lewm_baseline_s42    collision collision_baseline_fr_s42
launch bbdose_collision_lewm_structpos_s3072 collision collision_structpos_fr_pw30_id1k
launch bbdose_collision_lewm_structpos_s1234 collision sc_col_reweight_s1234
launch bbdose_collision_lewm_structpos_s42   collision sc_col_reweight_s42

# ---------------- frozen DINOv2 ----------------
launch bbdose_uniform_motion_dino_baseline_s3072  uniform_motion dinowm_um_fr_s3072
launch bbdose_uniform_motion_dino_baseline_s1234  uniform_motion dinowm_um_fr_s1234
launch bbdose_uniform_motion_dino_baseline_s42    uniform_motion dinowm_um_fr_s42
launch bbdose_uniform_motion_dino_structpos_s3072 uniform_motion dinowm_um_structpos_pw30
launch bbdose_uniform_motion_dino_structpos_s1234 uniform_motion dinowm_um_structpos_pw30_s1234
launch bbdose_uniform_motion_dino_structpos_s42   uniform_motion dinowm_um_structpos_pw30_s42

launch bbdose_parabola_dino_baseline_s3072  parabola dinowm_par_fr_s3072
launch bbdose_parabola_dino_baseline_s1234  parabola dinowm_par_fr_s1234
launch bbdose_parabola_dino_baseline_s42    parabola dinowm_par_fr_s42
launch bbdose_parabola_dino_structpos_s3072 parabola dinowm_par_structpos_pw30
launch bbdose_parabola_dino_structpos_s1234 parabola dinowm_par_structpos_pw30_s1234
launch bbdose_parabola_dino_structpos_s42   parabola dinowm_par_structpos_pw30_s42

launch bbdose_collision_dino_baseline_s3072  collision dinowm_col_fr_s3072
launch bbdose_collision_dino_baseline_s1234  collision dinowm_col_fr_s1234
launch bbdose_collision_dino_baseline_s42    collision dinowm_col_fr_s42
launch bbdose_collision_dino_structpos_s3072 collision dinowm_col_structpos_pw30
launch bbdose_collision_dino_structpos_s1234 collision dinowm_col_structpos_pw30_s1234
launch bbdose_collision_dino_structpos_s42   collision dinowm_col_structpos_pw30_s42

wait
echo "=== bb dose-response ALL DONE $(date) ===" >> "$QLOG"
