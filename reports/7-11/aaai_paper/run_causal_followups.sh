#!/bin/bash
# Three follow-ups to the causal-route interventions (2026-07-22).
#
# These are inference-only evaluations on existing checkpoints and each one
# holds well under 1GB of GPU memory, so every job is launched at once, spread
# round-robin over the cards. Nothing here is memory-bound.
#
# A. structdyn positive control. In that architecture position can only reach
#    the prediction through the slot, so its causal gain is the natural
#    yardstick for g -- it should come out near 1. The existing runs report
#    g=77, and the earlier "steer_small" rerun shows shrinking delta does NOT
#    fix it (additivity already passes at 77, so the response is linear, just
#    mis-scaled). That points at how the steering vector is built rather than
#    at its magnitude: --slot-direct writes pos+delta into the slot instead of
#    going through the probe pseudo-inverse, which is the right construction
#    when the slot is supervised to equal position.
#
# B. LeWM/parabola steering is unstable across seeds (-0.097/+0.897/+2.663)
#    while DINO/parabola is not. Sweep delta magnitude per seed: if the spread
#    collapses at smaller delta, it was an out-of-linear-range artifact; if it
#    persists, the instability is real and goes in the paper as a limitation.
#
# C. subset mode (Test 1 on rolled-out latents) is already complete -- 42 runs
#    under runs/causal_route/. Nothing to launch; see collect_causal_summary.py.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/causal_route/followup
QLOG=$LOG/queue.log
mkdir -p "$LOG"

JOB=0

run() {  # NAME DOMAIN CKPT EXTRA... -- launched in background, one card each
  local NAME=$1 DOM=$2 CKPT=$3; shift 3
  if grep -q "STEERING" "$LOG/${NAME}.log" 2>/dev/null; then
    echo "SKIP $NAME (done)" >> "$QLOG"; return
  fi
  local g=$((JOB % 8)); JOB=$((JOB + 1))
  echo "[$(date +%H:%M)] START $NAME on GPU$g  ($*)" >> "$QLOG"
  (
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/le-wm/.venv/bin/python" "$ROOT/phyworld/scripts/causal_route_eval.py" \
      --domain "$DOM" --mode steer --max-trajs 150 --ckpt "$CKPT" --tag "$NAME" "$@" \
      > "$LOG/${NAME}.log" 2>&1
    echo "[$(date +%H:%M)] DONE  $NAME ec=$?" >> "$QLOG"
  ) &
}

echo "=== causal follow-ups START $(date) ===" > "$QLOG"

# ---- A. structdyn yardstick: pseudo-inverse vs direct slot write ----------
for d in uniform_motion parabola collision; do
  case $d in
    uniform_motion) CK=$S/uniform_motion_structdyn_id1k/uniform_motion_structdyn_id1k_epoch_20_object.ckpt ;;
    parabola)       CK=$S/parabola_structdyn_fr_id1k/parabola_structdyn_fr_id1k_epoch_20_object.ckpt ;;
    collision)      CK=$S/collision_structdyn_fr_id1k/collision_structdyn_fr_id1k_epoch_20_object.ckpt ;;
  esac
  # direct slot write, small delta -- the construction we expect to give g~1
  run "A_${d}_structdyn_direct_d02" "$d" "$CK" --slot-direct --deltas -0.2 -0.1 0.1 0.2
  # same construction, default delta: separates "construction" from "magnitude"
  run "A_${d}_structdyn_direct_d10" "$d" "$CK" --slot-direct --deltas -1.0 -0.5 0.5 1.0
  # pseudo-inverse at small delta: the missing corner of that 2x2. Without it,
  # a fix at (direct, small) cannot be attributed to the construction rather
  # than to the magnitude.
  run "A_${d}_structdyn_pinv_d02" "$d" "$CK" --deltas -0.2 -0.1 0.1 0.2
done

# ---- B. parabola delta sensitivity, per seed ------------------------------
for s in 3072 1234 42; do
  case $s in
    3072) CK=$S/parabola_structpos_fr_pw30_id1k/parabola_structpos_fr_pw30_id1k_epoch_20_object.ckpt ;;
    1234) CK=$S/sc_par_reweight_s1234/sc_par_reweight_s1234_epoch_20_object.ckpt ;;
    42)   CK=$S/sc_par_reweight_s42/sc_par_reweight_s42_epoch_20_object.ckpt ;;
  esac
  [ -f "$CK" ] || { echo "MISSING ckpt for seed $s: $CK" >> "$QLOG"; continue; }
  run "B_parabola_structpos_s${s}_d005" parabola "$CK" --slot-direct --deltas -0.05 -0.025 0.025 0.05
  run "B_parabola_structpos_s${s}_d02"  parabola "$CK" --slot-direct --deltas -0.2 -0.1 0.1 0.2
  run "B_parabola_structpos_s${s}_d05"  parabola "$CK" --slot-direct --deltas -0.5 -0.25 0.25 0.5
  run "B_parabola_structpos_s${s}_d10"  parabola "$CK" --slot-direct --deltas -1.0 -0.5 0.5 1.0
done

# ---- B2. the same delta ladder on uniform, as the stable-domain reference --
# Without it, a delta-dependence on parabola cannot be told apart from a
# delta-dependence the method has everywhere.
for s in 3072 1234 42; do
  case $s in
    3072) CK=$S/uniform_motion_structpos_fr_pw30_id1k/uniform_motion_structpos_fr_pw30_id1k_epoch_20_object.ckpt ;;
    1234) CK=$S/uniform_structpos_fr_pw30_s1234/uniform_structpos_fr_pw30_s1234_epoch_20_object.ckpt ;;
    42)   CK=$S/uniform_structpos_fr_pw30_s42/uniform_structpos_fr_pw30_s42_epoch_20_object.ckpt ;;
  esac
  [ -f "$CK" ] || { echo "MISSING ckpt for uniform seed $s: $CK" >> "$QLOG"; continue; }
  run "B2_uniform_structpos_s${s}_d005" uniform_motion "$CK" --slot-direct --deltas -0.05 -0.025 0.025 0.05
  run "B2_uniform_structpos_s${s}_d10"  uniform_motion "$CK" --slot-direct --deltas -1.0 -0.5 0.5 1.0
done

wait
echo "=== causal follow-ups ALL DONE $(date) ===" >> "$QLOG"
