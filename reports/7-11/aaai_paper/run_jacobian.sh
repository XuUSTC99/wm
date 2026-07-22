#!/bin/bash
# Per-dimension Jacobian of the predicted position, slot dims vs black-box dims.
#
# Every interventional measurement of the black box in this study had to choose
# a direction first -- the probe's pseudo-inverse -- and a redundant distributed
# code barely moves along any single direction. That is why the min-norm patch
# reads 0.089 on a channel that reads 0.961 when replaced outright, and why five
# of six dose-ladder cells failed their controls.
#
# Differentiating chooses no direction and designs no intervention, so it covers
# all 36 cells rather than the one the ladder left standing. Smoke test on
# LeWM/uniform structpos: slot 0.4289 vs black box 0.1245 per dimension (3.44x),
# with the two slot dims ranked 1st and 3rd of 192 -- while the black box holds
# 96.5% of total sensitivity by sheer dimension count. Both channels carry load;
# the slot is the efficient one, the black box the voluminous one.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
export HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
S=$STABLEWM_HOME
LOG=/data1/likun-share/junjxu/runs/causal_route/jacobian
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
      --domain "$DOM" --mode jacobian --max-trajs 150 --ckpt "$CKPT" --tag "$NAME" "$@" \
      > "$LOG/${NAME}.log" 2>&1
    echo "[$(date +%H:%M)] DONE  $NAME ec=$?" >> "$QLOG"
  ) &
}

echo "=== jacobian sweep START $(date) ===" > "$QLOG"

# ---------------- LeWM ----------------
launch jac_uniform_motion_lewm_baseline_s3072  uniform_motion uniform_motion_baseline_fr_id1k
launch jac_uniform_motion_lewm_baseline_s1234  uniform_motion uniform_baseline_fr_s1234
launch jac_uniform_motion_lewm_baseline_s42    uniform_motion uniform_baseline_fr_s42
launch jac_uniform_motion_lewm_structpos_s3072 uniform_motion uniform_motion_structpos_fr_pw30_id1k
launch jac_uniform_motion_lewm_structpos_s1234 uniform_motion uniform_structpos_fr_pw30_s1234
launch jac_uniform_motion_lewm_structpos_s42   uniform_motion uniform_structpos_fr_pw30_s42

launch jac_parabola_lewm_baseline_s3072  parabola parabola_baseline_fr_id1k
launch jac_parabola_lewm_baseline_s1234  parabola parabola_baseline_fr_s1234
launch jac_parabola_lewm_baseline_s42    parabola parabola_baseline_fr_s42
launch jac_parabola_lewm_structpos_s3072 parabola parabola_structpos_fr_pw30_id1k
launch jac_parabola_lewm_structpos_s1234 parabola sc_par_reweight_s1234
launch jac_parabola_lewm_structpos_s42   parabola sc_par_reweight_s42

launch jac_collision_lewm_baseline_s3072  collision collision_baseline_fr_id1k
launch jac_collision_lewm_baseline_s1234  collision collision_baseline_fr_s1234
launch jac_collision_lewm_baseline_s42    collision collision_baseline_fr_s42
launch jac_collision_lewm_structpos_s3072 collision collision_structpos_fr_pw30_id1k
launch jac_collision_lewm_structpos_s1234 collision sc_col_reweight_s1234
launch jac_collision_lewm_structpos_s42   collision sc_col_reweight_s42

# ---------------- frozen DINOv2 ----------------
launch jac_uniform_motion_dino_baseline_s3072  uniform_motion dinowm_um_fr_s3072
launch jac_uniform_motion_dino_baseline_s1234  uniform_motion dinowm_um_fr_s1234
launch jac_uniform_motion_dino_baseline_s42    uniform_motion dinowm_um_fr_s42
launch jac_uniform_motion_dino_structpos_s3072 uniform_motion dinowm_um_structpos_pw30
launch jac_uniform_motion_dino_structpos_s1234 uniform_motion dinowm_um_structpos_pw30_s1234
launch jac_uniform_motion_dino_structpos_s42   uniform_motion dinowm_um_structpos_pw30_s42

launch jac_parabola_dino_baseline_s3072  parabola dinowm_par_fr_s3072
launch jac_parabola_dino_baseline_s1234  parabola dinowm_par_fr_s1234
launch jac_parabola_dino_baseline_s42    parabola dinowm_par_fr_s42
launch jac_parabola_dino_structpos_s3072 parabola dinowm_par_structpos_pw30
launch jac_parabola_dino_structpos_s1234 parabola dinowm_par_structpos_pw30_s1234
launch jac_parabola_dino_structpos_s42   parabola dinowm_par_structpos_pw30_s42

launch jac_collision_dino_baseline_s3072  collision dinowm_col_fr_s3072
launch jac_collision_dino_baseline_s1234  collision dinowm_col_fr_s1234
launch jac_collision_dino_baseline_s42    collision dinowm_col_fr_s42
launch jac_collision_dino_structpos_s3072 collision dinowm_col_structpos_pw30
launch jac_collision_dino_structpos_s1234 collision dinowm_col_structpos_pw30_s1234
launch jac_collision_dino_structpos_s42   collision dinowm_col_structpos_pw30_s42

wait
echo "=== jacobian sweep ALL DONE $(date) ===" >> "$QLOG"
