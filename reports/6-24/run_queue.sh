#!/bin/bash
# Autonomous experiment queue: wait for a genuinely-free GPU (util<15% AND no
# other-user compute process), then launch the next optimization experiment.
# Detached (setsid) so it survives and needs no babysitting. Good cluster citizen:
# never co-locates onto another user's active GPU.
# See reports/6-24/final/optimization_plan.md for the rationale.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ME=likun-share
ROOT=/home/likun-share/junjxu/wm
S=$ROOT/reports/6-24/run_structdyn.sh
LOG=/data1/likun-share/junjxu/runs/structdyn_eval
QLOG=$LOG/opt_queue.log
mkdir -p "$LOG"

# each item: NAME|DATA|DOM|SW|DYN|EXTRA
Q=(
  "collision_structpos_cons1.0_id1k|phyworld_collision_id1k_st|collision|1.0|false|loss.consistency.weight=1.0"
  "collision_baseline_fr_np16_id1k|phyworld_collision_id1k_st|collision|0.0|false|wm.num_preds=16 loader.batch_size=48"
  "uniform_motion_structcv_fr_pw100_np16_id1k|phyworld_uniform_motion_id1k|uniform_motion|1.0|true|dynamics.learnable_accel=false loss.pos_weight=100 wm.num_preds=16 loader.batch_size=48"
  "parabola_structdyn_areg_fr_pw30_np16_id1k|phyworld_parabola_id1k|parabola|1.0|true|dynamics.accel_reg=1.0 loss.pos_weight=30 wm.num_preds=16 loader.batch_size=48"
)

free_gpu() {
  # A GPU counts as free only if it has ZERO compute processes (from anyone) AND
  # >=30GB free memory. util is unreliable (a training GPU idles between steps),
  # so we key off process presence, never co-locating onto anyone's active job.
  for g in 0 1 2 3 4 5 6 7; do
    np=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$np" -ne 0 ] && continue
    mfree=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    [ "${mfree:-0}" -ge 30000 ] && { echo $g; return; }
  done
  echo ""
}

echo "=== opt-queue START $(date), ${#Q[@]} exps ===" > "$QLOG"
for item in "${Q[@]}"; do
  IFS='|' read NAME DATA DOM SW DYN EXTRA <<< "$item"
  g=""
  for w in $(seq 1 360); do            # wait up to ~12h for a slot
    g=$(free_gpu); [ -n "$g" ] && break
    sleep 120
  done
  if [ -z "$g" ]; then echo "[$(date)] no free GPU -> skip $NAME" >> "$QLOG"; continue; fi
  echo "[$(date)] launch $NAME on GPU$g (DATA=$DATA DOM=$DOM SW=$SW DYN=$DYN EXTRA=$EXTRA)" >> "$QLOG"
  DATA=$DATA DOM=$DOM SW=$SW DYN=$DYN setsid bash "$S" "$g" "$NAME" "$EXTRA" \
    > "$LOG/orch_${NAME}.log" 2>&1 </dev/null &
  sleep 90                             # let it seize the GPU before scanning again
done
echo "=== opt-queue ALL LAUNCHED $(date) ===" >> "$QLOG"
