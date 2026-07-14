#!/bin/bash
# LBR ablation phase 2: cross-domain pos_weight curves, FULL grid matching uniform.
# Jobs: {parabola, collision} x pw{1,3,10,100,300}  (pw30 already exists per domain)
#       = 10 runs, two waves (8 GPUs, one of ours per card).
# Chained: waits for all 8 phase-1 runs, launches wave 1 (8 jobs), then waits for
# >=2 wave-1 finishes and launches the remaining 2 on the freest GPUs.
# setsid-detached; survives session teardown.
set -u
LOG=/data1/likun-share/junjxu/runs/structdyn_eval
S=/home/likun-share/junjxu/wm/reports/6-24/run_structdyn.sh
PD=phyworld_parabola_id1k; CD=phyworld_collision_id1k_st
P1="orch_pw3 orch_pw300 orch_pw1s1234 orch_pw1s42 orch_pw10s1234 orch_pw10s42 orch_pw100s1234 orch_pw100s42"

wait_done () {  # wait_done "<orch names>" <count> <max_iters>
  local files=$1 need=$2 iters=$3 n=0 i
  for i in $(seq 1 $iters); do
    n=0; for f in $files; do grep -qi "ALL DONE" "$LOG/$f.log" 2>/dev/null && n=$((n+1)); done
    [ "$n" -ge "$need" ] && return 0
    sleep 120
  done
  echo "wait_done timeout ($n/$need)"; return 1
}

launch () {  # launch <gpu> <dom> <data> <pw>
  local g=$1 DOM=$2 DATA=$3 pw=$4
  local NAME=${DOM}_structpos_fr_pw${pw}_id1k
  DATA=$DATA DOM=$DOM SW=1.0 DYN=false setsid bash "$S" $g "$NAME" "loss.pos_weight=$pw" \
    > "$LOG/orch_p2_${DOM}_pw${pw}.log" 2>&1 </dev/null &
  echo "  launched $NAME on GPU$g $(date +%H:%M)"
}

echo "=== phase2 launcher START $(date): waiting for phase 1 (8/8) ==="
wait_done "$P1" 8 240

echo "=== wave 1: 8 jobs $(date) ==="
g=0
W1=""
for dom_meta in "parabola:$PD" "collision:$CD"; do
  IFS=: read DOM DATA <<< "$dom_meta"
  for pw in 1 10 100 300; do
    launch $g "$DOM" "$DATA" $pw
    W1="$W1 orch_p2_${DOM}_pw${pw}"
    g=$(( g + 1 ))
  done
done

echo "=== waiting for >=2 wave-1 finishes to free slots ==="
wait_done "$W1" 2 240

# pick the two freest GPUs for the pw3 stragglers
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -2 | cut -d',' -f1)
echo "=== wave 2: pw3 x2 on GPUs ${FREE[0]} ${FREE[1]} $(date) ==="
launch "${FREE[0]}" parabola "$PD" 3
launch "${FREE[1]}" collision "$CD" 3
echo "=== phase2 all 10 launched $(date) ==="
