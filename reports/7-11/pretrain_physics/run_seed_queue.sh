#!/bin/bash
# Dispatcher: 12 seed-replication runs for paper Table 2 (tab:scratch).
#   3 domains x {phys on, phys off} x 2 extra seeds {1234, 42}
# combined with the existing seed-3072 runs, this makes every cell 3-seed.
# 8 lanes, one per GPU; parabola (120 ep) gets its own lane, the 60-epoch
# domains are paired two-per-lane.
set -u
cd "$(dirname "$0")"
S=./run_pretrain_seed.sh
OUT=/data1/likun-share/junjxu/runs/pretrain_physics
mkdir -p "$OUT"
STATUS=$OUT/SEED_QUEUE_STATUS.txt
: > "$STATUS"

lane () {  # lane <gpu> <spec...>   spec = DOM:PHYS:SEED
  local gpu=$1; shift
  for spec in "$@"; do
    IFS=: read -r dom phys seed <<< "$spec"
    echo "[$(date +%H:%M:%S)] GPU$gpu START $dom phys=$phys seed=$seed" >> "$STATUS"
    SEED=$seed PHYS=$phys bash "$S" "$gpu" "$dom" >> "$OUT/queue_gpu${gpu}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $dom phys=$phys seed=$seed ec=$?" >> "$STATUS"
  done
}

# long pole first, alone on its lane (parabola = 120 epochs)
lane 1 parabola:off:1234 &
lane 3 parabola:on:1234  &
lane 4 parabola:off:42   &
lane 5 parabola:on:42    &
# 60-epoch domains, two per lane
lane 6 uniform_motion:off:1234 collision:off:1234 &
lane 7 uniform_motion:on:1234  collision:on:1234  &
lane 0 uniform_motion:off:42   collision:off:42   &
lane 2 uniform_motion:on:42    collision:on:42    &
wait
echo "[$(date +%H:%M:%S)] ALL LANES FINISHED" >> "$STATUS"
