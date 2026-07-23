#!/bin/bash
# Variance-matched control for the amnesic-projection limitation.
#
# The appendix concedes that the amnesic control is "matched in rank but not in
# variance". This run quantifies how large that mismatch is. The smoke cell says
# it is not small: INLP removes rank 154 of 190 while taking only 17.4% of the
# black box's variance, whereas a rank-matched random projection takes 81.9% --
# 4.7x more. If that holds across cells, "removing position damages the rollout
# less than removing the same number of random dimensions" is largely a statement
# about how much variance each ablation destroyed, not about which information it
# removed, and the control cannot support either reading.
#
# The variance-matched arm searches random rank-matched subspaces for the one
# whose removed variance is closest to INLP's, and reports what it achieves --
# at this rank the geometry caps it near 77%, so the honest finding may be that
# no valid same-rank control exists rather than that we found a better one.
set -u
cd /home/likun-share/junjxu/wm || exit 2
S=/data1/likun-share/junjxu/.stable_worldmodel
export STABLEWM_HOME=$S HF_HOME=/data1/likun-share/junjxu/.cache_huggingface
O=/data1/likun-share/junjxu/runs/causal_route/vm; mkdir -p "$O"
Q="$O/queue.log"; : > "$Q"

# domain|tag|run-dir-name
CELLS=(
  "uniform_motion|baseline|uniform_motion_baseline_fr_id1k"
  "uniform_motion|structpos|uniform_motion_structpos_fr_pw30_id1k"
  "parabola|baseline|parabola_baseline_fr_id1k"
  "parabola|structpos|parabola_structpos_fr_pw30_id1k"
  "collision|baseline|collision_baseline_fr_id1k"
  "collision|structpos|collision_structpos_fr_pw30_id1k"
)
g=0
for c in "${CELLS[@]}"; do
  IFS='|' read DOM TAG DIR <<< "$c"
  CK=$(ls -t "$S/$DIR"/*_object.ckpt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then echo "SKIP $DOM/$TAG (no ckpt in $DIR)" >> "$Q"; continue; fi
  echo "[$(date +%H:%M)] START $DOM/$TAG on GPU$g" >> "$Q"
  (
    CUDA_VISIBLE_DEVICES=$g le-wm/.venv/bin/python phyworld/scripts/causal_route_eval.py \
      --domain "$DOM" --ckpt "$CK" --tag "$TAG" --mode amnesic --max-trajs 150 \
      > "$O/vm_${DOM}_${TAG}.log" 2>&1
    echo "[$(date +%H:%M)] DONE $DOM/$TAG ec=$?" >> "$Q"
  ) &
  g=$(( (g + 1) % 8 ))
  sleep 4
done
wait
echo "[$(date +%H:%M)] ALL DONE" >> "$Q"
