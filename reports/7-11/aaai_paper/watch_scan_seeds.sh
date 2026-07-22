#!/bin/bash
# Watchdog for run_scan_seeds.sh — 2026-07-22.
# Waits for the queue to finish, retries anything that failed to produce a
# usable rollout, then writes a parsed summary so the 30-cell table is ready
# to read without digging through 46 logs.
#   Partition per domain matches the figure: uniform/collision = both-OOD,
#   parabola = r/m-OOD (both-OOD there has a degenerate nMSE denominator).
set -u
ROOT=/home/likun-share/junjxu/wm
LOG=/data1/likun-share/junjxu/runs/scan_seeds
QLOG=$LOG/queue.log
WLOG=$LOG/watchdog.log
SUMMARY=$LOG/summary.tsv
QUEUE=$ROOT/reports/7-11/aaai_paper/run_scan_seeds.sh

echo "=== watchdog START $(date) ===" > "$WLOG"

# 1. wait for the queue process to exit
while pgrep -f "run_scan_seeds.sh" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M)] queue process gone" >> "$WLOG"

partition_for() {  # domain -> the row the heatmap reads
  case "$1" in
    parabola) echo "r/m-OOD" ;;
    *)        echo "both-OOD" ;;
  esac
}

ok() {  # NAME DOM -> 0 if the rollout log carries the needed nMSE
  local n=$1 d=$2 p; p=$(partition_for "$d")
  grep -qE "^\s+${p}\s+n=.*nMSE=" "$LOG/rollout_${n}.log" 2>/dev/null
}

# 2. retry pass — re-invoke the queue script; its own skip-guard reruns only
#    the jobs whose rollout log is missing or truncated.
for attempt in 1 2; do
  MISSING=0
  while IFS='|' read -r NAME DATA DOM EXTRA; do
    case "$NAME" in sc_*) ;; *) continue ;; esac
    # the two uniform-structpos jobs were cancelled on purpose: raw_data already
    # holds that cell's seeds as uniform_motion_structpos_fr_pw1_s{1234,42}
    case "$NAME" in sc_um_structpos_s*) continue ;; esac
    ok "$NAME" "$DOM" || { echo "  attempt$attempt MISSING $NAME" >> "$WLOG"; MISSING=$((MISSING+1)); }
  done < <(grep -oE '"sc_[^"]+"' "$QUEUE" | tr -d '"')
  echo "[$(date +%H:%M)] attempt$attempt: $MISSING missing" >> "$WLOG"
  [ "$MISSING" -eq 0 ] && break
  echo "[$(date +%H:%M)] relaunching queue for the $MISSING missing job(s)" >> "$WLOG"
  bash "$QUEUE" >> "$LOG/relaunch_${attempt}.out" 2>&1
done

# 3. parsed summary
{
  printf "run\tdomain\tpartition\tnMSE\tcos\n"
  while IFS='|' read -r NAME DATA DOM EXTRA; do
    case "$NAME" in sc_*) ;; *) continue ;; esac
    case "$NAME" in sc_um_structpos_s*) continue ;; esac
    P=$(partition_for "$DOM")
    LINE=$(grep -E "^\s+${P}\s+n=.*nMSE=" "$LOG/rollout_${NAME}.log" 2>/dev/null | head -1)
    if [ -z "$LINE" ]; then
      printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$NAME" "$DOM" "$P"
    else
      NM=$(echo "$LINE" | grep -oE "nMSE=[0-9.]+" | cut -d= -f2)
      CO=$(echo "$LINE" | grep -oE "cos=[+-][0-9.]+" | cut -d= -f2)
      printf "%s\t%s\t%s\t%s\t%s\n" "$NAME" "$DOM" "$P" "$NM" "$CO"
    fi
  done < <(grep -oE '"sc_[^"]+"' "$QUEUE" | tr -d '"')
} > "$SUMMARY"

echo "=== watchdog DONE $(date) — summary at $SUMMARY ===" >> "$WLOG"
