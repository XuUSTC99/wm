#!/bin/bash
# Archive the dinowm cross-model evidence into raw_data/ and push (2026-07-16).
# WAITS for all dinowm queues to drain first — pushing a half-finished matrix would
# put numbers in the repo that don't match the final report.
set -u
trap '' SIGUSR1 SIGUSR2 SIGURG SIGHUP
ROOT=/home/likun-share/junjxu/wm
SRC=/data1/likun-share/junjxu/runs/dinowm
DST=$ROOT/raw_data/runs/dinowm
LOGF=$SRC/archive.log
echo "=== archive_dinowm WAITING for queues to drain ($(date)) ===" > "$LOGF"
for i in $(seq 1 400); do
  n=$(ps -eo cmd 2>/dev/null | grep '[t]rain.py' | grep -c dinov2)
  m=$(ps -eo cmd 2>/dev/null | grep -c '[r]ollout_eval_id1k\|[r]ollout_eval_physionpp')
  [ "$n" -eq 0 ] && [ "$m" -eq 0 ] && { echo "[$(date +%H:%M)] drained, archiving" >> "$LOGF"; break; }
  sleep 60
done
sleep 30   # let the last eval flush its log

mkdir -p "$DST"
cp -f "$SRC"/train_dinowm_*.log "$DST"/ 2>/dev/null
cp -f "$SRC"/rollout_dinowm_*.log "$DST"/ 2>/dev/null
cp -f "$SRC"/probe190_*.log "$DST"/ 2>/dev/null
cp -f "$SRC"/queue*.log "$DST"/ 2>/dev/null
cp -f "$SRC"/smoke.log "$SRC"/eval_smoke.log "$DST"/ 2>/dev/null
echo "[$(date +%H:%M)] copied $(ls "$DST" | wc -l) files ($(du -sh "$DST" | cut -f1))" >> "$LOGF"

# regenerate MANIFEST.tsv in the existing format: path \t bytes \t mtime
cd "$ROOT/raw_data" || exit 1
{
  printf "path\tbytes\tmtime\n"
  find . -type f ! -name MANIFEST.tsv | sed 's|^\./||' | sort | while read -r f; do
    printf "%s\t%s\t%s\n" "$f" "$(stat -c%s "$f")" "$(date -d "@$(stat -c%Y "$f")" '+%Y-%m-%d %H:%M')"
  done
} > MANIFEST.tsv
echo "[$(date +%H:%M)] MANIFEST rebuilt: $(( $(wc -l < MANIFEST.tsv) - 1 )) files" >> "$LOGF"

cd "$ROOT" || exit 1
git add raw_data/runs/dinowm raw_data/MANIFEST.tsv reports/7-11/aaai_paper/ le-wm/train.py 2>/dev/null
git commit -q -m "$(cat <<'MSG'
raw_data + reports: dinowm cross-model evidence (2nd JEPA instance)

Frozen DINOv2-small + trainable projector/predictor (DINO-WM style), same
losses/eval as LeWM -> controlled cross-backbone ablation of the paper's claims.

Headline reproduces: free-rollout >> teacher-forcing on 3 synthetic domains
(1.39-1.69x) and real Physion++ (3.99x), 3 seeds, disjoint error bars.
Injection arms land inside the baseline error bar; dyn/plain-slot arms hurt;
probe-190 shows the black-box bypass is intact (rho 0.973).

Includes the pos_weight anomaly and its controls (shuffled-slot / weak-pin):
the uniform-only gain at pw300 comes with a 30% ID regression and does not
transfer to collision/parabola (1.11x/1.14x worse).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)" >> "$LOGF" 2>&1
git push origin main >> "$LOGF" 2>&1
echo "[$(date +%H:%M)] push ec=$? ; HEAD=$(git rev-parse --short HEAD)" >> "$LOGF"
echo "=== archive_dinowm DONE $(date) ===" >> "$LOGF"
