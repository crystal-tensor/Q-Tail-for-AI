#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOG="$JOB_ROOT/logs/progress_loop.log"
WEB_SERVICES="$ROOT/scripts/qtail_web_services.sh"

while true; do
  if [ -d "$JOB_ROOT" ]; then
    /bin/zsh "$WEB_SERVICES" >> "$LOG" 2>&1 || true
    "$PYTHON" "$ROOT/tools/qtail_droid_full_progress.py" \
      --job-root "$JOB_ROOT" >> "$LOG" 2>&1 || true
    "$PYTHON" "$ROOT/tools/qtail_verify_droid_timeline.py" \
      --timeline \
      "$JOB_ROOT/results/qtail_droid_full/pipeline_timeline.json" \
      --out \
      "$JOB_ROOT/results/qtail_droid_full/pipeline_timeline_current_verification.json" \
      >> "$LOG" 2>&1 || true
  fi
  sleep 60
done
