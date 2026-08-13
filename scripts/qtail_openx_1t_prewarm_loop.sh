#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
DATA_ROOT="$JOB_ROOT/data/openx_demo"
OUT="$JOB_ROOT/results/qtail_openx_1t_expansion"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
MARKER_TOOL="$ROOT/tools/qtail_openx_stage_marker.py"
LOCK_DIR="$OUT/prewarm.lock"
LOG="$OUT/prewarm.log"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" || exit 0
fi
echo $$ > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

while ! "$PYTHON" "$MARKER_TOOL" validate \
  --root "$OUT" --stage training >/dev/null 2>&1; do
  if /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO (" \
    && [[ -f "$OUT/openx_1t_checksum_manifest.json" ]] \
    && [[ -f "$OUT/download_checksum_ledger.json" ]]; then
    "$PYTHON" "$ROOT/tools/qtail_openx_feature_prewarm.py" \
      --data-dir "$DATA_ROOT" \
      --manifest "$OUT/openx_1t_checksum_manifest.json" \
      --ledger "$OUT/download_checksum_ledger.json" \
      --cache-dir "$OUT/feature_cache" \
      --status "$OUT/prewarm_status.json" \
      --records-per-shard 4 \
      --process-lock "$OUT/prewarm-process.lock" >> "$LOG" 2>&1 || true
  fi
  sleep 30
done
