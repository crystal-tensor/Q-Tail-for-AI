#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
OUT="/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

while true; do
  if /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
    "$PYTHON" "$ROOT/tools/qtail_openx_expansion_status.py" \
      --root "$OUT" \
      --out "$OUT/status.json" \
      >> "$ROOT/.tmp/qtail-openx-1t-status-loop.log" 2>&1 || true
  fi
  sleep 15
done
