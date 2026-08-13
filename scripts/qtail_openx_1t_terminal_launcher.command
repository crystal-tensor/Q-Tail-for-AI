#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
OUT="/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion"
PIPELINE="$ROOT/scripts/qtail_openx_1t_pipeline.sh"
STATUS_LOOP="$ROOT/scripts/qtail_openx_1t_status_loop.sh"
PREWARM_LOOP="$ROOT/scripts/qtail_openx_1t_prewarm_loop.sh"
MARKER_TOOL="$ROOT/tools/qtail_openx_stage_marker.py"
PAGE_QA_TOOL="$ROOT/tools/qtail_openx_final_page_qa.py"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOCAL_LOG="$ROOT/.tmp/qtail-openx-1t-terminal-launcher.log"

mkdir -p "$ROOT/.tmp"
if ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
  printf '[%s] ORICO unavailable; retry on next launch interval\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOCAL_LOG"
  exit 0
fi

probe="$OUT/.terminal-write-probe.$$"
if ! { printf '%s\n' "$$" > "$probe" && /bin/rm "$probe"; } 2>/dev/null; then
  printf '[%s] Terminal lacks ORICO write access\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOCAL_LOG"
  exit 1
fi

if ! pgrep -f -x "/bin/zsh $STATUS_LOOP" >/dev/null 2>&1; then
  /usr/bin/screen -S qtail-openx-1t-status -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -dmS qtail-openx-1t-status \
    /bin/zsh -lc "exec /bin/zsh '$STATUS_LOOP'"
fi

if ! "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage training >/dev/null 2>&1 \
  && ! pgrep -f -x "/bin/zsh $PREWARM_LOOP" >/dev/null 2>&1; then
  /usr/bin/screen -S qtail-openx-1t-prewarm -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -dmS qtail-openx-1t-prewarm \
    /bin/zsh -lc "exec /bin/zsh '$PREWARM_LOOP'"
fi

pipeline_complete=false
if "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage synthesis \
  >/dev/null 2>&1 \
  && "$PYTHON" "$PAGE_QA_TOOL" validate --root "$OUT" --workspace "$ROOT" \
    >/dev/null 2>&1; then
  pipeline_complete=true
fi

if [[ "$pipeline_complete" == false ]] \
  && ! pgrep -f -x "/bin/zsh $PIPELINE" >/dev/null 2>&1; then
  /usr/bin/screen -S qtail-openx-1t-pipeline -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -dmS qtail-openx-1t-pipeline \
    /bin/zsh -lc "exec /bin/zsh '$PIPELINE'"
fi

printf '[%s] Open X 1 TiB workers supervised\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOCAL_LOG"
exit 0
