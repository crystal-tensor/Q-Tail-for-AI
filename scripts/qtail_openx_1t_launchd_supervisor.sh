#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
PIPELINE="$ROOT/scripts/qtail_openx_1t_pipeline.sh"
STATUS_LOOP="$ROOT/scripts/qtail_openx_1t_status_loop.sh"
PREWARM_LOOP="$ROOT/scripts/qtail_openx_1t_prewarm_loop.sh"
TERMINAL_LAUNCHER="$ROOT/scripts/qtail_openx_1t_terminal_launcher.command"
MARKER_TOOL="$ROOT/tools/qtail_openx_stage_marker.py"
PAGE_QA_TOOL="$ROOT/tools/qtail_openx_final_page_qa.py"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
WEB_SERVER="$ROOT/node_modules/.bin/serve"
WEB_URL="http://127.0.0.1:54655/qtail-openx-training"

if ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
  exit 0
fi

# The live ledger is part of the deliverable, so keep its static server under
# the same restart supervision as the downloader and training pipeline.
if ! /usr/bin/curl --silent --show-error --fail --max-time 3 "$WEB_URL" >/dev/null 2>&1; then
  if ! /usr/bin/pgrep -f -x "node $WEB_SERVER --symlinks -l tcp://0.0.0.0:54655" >/dev/null 2>&1; then
    /usr/bin/screen -S qtail-web-54655 -X quit >/dev/null 2>&1 || true
    /usr/bin/screen -dmS qtail-web-54655 /bin/zsh -lc \
      "cd '$ROOT' && exec '$WEB_SERVER' --symlinks -l tcp://0.0.0.0:54655"
  fi
fi

pipeline_running=false
pipeline_complete=false
status_running=false
prewarm_required=true
prewarm_running=false
pgrep -f -x "/bin/zsh $PIPELINE" >/dev/null 2>&1 && pipeline_running=true
if "$PYTHON" "$MARKER_TOOL" validate \
  --root "/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion" \
  --stage synthesis >/dev/null 2>&1 \
  && "$PYTHON" "$PAGE_QA_TOOL" validate \
    --root "/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion" \
    --workspace "$ROOT" >/dev/null 2>&1; then
  pipeline_complete=true
fi
pgrep -f -x "/bin/zsh $STATUS_LOOP" >/dev/null 2>&1 && status_running=true
"$PYTHON" "$MARKER_TOOL" validate \
  --root "/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion" \
  --stage training >/dev/null 2>&1 && prewarm_required=false
pgrep -f -x "/bin/zsh $PREWARM_LOOP" >/dev/null 2>&1 && prewarm_running=true

if [[ ( "$pipeline_complete" == false && "$pipeline_running" == false ) \
  || "$status_running" == false \
  || ( "$prewarm_required" == true && "$prewarm_running" == false ) ]]; then
  # Launch the screen-managed workers directly so periodic recovery never
  # opens a Terminal window on the desktop.
  /bin/zsh "$TERMINAL_LAUNCHER" \
    >> "$ROOT/.tmp/qtail-openx-1t-background-launcher.log" 2>&1
fi
exit 0
