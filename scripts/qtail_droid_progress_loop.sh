#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOG="$JOB_ROOT/logs/progress_loop.log"
WEB_SERVICES="$ROOT/scripts/qtail_web_services.sh"

archive_local_supervision_logs() {
  local source name destination temporary
  while IFS='|' read -r source name; do
    [ -f "$source" ] || continue
    destination="$JOB_ROOT/logs/$name"
    temporary="$destination.$$.tmp"
    if ! /bin/cp -p "$source" "$temporary" 2>> "$LOG"; then
      /bin/rm -f "$temporary"
      continue
    fi
    /bin/mv -f "$temporary" "$destination" 2>> "$LOG" || \
      /bin/rm -f "$temporary"
  done <<EOF
$ROOT/.tmp/qtail-droid-terminal-launcher.log|qtail_droid_terminal_launcher.log
$ROOT/.tmp/qtail-droid-launchd.err.log|qtail_droid_launchd_stderr.log
$ROOT/.tmp/qtail-droid-launchd.out.log|qtail_droid_launchd_stdout.log
$ROOT/.tmp/qtail-uniclash-guard.err.log|qtail_uniclash_guard_stderr.log
$ROOT/.tmp/qtail-uniclash-guard.out.log|qtail_uniclash_guard_stdout.log
$ROOT/.tmp/qtail-web-services.log|qtail_web_services_local.log
EOF
}

while true; do
  if [ -d "$JOB_ROOT" ]; then
    archive_local_supervision_logs
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
