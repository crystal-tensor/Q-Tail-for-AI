#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
LOG="$JOB_ROOT/logs/launchd_supervisor.log"
LOCAL_LOG_ROOT="$ROOT/.tmp"
PROGRESS_LOOP="$ROOT/scripts/qtail_droid_progress_loop.sh"
SERVE="$ROOT/node_modules/.bin/serve"

while ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; do
  sleep 30
done

mkdir -p "$JOB_ROOT/logs"
mkdir -p "$LOCAL_LOG_ROOT"

if ! pgrep -f -x "/bin/zsh $PROGRESS_LOOP" >/dev/null 2>&1; then
  nohup /bin/zsh "$PROGRESS_LOOP" \
    >> "$LOCAL_LOG_ROOT/qtail-droid-progress.log" 2>&1 &
fi

start_web() {
  port="$1"
  if /usr/bin/curl --silent --max-time 2 \
    "http://127.0.0.1:$port/qtail-droid-full-training" \
    >/dev/null 2>&1; then
    return
  fi
  nohup "$SERVE" -l "tcp://0.0.0.0:$port" \
    >> "$LOCAL_LOG_ROOT/qtail-web-$port.log" 2>&1 &
}

start_web 54655
start_web 6222

while true; do
  printf '[%s] starting full pipeline\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  /bin/zsh "$ROOT/scripts/qtail_orico_full_pipeline.sh" >> "$LOG" 2>&1
  code=$?
  printf '[%s] pipeline exited code=%s; restarting in 60s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$LOG"
  sleep 60
done
