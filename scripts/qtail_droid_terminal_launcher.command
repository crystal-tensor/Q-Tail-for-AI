#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
WEB_SERVICES="$ROOT/scripts/qtail_web_services.sh"
PROGRESS_LOOP="$ROOT/scripts/qtail_droid_progress_loop.sh"
WATCHDOG="$ROOT/scripts/qtail_droid_pipeline_watchdog.sh"
PREWARM_LOOP="$ROOT/scripts/qtail_droid_feature_prewarm_loop.sh"
PIPELINE="$ROOT/scripts/qtail_orico_full_pipeline.sh"
RELOAD_HANDOFF="$ROOT/scripts/qtail_reload_pipeline_after_download.sh"
MARKER_ROOT="/Volumes/ORICO/qtail_full_training/manifests"
LOCAL_LOG="$ROOT/.tmp/qtail-droid-terminal-launcher.log"
LATEST_STATUS="$JOB_ROOT/results/qtail_droid_full/latest.json"
WATCHDOG_STATUS="$JOB_ROOT/logs/pipeline_watchdog_status.json"
PREWARM_STATUS="$JOB_ROOT/results/qtail_droid_full/droid_feature_prewarm_status.json"
PREWARM_ACTIVE_STATUS="$JOB_ROOT/results/qtail_droid_full/droid_feature_extraction_status.json"
PREWARM_HEARTBEAT="$JOB_ROOT/results/qtail_droid_full/droid_feature_prewarm_heartbeat.json"
DATA_ROOT="$JOB_ROOT/data/droid"
WEB_SUPERVISOR_COMMAND="/bin/zsh $WEB_SERVICES"
WEB_SUPERVISOR_SESSION="qtail-web-supervisor"

mkdir -p "$ROOT/.tmp"

if ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
  printf '[%s] ORICO is not mounted; scheduled supervisor will retry\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOCAL_LOG"
  exit 0
fi

if ! pgrep -f -x "$WEB_SUPERVISOR_COMMAND" >/dev/null 2>&1; then
  /usr/bin/screen -S "$WEB_SUPERVISOR_SESSION" -X quit \
    >/dev/null 2>&1 || true
  /usr/bin/screen -wipe >/dev/null 2>&1 || true
  /usr/bin/screen -dmS "$WEB_SUPERVISOR_SESSION" \
    /bin/zsh -lc "exec /bin/zsh '$WEB_SERVICES'"
  printf '[%s] started %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$WEB_SUPERVISOR_SESSION" >> "$LOCAL_LOG"
fi

file_fresh() {
  local target_path="$1"
  local max_age="$2"
  local modified now age

  [ -f "$target_path" ] || return 1
  modified="$(/usr/bin/stat -f %m "$target_path" 2>/dev/null || true)"
  [[ "$modified" =~ '^[0-9]+$' ]] || return 1
  now="$(date +%s)"
  age="$(( now - modified ))"
  [ "$age" -ge 0 ] && [ "$age" -le "$max_age" ]
}

progress_healthy() {
  file_fresh "$LATEST_STATUS" 180
}

watchdog_healthy() {
  file_fresh "$WATCHDOG_STATUS" 120
}

prewarm_heartbeat_valid() {
  local heartbeat_pid expected_pid
  local prewarm_pids=()

  prewarm_pids=("${(@f)$(pgrep -f -x "/bin/zsh $PREWARM_LOOP" 2>/dev/null)}")
  [ "${#prewarm_pids[@]}" -eq 1 ] || return 1
  expected_pid="${prewarm_pids[1]}"
  file_fresh "$PREWARM_HEARTBEAT" 150 || return 1
  heartbeat_pid="$(
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
      - "$PREWARM_HEARTBEAT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("control") != "droid_feature_prewarm_pid_heartbeat_v1":
        raise ValueError("unexpected heartbeat control")
    if payload.get("status") != "alive":
        raise ValueError("prewarm heartbeat is not alive")
    print(int(payload["pid"]))
except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
    print(-1)
PY
  )"
  [[ "$heartbeat_pid" =~ '^[0-9]+$' ]] \
    && [ "$heartbeat_pid" -eq "$expected_pid" ]
}

prewarm_healthy() {
  local complete_shards committed_shards

  complete_shards="$(
    find "$DATA_ROOT" -type f -iname '*tfrecord*' \
      ! -name '*.qtail.part*' \
      ! -name '*.headers' \
      ! -name '*.gstmp' \
      ! -name '*.tmp' 2>/dev/null \
      | wc -l | tr -d ' '
  )"
  committed_shards="$(
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
      - "$PREWARM_STATUS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(int(payload.get("shard_count", -1)))
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    print(-1)
PY
  )"
  if [[ "$complete_shards" =~ '^[0-9]+$' ]] \
    && [[ "$committed_shards" =~ '^-?[0-9]+$' ]] \
    && [ "$committed_shards" -ge "$complete_shards" ]; then
    return 0
  fi
  prewarm_heartbeat_valid || file_fresh "$PREWARM_ACTIVE_STATUS" 300
}

descendants_of() {
  local root_pid="$1"
  local frontier=("$root_pid")
  local discovered=()
  local parent child

  while [ "${#frontier[@]}" -gt 0 ]; do
    parent="${frontier[1]}"
    frontier=("${frontier[@]:1}")
    while IFS= read -r child; do
      [ -n "$child" ] || continue
      discovered+=("$child")
      frontier+=("$child")
    done < <(pgrep -P "$parent" 2>/dev/null || true)
  done
  printf '%s\n' "${discovered[@]}"
}

stop_process_tree() {
  local root_pid="$1"
  local descendants=()
  local pid

  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    descendants+=("$pid")
  done < <(descendants_of "$root_pid")
  for pid in "${(@Oa)descendants}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  kill -TERM "$root_pid" 2>/dev/null || true
  sleep 1
  for pid in "${(@Oa)descendants}" "$root_pid"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

ensure_screen() {
  local process_command="$1"
  local session="$2"
  local script="$3"
  local health_check="$4"
  local pids=()
  local pid

  pids=("${(@f)$(pgrep -f -x "$process_command" 2>/dev/null || true)}")
  pids=("${(@)pids:#}")
  if [ "${#pids[@]}" -eq 1 ] && "$health_check"; then
    printf '[%s] %s already active\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$session" >> "$LOCAL_LOG"
    return 0
  fi
  if [ "${#pids[@]}" -gt 0 ]; then
    printf '[%s] replacing stale/duplicate %s pids=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$session" "${pids[*]}" >> "$LOCAL_LOG"
    for pid in "${pids[@]}"; do
      stop_process_tree "$pid"
    done
  fi
  /usr/bin/screen -S "$session" -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -wipe >/dev/null 2>&1 || true
  /usr/bin/screen -dmS "$session" /bin/zsh -lc "exec /bin/zsh '$script'"
  printf '[%s] started %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$session" >> "$LOCAL_LOG"
}

ensure_screen \
  "/bin/zsh $PROGRESS_LOOP" \
  qtail-droid-progress \
  "$PROGRESS_LOOP" \
  progress_healthy

if [ ! -f "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE" ]; then
  ensure_screen \
    "/bin/zsh $PREWARM_LOOP" \
    qtail-droid-prewarm \
    "$PREWARM_LOOP" \
    prewarm_healthy
fi

ensure_screen \
  "/bin/zsh $WATCHDOG" \
  qtail-droid-watchdog \
  "$WATCHDOG" \
  watchdog_healthy

if [ ! -f "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE" ]; then
  pipeline_pid="$(
    pgrep -f -x "/bin/zsh $PIPELINE" 2>/dev/null \
      | head -n 1
  )"
  if [ -n "$pipeline_pid" ]; then
    handoff_command="/bin/zsh $RELOAD_HANDOFF $pipeline_pid"
    if ! pgrep -f -x "$handoff_command" >/dev/null 2>&1; then
      /usr/bin/screen -S qtail-droid-generation-handoff -X quit \
        >/dev/null 2>&1 || true
      /usr/bin/screen -dmS qtail-droid-generation-handoff \
        /bin/zsh -lc "exec /bin/zsh '$RELOAD_HANDOFF' '$pipeline_pid'"
      printf '[%s] armed generation handoff for pipeline pid=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" >> "$LOCAL_LOG"
    fi
  fi
fi
exit 0
