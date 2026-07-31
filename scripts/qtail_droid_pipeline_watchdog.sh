#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
PIPELINE="$ROOT/scripts/qtail_orico_full_pipeline.sh"
HANDOFF="$ROOT/scripts/qtail_reload_pipeline_after_download.sh"
DOWNLOAD_MARKER="$JOB_ROOT/manifests/DROID_DOWNLOAD_COMPLETE"
LOG="$JOB_ROOT/logs/pipeline_watchdog.log"
STATUS="$JOB_ROOT/logs/pipeline_watchdog_status.json"
EXPECTED_COMMAND="/bin/zsh $PIPELINE"
launched_pid=""
stopped_since=0

write_status() {
  local pipeline_pid="$1"
  local pipeline_state="$2"
  local temporary="$STATUS.tmp.$$"
  printf '{"generated_at":"%s","watchdog_pid":%s,"pipeline_pid":%s,"pipeline_state":"%s","expected_command":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$$" \
    "${pipeline_pid:-null}" \
    "$pipeline_state" \
    "$EXPECTED_COMMAND" > "$temporary"
  mv "$temporary" "$STATUS"
}

ensure_download_handoff() {
  local pipeline_pid="$1"
  local expected_handoff="/bin/zsh $HANDOFF $pipeline_pid"
  local handoff_pid

  # A post-download handoff would repeatedly restart checksum or training.
  if [ -f "$DOWNLOAD_MARKER" ]; then
    return 0
  fi

  handoff_pid="$(pgrep -f -x "$expected_handoff" | head -1 || true)"
  if [ -n "$handoff_pid" ]; then
    return 0
  fi

  printf '[%s] handoff missing; starting target_pipeline_pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" >> "$LOG"
  /usr/bin/nohup /bin/zsh "$HANDOFF" "$pipeline_pid" \
    >/dev/null 2>&1 &

  for _ in {1..5}; do
    sleep 1
    handoff_pid="$(pgrep -f -x "$expected_handoff" | head -1 || true)"
    if [ -n "$handoff_pid" ]; then
      printf '[%s] handoff started pid=%s target_pipeline_pid=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$handoff_pid" "$pipeline_pid" >> "$LOG"
      return 0
    fi
  done

  printf '[%s] handoff launch did not become observable target_pipeline_pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" >> "$LOG"
  return 1
}

while true; do
  if ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
    write_status "" "waiting_for_orico"
    sleep 60
    continue
  fi

  pipeline_pid="$(pgrep -f -x "$EXPECTED_COMMAND" | head -1 || true)"
  if [ -z "$pipeline_pid" ]; then
    if [ -n "$launched_pid" ]; then
      wait "$launched_pid" 2>/dev/null || true
      launched_pid=""
    fi
    printf '[%s] pipeline missing; restarting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    /bin/zsh "$PIPELINE" >> "$LOG" 2>&1 &
    launched_pid="$!"
    write_status "$launched_pid" "restarted"
    stopped_since=0
    sleep 30
    continue
  fi

  pipeline_command="$(ps -p "$pipeline_pid" -o command= 2>/dev/null || true)"
  if [ "$pipeline_command" != "$EXPECTED_COMMAND" ]; then
    printf '[%s] refusing unexpected pipeline pid=%s command=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" "$pipeline_command" >> "$LOG"
    write_status "$pipeline_pid" "unexpected_command"
    sleep 30
    continue
  fi
  ensure_download_handoff "$pipeline_pid" || true
  pipeline_state="$(ps -p "$pipeline_pid" -o state= 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$pipeline_state" == *T* ]]; then
    if [ "$stopped_since" -eq 0 ]; then
      stopped_since="$(date +%s)"
    fi
    stopped_for="$(( $(date +%s) - stopped_since ))"
    if [ "$stopped_for" -ge 180 ]; then
      printf '[%s] pipeline pid=%s stopped for %ss; terminating for clean restart\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" "$stopped_for" >> "$LOG"
      kill -TERM "$pipeline_pid" 2>/dev/null || true
      kill -CONT "$pipeline_pid" 2>/dev/null || true
    fi
  else
    stopped_since=0
  fi
  write_status "$pipeline_pid" "${pipeline_state:-unknown}"
  sleep 30
done
