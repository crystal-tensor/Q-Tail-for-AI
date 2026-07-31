#!/bin/zsh
set -u

JOB_ROOT="/Volumes/ORICO/qtail_full_training"
MARKER_ROOT="$JOB_ROOT/manifests"
RESULT_ROOT="$JOB_ROOT/results/qtail_droid_full"
DATA_ROOT="$JOB_ROOT/data/droid"
LOG="$JOB_ROOT/logs/pipeline_generation_handoff.log"
TARGET_PID="${1:-}"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
DOWNLOAD_MARKER_VERIFIER="/Users/avalok/work/Q-TAIL-MVP/tools/qtail_verify_droid_download_marker.py"
DOWNLOAD_MARKER="$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE"
OBJECT_MANIFEST="$RESULT_ROOT/droid_object_manifest.json"
CHECKSUM_MANIFEST="$RESULT_ROOT/droid_object_checksum_manifest.json"
CHECKSUM_LEDGER="$RESULT_ROOT/droid_object_checksum_ledger.json"
TRANSPORT_STATUS="$RESULT_ROOT/parallel_download_status.json"
REMOTE_BYTES=3700745265151
EXPECTED_COMMAND="/bin/zsh /Users/avalok/work/Q-TAIL-MVP/scripts/qtail_orico_full_pipeline.sh"
WATCHDOG="/Users/avalok/work/Q-TAIL-MVP/scripts/qtail_droid_pipeline_watchdog.sh"
EXPECTED_WATCHDOG_COMMAND="/bin/zsh $WATCHDOG"
PREWARM="/Users/avalok/work/Q-TAIL-MVP/scripts/qtail_droid_feature_prewarm_loop.sh"
EXPECTED_PREWARM_COMMAND="/bin/zsh $PREWARM"
DOWNLOAD_MARKER_POLL_SECONDS=1

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

if [[ ! "$TARGET_PID" =~ '^[0-9]+$' ]]; then
  printf '[%s] invalid target pid: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"
  exit 2
fi

watchdog_pids=("${(@f)$(pgrep -f -x "$EXPECTED_WATCHDOG_COMMAND" 2>/dev/null || true)}")
watchdog_pids=("${(@)watchdog_pids:#}")
if [ "${#watchdog_pids[@]}" -ne 1 ]; then
  printf '[%s] expected exactly one watchdog, found %s: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${#watchdog_pids[@]}" "${watchdog_pids[*]:-none}" >> "$LOG"
  exit 10
fi
WATCHDOG_PID="${watchdog_pids[1]}"

# The detached screen/login wrapper reparents the pipeline, so parentage does
# not prove ownership. Lock the handoff to the sole exact-command pipeline
# observed alongside the sole exact-command watchdog.
pipeline_pids=("${(@f)$(pgrep -f -x "$EXPECTED_COMMAND" 2>/dev/null || true)}")
pipeline_pids=("${(@)pipeline_pids:#}")
if [ "${#pipeline_pids[@]}" -ne 1 ] || [ "${pipeline_pids[1]:-}" != "$TARGET_PID" ]; then
  printf '[%s] target is not the sole expected pipeline: target=%s observed=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$TARGET_PID" "${pipeline_pids[*]:-none}" >> "$LOG"
  exit 15
fi

while true; do
  if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    printf '[%s] target pipeline pid %s already exited; no handoff needed\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"
    exit 0
  fi
  if [ -f "$DOWNLOAD_MARKER" ]; then
    if "$PYTHON" "$DOWNLOAD_MARKER_VERIFIER" \
      --data-dir "$DATA_ROOT" \
      --manifest "$OBJECT_MANIFEST" \
      --checksum-manifest "$CHECKSUM_MANIFEST" \
      --checksum-ledger "$CHECKSUM_LEDGER" \
      --transport-status "$TRANSPORT_STATUS" \
      --marker "$DOWNLOAD_MARKER" \
      --expected-bytes "$REMOTE_BYTES" >> "$LOG" 2>&1; then
      break
    fi
    printf '[%s] download marker exists but failed semantic binding; handoff remains withheld\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  fi
  sleep "$DOWNLOAD_MARKER_POLL_SECONDS"
done

command="$(ps -p "$TARGET_PID" -o command= 2>/dev/null || true)"
if [ "$command" != "$EXPECTED_COMMAND" ]; then
  printf '[%s] pid %s command changed; refusing to signal: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" "$command" >> "$LOG"
  exit 3
fi

printf '[%s] download gate complete; requesting pipeline pid %s reload before training\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"

# Freeze the old shell before it can advance into checksum/training. If it
# already spawned a child in the polling window, stop that entire descendant
# tree so the watchdog restarts one coherent code generation.
if ! kill -STOP "$TARGET_PID" 2>/dev/null; then
  printf '[%s] failed to freeze target pipeline pid %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"
  exit 4
fi
sleep 1
state="$(ps -p "$TARGET_PID" -o state= 2>/dev/null | tr -d '[:space:]' || true)"
if [[ "$state" != *T* ]]; then
  printf '[%s] target pipeline pid %s did not enter stopped state: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" "$state" >> "$LOG"
  kill -CONT "$TARGET_PID" 2>/dev/null || true
  exit 5
fi
descendants=()
while IFS= read -r pid; do
  [ -n "$pid" ] || continue
  descendants+=("$pid")
done < <(descendants_of "$TARGET_PID")
if [ "${#descendants[@]}" -gt 0 ]; then
  printf '[%s] stopping old-generation descendants: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${descendants[*]}" >> "$LOG"
  for pid in "${descendants[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${descendants[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
fi
if ! kill -TERM "$TARGET_PID" 2>/dev/null; then
  kill -CONT "$TARGET_PID" 2>/dev/null || true
  exit 6
fi
if ! kill -CONT "$TARGET_PID" 2>/dev/null; then
  exit 7
fi

for _ in {1..30}; do
  if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$TARGET_PID" 2>/dev/null; then
  printf '[%s] target pipeline pid %s did not exit after TERM/CONT; forcing exit\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"
  kill -KILL "$TARGET_PID" 2>/dev/null || exit 8
  sleep 1
fi
if kill -0 "$TARGET_PID" 2>/dev/null; then
  printf '[%s] target pipeline pid %s is still alive; reload not committed\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"
  exit 9
fi

printf '{"status":"committed","old_pipeline_pid":%s,"committed_at":"%s"}\n' \
  "$TARGET_PID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$MARKER_ROOT/PIPELINE_RELOAD_REQUESTED_AFTER_DOWNLOAD.tmp"
mv "$MARKER_ROOT/PIPELINE_RELOAD_REQUESTED_AFTER_DOWNLOAD.tmp" \
  "$MARKER_ROOT/PIPELINE_RELOAD_REQUESTED_AFTER_DOWNLOAD"
printf '[%s] old pipeline pid %s exited; watchdog may load the new generation\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_PID" >> "$LOG"

if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
  kill -TERM "$WATCHDOG_PID" 2>/dev/null || exit 11
  for _ in {1..10}; do
    if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -KILL "$WATCHDOG_PID" 2>/dev/null || exit 12
  fi
fi
/usr/bin/screen -S qtail-droid-watchdog -X quit >/dev/null 2>&1 || true
/usr/bin/screen -dmS qtail-droid-watchdog \
  /bin/zsh -lc "exec /bin/zsh '$WATCHDOG'"
NEW_WATCHDOG_PID=""
for _ in {1..10}; do
  NEW_WATCHDOG_PID="$(
    pgrep -f -x "$EXPECTED_WATCHDOG_COMMAND" | head -1 || true
  )"
  if [ -n "$NEW_WATCHDOG_PID" ]; then
    printf '[%s] watchdog reloaded pid=%s with current generation\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NEW_WATCHDOG_PID" >> "$LOG"
    break
  fi
  sleep 1
done
if [ -z "$NEW_WATCHDOG_PID" ]; then
  printf '[%s] current-generation watchdog did not start\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  exit 13
fi

# The old prewarm shell may still be completing an atomic cache pass. Let it
# finish naturally, then load the current-generation loop so checksum repairs
# can continue to be prewarmed without running duplicate writers.
while pgrep -f -x "$EXPECTED_PREWARM_COMMAND" >/dev/null 2>&1; do
  sleep 10
done
if [ ! -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then
  /usr/bin/screen -S qtail-droid-prewarm -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -dmS qtail-droid-prewarm \
    /bin/zsh -lc "exec /bin/zsh '$PREWARM'"
  NEW_PREWARM_PID=""
  for _ in {1..10}; do
    NEW_PREWARM_PID="$(
      pgrep -f -x "$EXPECTED_PREWARM_COMMAND" | head -1 || true
    )"
    if [ -n "$NEW_PREWARM_PID" ]; then
      printf '[%s] feature prewarm reloaded pid=%s with current generation\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NEW_PREWARM_PID" >> "$LOG"
      exit 0
    fi
    sleep 1
  done
  printf '[%s] current-generation feature prewarm did not start\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  exit 14
fi
printf '[%s] checksum already verified; feature prewarm reload is unnecessary\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
exit 0
