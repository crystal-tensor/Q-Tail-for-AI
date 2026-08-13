#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
DATA_ROOT="$JOB_ROOT/data/droid"
RESULT_ROOT="$JOB_ROOT/results/qtail_droid_full"
MARKER_ROOT="$JOB_ROOT/manifests"
LOG="$JOB_ROOT/logs/droid_feature_prewarm.log"
TRAINER="$ROOT/tools/qtail_train_droid_full.py"
CACHE_VERIFIER="$ROOT/tools/qtail_verify_droid_feature_cache.py"
CLOSURE_AUDITOR="$ROOT/tools/qtail_audit_droid_incremental_closure.py"
CLOSURE_SELFTEST="$ROOT/tools/qtail_droid_incremental_closure_selftest.py"
RELEASE_MILESTONE_SEALER="$ROOT/tools/qtail_seal_droid_release_milestones.py"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
OBJECT_MANIFEST="$RESULT_ROOT/droid_object_manifest.json"
CLEAR_CHECKSUM_MANIFEST="$RESULT_ROOT/droid_object_checksum_manifest.json"
CHECKSUM_LEDGER="$RESULT_ROOT/droid_object_checksum_ledger.json"
CACHE_MANIFEST="$RESULT_ROOT/droid_feature_cache_manifest.json"
PARTIAL_CACHE_VERIFICATION="$RESULT_ROOT/droid_feature_cache_partial_verification.json"
INCREMENTAL_CLOSURE="$RESULT_ROOT/droid_incremental_closure_audit.json"
INCREMENTAL_CLOSURE_SELFTEST="$RESULT_ROOT/droid_incremental_closure_selftest.json"
RELEASE_MILESTONE_STATUS="$RESULT_ROOT/droid_release_milestone_status.json"
RELEASE_MILESTONE_DIR="$RESULT_ROOT/release_milestones"
PREWARM_HEARTBEAT="$RESULT_ROOT/droid_feature_prewarm_heartbeat.json"
LAST_SHARD_COUNT=-1

mkdir -p "$RESULT_ROOT" "$MARKER_ROOT" "$JOB_ROOT/logs"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

complete_shard_count() {
  find "$DATA_ROOT" -type f -iname '*tfrecord*' \
    ! -name '*.qtail.part*' \
    ! -name '*.headers' \
    ! -name '*.gstmp' \
    ! -name '*.tmp' 2>/dev/null | wc -l | tr -d ' '
}

write_heartbeat() {
  local phase="$1"
  local shard_count="$2"
  local child_pid="${3:-0}"
  if ! "$PYTHON" - \
    "$PREWARM_HEARTBEAT" "$phase" "$shard_count" "$child_pid" "$$" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
payload = {
    "child_pid": int(sys.argv[4]),
    "control": "droid_feature_prewarm_pid_heartbeat_v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "observed_complete_shards": int(sys.argv[3]),
    "phase": sys.argv[2],
    "pid": int(sys.argv[5]),
    "status": "alive",
}
try:
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY
  then
    log "heartbeat atomic publish failed phase=$phase path=$PREWARM_HEARTBEAT"
    return 1
  fi
}

run_with_heartbeat() {
  local phase="$1"
  local shard_count="$2"
  local child_pid code
  shift 2

  "$@" >> "$LOG" 2>&1 &
  child_pid=$!
  write_heartbeat "$phase" "$shard_count" "$child_pid"
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 20
    if kill -0 "$child_pid" 2>/dev/null; then
      write_heartbeat "$phase" "$shard_count" "$child_pid"
    fi
  done
  wait "$child_pid"
  code=$?
  write_heartbeat "${phase}_complete" "$shard_count" 0
  return "$code"
}

heartbeat_sleep() {
  local phase="$1"
  local shard_count="$2"
  local seconds="$3"

  write_heartbeat "$phase" "$shard_count" 0
  sleep "$seconds"
}

while true; do
  if ! write_heartbeat "checking_mount" "$LAST_SHARD_COUNT" 0; then
    exit 23
  fi
  if ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
    heartbeat_sleep "waiting_for_mount" "$LAST_SHARD_COUNT" 60
    continue
  fi

  if [ -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then
    write_heartbeat "checksum_verified_exit" "$LAST_SHARD_COUNT" 0
    log "checksum-verified marker present; prewarm loop exiting"
    exit 0
  fi

  shard_count="$(complete_shard_count)"
  if [ "$shard_count" -gt 0 ] && [ "$shard_count" -ne "$LAST_SHARD_COUNT" ]; then
    log "prewarming $shard_count complete TFRecord shards"
    run_with_heartbeat "feature_extraction" "$shard_count" \
      "$PYTHON" "$TRAINER" \
      --data-dir "$DATA_ROOT" \
      --out "$RESULT_ROOT" \
      --required-mount /Volumes/ORICO \
      --records-per-shard 0 \
      --min-shards 1 \
      --status-every-shards 1 \
      --features-only
    code=$?
    log "prewarm pass exited code=$code for $shard_count shards"
    if [ "$code" -eq 0 ]; then
      run_with_heartbeat "official_record_audit" "$shard_count" \
        "$PYTHON" "$CACHE_VERIFIER" \
        --data-dir "$DATA_ROOT" \
        --object-manifest "$OBJECT_MANIFEST" \
        --cache-manifest "$CACHE_MANIFEST" \
        --out "$PARTIAL_CACHE_VERIFICATION"
      verification_code=$?
      log "official shardLengths audit exited code=$verification_code for $shard_count shards"
      if [ "$verification_code" -eq 0 ]; then
        run_with_heartbeat "incremental_closure" "$shard_count" \
          "$PYTHON" "$CLOSURE_AUDITOR" \
          --data-dir "$DATA_ROOT" \
          --checksum-manifest "$CLEAR_CHECKSUM_MANIFEST" \
          --checksum-ledger "$CHECKSUM_LEDGER" \
          --cache-manifest "$CACHE_MANIFEST" \
          --record-audit "$PARTIAL_CACHE_VERIFICATION" \
          --out "$INCREMENTAL_CLOSURE"
        closure_code=$?
        log "incremental MD5/record/cache closure exited code=$closure_code for $shard_count shards"
        if [ "$closure_code" -eq 0 ]; then
          run_with_heartbeat "incremental_closure_selftest" "$shard_count" \
            "$PYTHON" "$CLOSURE_SELFTEST" \
            --auditor "$CLOSURE_AUDITOR" \
            --python "$PYTHON" \
            --data-dir "$DATA_ROOT" \
            --checksum-manifest "$CLEAR_CHECKSUM_MANIFEST" \
            --checksum-ledger "$CHECKSUM_LEDGER" \
            --cache-manifest "$CACHE_MANIFEST" \
            --record-audit "$PARTIAL_CACHE_VERIFICATION" \
            --out "$INCREMENTAL_CLOSURE_SELFTEST"
          closure_selftest_code=$?
          log "incremental closure controls exited code=$closure_selftest_code for $shard_count shards"
          if [ "$closure_selftest_code" -eq 0 ]; then
            run_with_heartbeat "release_milestone_audit" "$shard_count" \
              "$PYTHON" "$RELEASE_MILESTONE_SEALER" \
              --data-dir "$DATA_ROOT" \
              --checksum-manifest "$CLEAR_CHECKSUM_MANIFEST" \
              --checksum-ledger "$CHECKSUM_LEDGER" \
              --closure "$INCREMENTAL_CLOSURE" \
              --milestone-dir "$RELEASE_MILESTONE_DIR" \
              --out "$RELEASE_MILESTONE_STATUS"
            milestone_code=$?
            log "official release milestone audit exited code=$milestone_code for $shard_count shards"
            if [ "$milestone_code" -eq 0 ]; then
              LAST_SHARD_COUNT="$shard_count"
            fi
          fi
        fi
      fi
    fi
  fi
  heartbeat_sleep "idle" "$shard_count" 60
done
