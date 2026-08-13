#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
DATA_ROOT="$JOB_ROOT/data/openx_demo"
OUT="$JOB_ROOT/results/qtail_openx_1t_expansion"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
MANIFEST="$OUT/openx_1t_object_manifest.json"
CHECKSUM_MANIFEST="$OUT/openx_1t_checksum_manifest.json"
LEDGER="$OUT/download_checksum_ledger.json"
QUARANTINE="$OUT/checksum_quarantine"
DOWNLOAD_STATUS="$OUT/download_status.json"
VERIFY="$OUT/download_verification.json"
TRAINING_OUT="$OUT/training"
SYNTHESIS_OUT="$OUT/synthesis"
LOCK_DIR="$OUT/pipeline.lock"
LOG="$OUT/pipeline.log"
RESERVE_BYTES=966367641600
MARKER_TOOL="$ROOT/tools/qtail_openx_stage_marker.py"
PAGE_QA_TOOL="$ROOT/tools/qtail_openx_final_page_qa.py"

export QTAIL_OPENX_FEATURE_CACHE_DIR="$OUT/feature_cache"
export QTAIL_OPENX_CHECKSUM_LEDGER="$LEDGER"

mkdir -p "$OUT" "$QUARANTINE"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

while ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; do
  sleep 30
done

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

if "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage synthesis >/dev/null 2>&1 \
  && "$PYTHON" "$PAGE_QA_TOOL" validate --root "$OUT" --workspace "$ROOT" \
    >/dev/null 2>&1; then
  exit 0
fi

if [[ ! -f "$MANIFEST" || ! -f "$CHECKSUM_MANIFEST" ]]; then
  log "building direct Open X expansion manifest"
  "$PYTHON" "$ROOT/tools/qtail_build_openx_expansion_manifest.py" \
    --data-dir "$DATA_ROOT" \
    --out-dir "$OUT" \
    --interface en1 \
    --max-single-fraction 0.80 >> "$LOG" 2>&1 || exit 1
fi

if ! "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage download >/dev/null 2>&1; then
  rm -f "$OUT/OPENX_1T_DOWNLOAD_COMPLETE"
  log "requiring fresh UniClash direct-transport gate"
  "$PYTHON" "$ROOT/tools/qtail_assert_uniclash_transport_gate.py" \
    --guard "$ROOT/.tmp/qtail-uniclash-transport-guard.json" \
    --out "$OUT/uniclash_launch_gate.json" \
    --expected-interface en1 \
    --max-age-seconds 10 \
    --quiet || exit 1

  log "starting/resuming 1 TiB Open X direct download"
  "$PYTHON" "$ROOT/tools/qtail_parallel_gcs_download.py" \
    --manifest "$MANIFEST" \
    --checksum-manifest "$CHECKSUM_MANIFEST" \
    --checksum-ledger "$LEDGER" \
    --checksum-quarantine "$QUARANTINE" \
    --target "$DATA_ROOT" \
    --status "$DOWNLOAD_STATUS" \
    --workers 32 \
    --chunk-mib 64 \
    --primary-endpoints 2 \
    --proxy direct \
    --forbid-tunnel-route \
    --expected-interface en1 \
    --required-mount /Volumes/ORICO \
    --process-lock "$OUT/downloader.lock" \
    --reserve-free-bytes "$RESERVE_BYTES" >> "$LOG" 2>&1 || exit 1

  log "verifying selected files and official MD5 ledger"
  "$PYTHON" "$ROOT/tools/qtail_verify_openx_expansion.py" \
    --manifest "$MANIFEST" \
    --checksum-manifest "$CHECKSUM_MANIFEST" \
    --ledger "$LEDGER" \
    --data-dir "$DATA_ROOT" \
    --out "$VERIFY" \
    --require-complete >> "$LOG" 2>&1 || exit 1

  "$PYTHON" "$MARKER_TOOL" write --root "$OUT" --stage download \
    >> "$LOG" 2>&1 || exit 1
fi

if ! "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage training >/dev/null 2>&1; then
  rm -f "$OUT/OPENX_1T_TRAINING_COMPLETE"
  log "requiring complete checksum-bound TFRecord feature cache"
  "$PYTHON" "$ROOT/tools/qtail_openx_feature_prewarm.py" \
    --data-dir "$DATA_ROOT" \
    --manifest "$CHECKSUM_MANIFEST" \
    --ledger "$LEDGER" \
    --cache-dir "$OUT/feature_cache" \
    --status "$OUT/prewarm_status.json" \
    --records-per-shard 4 \
    --process-lock "$OUT/prewarm-process.lock" \
    --require-complete >> "$LOG" 2>&1 || exit 1

  log "download verified; starting 20,000-step Source/Q-Tail training"
  mkdir -p "$TRAINING_OUT"
  "$PYTHON" "$ROOT/tools/qtail_run_openx_training_with_status.py" \
    --trainer "$ROOT/tools/qtail_train_openx_cached.py" \
    --data-dir "$DATA_ROOT" \
    --out "$TRAINING_OUT" \
    --steps 20000 \
    --records-per-shard 4 \
    --min-record-parse-rate 0.95 >> "$LOG" 2>&1 || exit 1

  "$PYTHON" "$MARKER_TOOL" write --root "$OUT" --stage training \
    >> "$LOG" 2>&1 || exit 1
fi

if ! "$PYTHON" "$MARKER_TOOL" validate --root "$OUT" --stage synthesis \
  >/dev/null 2>&1; then
  rm -f "$OUT/OPENX_1T_SYNTHESIS_COMPLETE" "$OUT/final_page_qa.json"
  log "training complete; generating PT-heavy-tail synthetic delivery package"
  mkdir -p "$SYNTHESIS_OUT"
  "$PYTHON" "$ROOT/tools/qtail_run_openx_synthesis_with_status.py" \
    --generator "$ROOT/tools/qtail_openx_service_model.py" \
    --validator "$ROOT/tools/qtail_validate_package.py" \
    --input "$ROOT/data/customer_semifinal_embodied_tasks.csv" \
    --out "$SYNTHESIS_OUT" \
    --training-report "$TRAINING_OUT/openx_demo_training_report.json" \
    --training-rows "$TRAINING_OUT/openx_shard_training_rows.csv" \
    --synthetic-budget 100000 \
    --top-k 128 >> "$LOG" 2>&1 || exit 1

  "$PYTHON" "$MARKER_TOOL" write --root "$OUT" --stage synthesis \
    >> "$LOG" 2>&1 || exit 1
fi
log "Open X 1 TiB expansion training and long-tail synthesis complete; projecting page"
"$PYTHON" "$ROOT/tools/qtail_openx_expansion_status.py" \
  --root "$OUT" --out "$OUT/status.json" >> "$LOG" 2>&1 || exit 1
"$PYTHON" "$PAGE_QA_TOOL" write --root "$OUT" --workspace "$ROOT" \
  >> "$LOG" 2>&1 || exit 1
"$PYTHON" "$ROOT/tools/qtail_openx_expansion_status.py" \
  --root "$OUT" --out "$OUT/status.json" >> "$LOG" 2>&1 || exit 1
"$PYTHON" "$PAGE_QA_TOOL" validate --root "$OUT" --workspace "$ROOT" \
  >> "$LOG" 2>&1 || exit 1
log "Open X final page projection verified"
