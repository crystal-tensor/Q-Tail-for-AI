#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
JOB_ROOT="/Volumes/ORICO/qtail_full_training"
DATA_ROOT="$JOB_ROOT/data/droid"
OPENX_SOURCE="$ROOT/data/openx_demo"
OPENX_TARGET="$JOB_ROOT/data/openx_demo"
OPENX_BACKUP="$ROOT/data/openx_demo.migration_source"
RESULT_ROOT="$JOB_ROOT/results/qtail_droid_full"
MARKER_ROOT="$JOB_ROOT/manifests"
LOG_ROOT="$JOB_ROOT/logs"
LOG="$LOG_ROOT/droid_full_pipeline.log"
LOCK_DIR="$MARKER_ROOT/pipeline.lock"
GSUTIL="$JOB_ROOT/envs/downloader/bin/gsutil"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
REMOTE_URI="gs://gresearch/robotics/droid"
REMOTE_BYTES=3700745265151
SOURCE_PROBE_REPORT="$RESULT_ROOT/droid_source_probe.json"
OBJECT_MANIFEST="$RESULT_ROOT/droid_object_manifest.json"
CHECKSUM_MANIFEST="$RESULT_ROOT/droid_object_checksum_manifest.json"
CHECKSUM_LEDGER="$RESULT_ROOT/droid_object_checksum_ledger.json"
CHECKSUM_QUARANTINE="$RESULT_ROOT/checksum_quarantine"
CHECKSUM_STAT_CONTINUITY_REPORT="$RESULT_ROOT/droid_checksum_stat_continuity.json"
RELEASE_METADATA_AUDIT="$RESULT_ROOT/droid_release_metadata_audit.json"
DOWNLOAD_MARKER="$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE"
DOWNLOAD_MARKER_LINK="$RESULT_ROOT/download_completion_marker.json"
DOWNLOAD_MARKER_VERIFIER="$ROOT/tools/qtail_verify_droid_download_marker.py"
DOWNLOAD_MARKER_SELFTEST="$ROOT/tools/qtail_droid_download_marker_selftest.py"
DOWNLOAD_MARKER_SELFTEST_REPORT="$RESULT_ROOT/droid_download_marker_selftest.json"
MIRROR_VERIFIER="$ROOT/tools/qtail_verify_droid_mirror.py"
MIRROR_VERIFIER_SELFTEST="$ROOT/tools/qtail_droid_mirror_verifier_selftest.py"
MIRROR_VERIFIER_SELFTEST_REPORT="$RESULT_ROOT/droid_mirror_verifier_selftest.json"
RUNTIME_PROCESS_SELFTEST="$ROOT/tools/qtail_runtime_process_contract_selftest.py"
RUNTIME_PROCESS_SELFTEST_REPORT="$RESULT_ROOT/droid_runtime_process_contract_selftest.json"
UNICLASH_GATE="$ROOT/tools/qtail_assert_uniclash_transport_gate.py"
UNICLASH_GATE_REPORT="$RESULT_ROOT/uniclash_pre_checksum_gate.json"
UNICLASH_CHECKSUM_HANDOFF_GATE_REPORT="$RESULT_ROOT/uniclash_checksum_handoff_gate.json"
UNICLASH_PRE_ENVIRONMENT_GATE_REPORT="$RESULT_ROOT/uniclash_pre_environment_gate.json"
UNICLASH_PRE_TRAINING_GATE_REPORT="$RESULT_ROOT/uniclash_pre_training_gate.json"
UNICLASH_GATE_SELFTEST="$ROOT/tools/qtail_uniclash_transport_gate_selftest.py"
UNICLASH_GATE_SELFTEST_REPORT="$RESULT_ROOT/uniclash_pre_checksum_gate_selftest.json"
UNICLASH_GUARD_STATUS="$ROOT/.tmp/qtail-uniclash-transport-guard.json"
DOWNLOADER_SELFTEST="$ROOT/tools/qtail_downloader_single_writer_selftest.py"
DOWNLOADER_SELFTEST_REPORT="$RESULT_ROOT/droid_downloader_single_writer_selftest.json"
CLASSIFIER_SELFTEST_REPORT="$RESULT_ROOT/uniclash_transport_guard_classifier_v6_selftest.json"
LIVE_PARTIAL_SELFTEST="$ROOT/tools/qtail_capture_droid_partial_marker_rejection.py"
LIVE_PARTIAL_SELFTEST_REPORT="$RESULT_ROOT/droid_live_partial_marker_rejection.json"
STAGE_HARDENING_SELFTEST="$ROOT/tools/qtail_stage_marker_hardening_selftest.py"
STAGE_HARDENING_SELFTEST_REPORT="$RESULT_ROOT/droid_stage_marker_hardening_selftest.json"
PREVIEW_SELFTEST="$ROOT/tools/qtail_progress_preview_selftest.py"
PREVIEW_SELFTEST_REPORT="$RESULT_ROOT/droid_progress_preview_selftest.json"
MANIFEST_SELFTEST="$ROOT/tools/qtail_artifact_manifest_merge_selftest.py"
MANIFEST_SELFTEST_REPORT="$RESULT_ROOT/droid_artifact_manifest_merge_selftest.json"
SHELL_CONTRACT_SELFTEST="$ROOT/tools/qtail_pipeline_shell_contract_selftest.py"
SHELL_CONTRACT_SELFTEST_REPORT="$RESULT_ROOT/droid_pipeline_shell_contract_selftest.json"
PIPELINE_GENERATION_GATE_REPORT="$RESULT_ROOT/pipeline_generation_gate.json"
ORCHESTRATION_SNAPSHOT_MANIFEST="$JOB_ROOT/code/qtail_orchestration/SHA256SUMS"
ORCHESTRATION_SNAPSHOT_PUBLISHER="$ROOT/tools/qtail_publish_orchestration_snapshot.py"
TRAINING_GATE_ORDER_SELFTEST="$ROOT/tools/qtail_droid_training_gate_order_selftest.py"
TRAINING_GATE_ORDER_SELFTEST_REPORT="$RESULT_ROOT/droid_training_gate_order_selftest.json"
FINAL_QA_CONTRACT_BLOCKER="$RESULT_ROOT/final_qa_contract_blocked.json"
DOWNLOAD_PROXY="${QTAIL_DROID_DOWNLOAD_PROXY:-direct}"
DOWNLOAD_WORKERS="${QTAIL_DROID_DOWNLOAD_WORKERS:-16}"
DOWNLOAD_PRIMARY_ENDPOINTS="${QTAIL_DROID_PRIMARY_ENDPOINTS:-2}"
DOWNLOAD_RESERVE_FREE_BYTES="${QTAIL_DROID_RESERVE_FREE_BYTES:-185037263258}"
DOWNLOAD_INTERFACE="${QTAIL_DROID_DOWNLOAD_INTERFACE:-en1}"

wait_for_volume() {
  while ! /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; do
    sleep 60
  done
}

refresh_status() {
  "$PYTHON" "$ROOT/tools/qtail_droid_full_progress.py" --job-root "$JOB_ROOT" >> "$LOG_ROOT/progress_refresh.log" 2>&1 || true
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

write_pipeline_started_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/PIPELINE_STARTED" \
    "$ROOT/scripts/qtail_orico_full_pipeline.sh" \
    "$$" \
    "$PPID" \
    "$LOCK_DIR" \
    "$JOB_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
script = Path(sys.argv[2])
pid = int(sys.argv[3])
ppid = int(sys.argv[4])
lock_path = Path(sys.argv[5])
job_root = Path(sys.argv[6])
payload = {
    "format_version": "qtail_pipeline_started_marker_v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "running",
    "pid": pid,
    "ppid": ppid,
    "script": str(script),
    "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "lock_path": str(lock_path),
    "lock_owner_pid": int(str(lock_path.readlink())),
    "job_root": str(job_root),
    "claim_boundary": (
        "This binds the unique pipeline process to the script bytes observed "
        "immediately after acquiring the pipeline lock. A later workspace "
        "edit is a pending generation until the supervised handoff starts a "
        "new process and rewrites this marker."
    ),
}
if payload["lock_owner_pid"] != pid:
    raise SystemExit("pipeline lock owner differs from marker pid")
temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
PY
}

require_pipeline_generation_marker() {
  local gate="$1"
  "$PYTHON" - \
    "$MARKER_ROOT/PIPELINE_STARTED" \
    "$ROOT/scripts/qtail_orico_full_pipeline.sh" \
    "$$" \
    "$LOCK_DIR" \
    "$JOB_ROOT" \
    "$PIPELINE_GENERATION_GATE_REPORT" \
    "$gate" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

marker_path = Path(sys.argv[1])
script = Path(sys.argv[2])
pid = int(sys.argv[3])
lock_path = Path(sys.argv[4])
job_root = Path(sys.argv[5])
report_path = Path(sys.argv[6])
gate = sys.argv[7]
try:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    marker = {}
try:
    lock_owner_pid = int(str(lock_path.readlink()))
except (OSError, ValueError):
    lock_owner_pid = None
script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
command = subprocess.run(
    ["ps", "-p", str(pid), "-o", "command="],
    check=False,
    capture_output=True,
    text=True,
).stdout.strip()
expected_command = f"/bin/zsh {script}"
checks = {
    "semantic_marker": (
        marker.get("format_version")
        == "qtail_pipeline_started_marker_v2"
    ),
    "running_status": marker.get("status") == "running",
    "marker_pid_matches": marker.get("pid") == pid,
    "marker_script_matches": marker.get("script") == str(script),
    "marker_job_root_matches": marker.get("job_root") == str(job_root),
    "marker_sha_matches_current_source": (
        marker.get("script_sha256") == script_sha256
    ),
    "marker_lock_owner_matches": marker.get("lock_owner_pid") == pid,
    "live_lock_owner_matches": lock_owner_pid == pid,
    "live_command_matches": command == expected_command,
}
passed = all(checks.values())
try:
    previous = json.loads(report_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    previous = {}
history = previous.get("gates", [])
if not isinstance(history, list):
    history = []
entry = {
    "gate": gate,
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "passed": passed,
    "pid": pid,
    "lock_owner_pid": lock_owner_pid,
    "command": command,
    "expected_command": expected_command,
    "marker_script_sha256": marker.get("script_sha256"),
    "current_script_sha256": script_sha256,
    "checks": checks,
}
history = [
    item for item in history
    if isinstance(item, dict) and item.get("gate") != gate
]
history.append(entry)
payload = {
    "format_version": "qtail_pipeline_generation_gate_v1",
    "generated_at": entry["checked_at"],
    "status": "passed" if passed else "blocked",
    "latest_gate": gate,
    "gates": history,
    "claim_boundary": (
        "Each irreversible stage requires the semantic start marker, unique "
        "live pipeline command, pipeline lock owner, and current script "
        "SHA-256 to remain bound to this PID. A mismatch prevents checksum, "
        "environment capture, or formal training from advancing."
    ),
}
report_path.parent.mkdir(parents=True, exist_ok=True)
temporary = report_path.with_name(
    f".{report_path.name}.{os.getpid()}.tmp"
)
temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, report_path)
if not passed:
    raise SystemExit(1)
PY
}

validate_droid_source_probe_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/DROID_SOURCE_PROBED" \
    "$SOURCE_PROBE_REPORT" \
    "$REMOTE_URI" \
    "$REMOTE_BYTES" \
    "$JOB_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
source = sys.argv[3]
expected_bytes = int(sys.argv[4])
job_root = sys.argv[5]
try:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
storage = report.get("storage", {})
if (
    marker.get("format_version")
    != "qtail_droid_source_probe_marker_v2"
    or marker.get("status") != "verified"
    or marker.get("source") != source
    or marker.get("remote_bytes") != expected_bytes
    or marker.get("job_root") != job_root
    or marker.get("report") != str(report_path)
    or marker.get("report_sha256") != report_sha256
    or marker.get("capacity_gate_passed_at_probe") is not True
    or report.get("status") != "verified"
    or report.get("source") != source
    or report.get("remote_bytes") != expected_bytes
    or report.get("job_root") != job_root
    or not isinstance(storage, dict)
    or storage.get("capacity_gate_passed") is not True
):
    raise SystemExit(1)
PY
}

validate_openx_migration_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/OPENX_MIGRATION_COMPLETE" \
    "$OPENX_SOURCE" \
    "$OPENX_TARGET" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
source = Path(sys.argv[2])
target = Path(sys.argv[3])
try:
    payload = json.loads(marker.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if (
    payload.get("format_version") != "qtail_openx_migration_marker_v2"
    or payload.get("status") != "verified"
    or payload.get("source_symlink") != str(source)
    or payload.get("resolved_source") != str(target)
    or payload.get("target") != str(target)
    or not source.is_symlink()
    or source.resolve() != target
    or not target.is_dir()
):
    raise SystemExit(1)
file_count = 0
logical_bytes = 0
dataset_file_count = 0
dataset_logical_bytes = 0
excluded_metadata = []
try:
    for path in sorted(target.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            size = path.stat().st_size
            file_count += 1
            logical_bytes += size
            if path.name == ".DS_Store":
                excluded_metadata.append(
                    {
                        "path": str(path.relative_to(target)),
                        "bytes": size,
                    }
                )
            else:
                dataset_file_count += 1
                dataset_logical_bytes += size
except OSError:
    raise SystemExit(1)
if (
    payload.get("file_count") != file_count
    or payload.get("logical_bytes") != logical_bytes
    or payload.get("dataset_file_count") != dataset_file_count
    or payload.get("dataset_logical_bytes") != dataset_logical_bytes
    or payload.get("excluded_filesystem_metadata") != excluded_metadata
):
    raise SystemExit(1)
PY
}

write_openx_migration_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/OPENX_MIGRATION_COMPLETE" \
    "$OPENX_SOURCE" \
    "$OPENX_TARGET" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
source = Path(sys.argv[2])
target = Path(sys.argv[3])
if not source.is_symlink() or source.resolve() != target or not target.is_dir():
    raise SystemExit(1)
file_count = 0
logical_bytes = 0
dataset_file_count = 0
dataset_logical_bytes = 0
excluded_metadata = []
for path in sorted(target.rglob("*"), key=lambda item: item.as_posix()):
    if path.is_file():
        size = path.stat().st_size
        file_count += 1
        logical_bytes += size
        if path.name == ".DS_Store":
            excluded_metadata.append(
                {
                    "path": str(path.relative_to(target)),
                    "bytes": size,
                }
            )
        else:
            dataset_file_count += 1
            dataset_logical_bytes += size
payload = {
    "format_version": "qtail_openx_migration_marker_v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "verified",
    "source_symlink": str(source),
    "resolved_source": str(source.resolve()),
    "target": str(target),
    "file_count": file_count,
    "logical_bytes": logical_bytes,
    "dataset_file_count": dataset_file_count,
    "dataset_logical_bytes": dataset_logical_bytes,
    "excluded_filesystem_metadata": excluded_metadata,
    "claim_boundary": (
        "This binds the workspace symlink, total directory bytes, and dataset "
        "bytes after explicitly excluding .DS_Store metadata to the ORICO "
        "target. It is not a per-file checksum manifest."
    ),
}
temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
PY
}

validate_droid_backend_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/DROID_BACKEND_READY" \
    "$MARKER_ROOT/droid_policy_learning_commit.txt" \
    "$JOB_ROOT/code/droid_policy_learning" \
    "$ROOT/external_data/embodied_full/training_backends/droid_policy_learning" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

marker = Path(sys.argv[1])
commit_marker = Path(sys.argv[2])
backend = Path(sys.argv[3])
source = Path(sys.argv[4])
try:
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected_commit = commit_marker.read_text(encoding="utf-8").strip()
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if (
    payload.get("format_version") != "qtail_droid_backend_marker_v2"
    or payload.get("status") != "verified"
    or payload.get("source_root") != str(source)
    or payload.get("backend_root") != str(backend)
    or payload.get("git_commit") != expected_commit
    or payload.get("commit_marker") != str(commit_marker)
    or payload.get("commit_marker_sha256")
    != hashlib.sha256(commit_marker.read_bytes()).hexdigest()
    or payload.get("git_fsck_passed") is not True
    or not backend.is_dir()
):
    raise SystemExit(1)
head = subprocess.run(
    ["git", "-C", str(backend), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
).stdout.strip()
if head != expected_commit:
    raise SystemExit(1)
if subprocess.run(
    ["git", "-C", str(backend), "fsck", "--no-progress"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode:
    raise SystemExit(1)
PY
}

write_droid_backend_marker() {
  "$PYTHON" - \
    "$MARKER_ROOT/DROID_BACKEND_READY" \
    "$MARKER_ROOT/droid_policy_learning_commit.txt" \
    "$JOB_ROOT/code/droid_policy_learning" \
    "$ROOT/external_data/embodied_full/training_backends/droid_policy_learning" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
commit_marker = Path(sys.argv[2])
backend = Path(sys.argv[3])
source = Path(sys.argv[4])
commit = commit_marker.read_text(encoding="utf-8").strip()
payload = {
    "format_version": "qtail_droid_backend_marker_v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "verified",
    "source_root": str(source),
    "backend_root": str(backend),
    "git_commit": commit,
    "commit_marker": str(commit_marker),
    "commit_marker_sha256": hashlib.sha256(commit_marker.read_bytes()).hexdigest(),
    "git_fsck_passed": True,
    "claim_boundary": (
        "This binds the ORICO backend checkout to the recorded Git commit "
        "after git fsck. It does not claim that untracked runtime files are "
        "part of that commit."
    ),
}
temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
PY
}

require_fresh_direct_gate() {
  local phase="$1"
  local report="$2"
  if ! "$PYTHON" "$UNICLASH_GATE" \
    --guard "$UNICLASH_GUARD_STATUS" \
    --out "$report" \
    --expected-interface en1 \
    --max-age-seconds 10 \
    --quiet >> "$LOG" 2>&1; then
    log "fresh UniClash Core ON/TUN OFF/direct gate failed at $phase; aborting transition"
    exit 86
  fi
  log "fresh UniClash Core ON/TUN OFF/direct gate passed at $phase"
}

commit_checksum_marker() {
  "$PYTHON" - "$RESULT_ROOT/download_verification.json" \
    "$DOWNLOAD_MARKER" \
    "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

verification = Path(sys.argv[1])
download_marker = Path(sys.argv[2])
marker = Path(sys.argv[3])
payload = {
    "version": "droid_checksum_verified_marker_v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "verified",
    "download_verification": str(verification),
    "download_verification_bytes": verification.stat().st_size,
    "download_verification_sha256": hashlib.sha256(
        verification.read_bytes()
    ).hexdigest(),
    "download_completion_marker": str(download_marker),
    "download_completion_marker_bytes": download_marker.stat().st_size,
    "download_completion_marker_sha256": hashlib.sha256(
        download_marker.read_bytes()
    ).hexdigest(),
}
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
}

invalidate_final_qa() {
  rm -f \
    "$MARKER_ROOT/FINAL_PAGE_QA_PREVIEW" \
    "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" \
    "$MARKER_ROOT/DROID_PUBLIC_PROJECTION_COMMITTED" \
    "$MARKER_ROOT/DROID_POSTCOMMIT_PAGE_QA_COMPLETE" \
    "$RESULT_ROOT/final_page_qa.json" \
    "$RESULT_ROOT/final_page_desktop.png" \
    "$RESULT_ROOT/final_page_mobile.png" \
    "$RESULT_ROOT/final_page_postcommit_qa.json" \
    "$RESULT_ROOT/final_page_postcommit_desktop.png" \
    "$RESULT_ROOT/final_page_postcommit_mobile.png" \
    "$RESULT_ROOT/latest_final.json" \
    "$RESULT_ROOT/completion_audit_final.json" \
    "$RESULT_ROOT/download_progress_samples_final.json" \
    "$RESULT_ROOT/pipeline_timeline_final.json" \
    "$RESULT_ROOT/pipeline_timeline_final_verification.json" \
    "$RESULT_ROOT/droid_process_log_manifest.json" \
    "$RESULT_ROOT/uniclash_transport_guard_final.json"
  rm -rf "$RESULT_ROOT/process_logs_final"
  "$PYTHON" - "$RESULT_ROOT/droid_artifact_manifest.json" "$RESULT_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
result_root = Path(sys.argv[2])
if not manifest_path.is_file():
    raise SystemExit(0)
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
final_names = {
    "final_page_qa.json",
    "final_page_desktop.png",
    "final_page_mobile.png",
    "download_progress_samples_final.json",
    "pipeline_timeline_final.json",
    "pipeline_timeline_final_verification.json",
    "droid_process_log_manifest.json",
    "uniclash_transport_guard_final.json",
}
entries = []
for entry in payload.get("artifacts", []):
    path = Path(str(entry.get("path", "")))
    if path.name in final_names or result_root / "process_logs_final" in path.parents:
        continue
    entries.append(entry)
payload["artifacts"] = entries
temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest_path)
PY
  prune_code=$?
  if [ "$prune_code" -ne 0 ]; then
    log "failed to prune stale final artifacts from manifest"
    exit "$prune_code"
  fi
}

invalidate_postcommit_qa() {
  rm -f \
    "$MARKER_ROOT/DROID_POSTCOMMIT_PAGE_QA_COMPLETE" \
    "$RESULT_ROOT/final_page_postcommit_qa.json" \
    "$RESULT_ROOT/final_page_postcommit_desktop.png" \
    "$RESULT_ROOT/final_page_postcommit_mobile.png"
}

refresh_final_process_logs() {
  "$PYTHON" - "$JOB_ROOT" "$RESULT_ROOT" "$ROOT" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

job_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
repo_root = Path(sys.argv[3])
log_root = job_root / "logs"
snapshot_root = result_root / "process_logs_final"
snapshot_root.mkdir(parents=True, exist_ok=True)
required_sources = {
    "droid_full_pipeline.log": (
        log_root / "droid_full_pipeline.log",
        "full pipeline terminal evidence",
    ),
    "droid_feature_prewarm.log": (
        log_root / "droid_feature_prewarm.log",
        "full-record feature scan",
    ),
    "pipeline_watchdog.log": (
        log_root / "pipeline_watchdog.log",
        "pipeline supervision",
    ),
    "progress_loop.log": (
        log_root / "progress_loop.log",
        "continuous public status",
    ),
    "progress_refresh.log": (
        log_root / "progress_refresh.log",
        "stage transition status",
    ),
    "pipeline_generation_handoff.log": (
        log_root / "pipeline_generation_handoff.log",
        "download/checksum handoff",
    ),
    "manual_endpoint_generation_handoff.log": (
        log_root / "manual_endpoint_generation_handoff.log",
        "transport tuning handoff",
    ),
    "qtail-web-services.log": (
        log_root / "qtail-web-services.log",
        "dual-port DROID page supervision and recovery",
    ),
}
optional_sources = {
    "pipeline_watchdog_status.json": (
        log_root / "pipeline_watchdog_status.json",
        "last watchdog process snapshot",
    ),
    "qtail_droid_terminal_launcher.log": (
        log_root / "qtail_droid_terminal_launcher.log",
        "scheduled terminal launcher supervision",
    ),
    "qtail_droid_launchd_stderr.log": (
        log_root / "qtail_droid_launchd_stderr.log",
        "scheduled launcher stderr history",
    ),
    "qtail_droid_launchd_stdout.log": (
        log_root / "qtail_droid_launchd_stdout.log",
        "scheduled launcher stdout history",
    ),
    "qtail_uniclash_guard_stderr.log": (
        log_root / "qtail_uniclash_guard_stderr.log",
        "UniClash transport guard stderr history",
    ),
    "qtail_uniclash_guard_stdout.log": (
        log_root / "qtail_uniclash_guard_stdout.log",
        "UniClash transport guard stdout history",
    ),
    "qtail_web_services_local.log": (
        log_root / "qtail_web_services_local.log",
        "local web-service supervision history",
    ),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def entry(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

entries = []
missing_required = []
sources = [
    *((
        name,
        source,
        role,
        True,
    ) for name, (source, role) in required_sources.items()),
    *((
        name,
        source,
        role,
        False,
    ) for name, (source, role) in optional_sources.items()),
]
for name, source, role, required in sources:
    if not source.is_file() or (required and source.stat().st_size <= 0):
        if required:
            missing_required.append(str(source))
        continue
    destination = snapshot_root / name
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    text = destination.read_text(encoding="utf-8", errors="replace")
    entries.append({
        **entry(destination),
        "source": str(source),
        "role": role,
        "required": required,
        "line_count": len(text.splitlines()),
    })
if missing_required:
    raise SystemExit(
        "required final process logs are empty or missing: "
        + ", ".join(missing_required)
    )

process_manifest = result_root / "droid_process_log_manifest.json"
process_payload = {
    "status": "complete",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "contract": {
        "snapshot_is_immutable": True,
        "live_logs_continue_after_snapshot": True,
        "required_log_count": len(required_sources),
        "captured_required_log_count": sum(
            bool(item["required"]) for item in entries
        ),
        "optional_log_count": len(optional_sources),
        "captured_optional_log_count": sum(
            not bool(item["required"]) for item in entries
        ),
    },
    "missing_required": missing_required,
    "logs": entries,
}
process_tmp = process_manifest.with_name(
    f".{process_manifest.name}.{os.getpid()}.tmp"
)
process_tmp.write_text(json.dumps(process_payload, indent=2) + "\n")
os.replace(process_tmp, process_manifest)

artifact_manifest = result_root / "droid_artifact_manifest.json"
artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
replaced = {
    str(process_manifest),
    *(str(item["path"]) for item in entries),
}
retained = [
    item
    for item in artifact_payload.get("artifacts", [])
    if str(item.get("path", "")) not in replaced
]
artifact_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
artifact_payload["artifacts"] = sorted(
    [
        *retained,
        entry(process_manifest),
        *(entry(Path(item["path"])) for item in entries),
    ],
    key=lambda item: item["path"],
)
artifact_tmp = artifact_manifest.with_name(
    f".{artifact_manifest.name}.{os.getpid()}.tmp"
)
artifact_tmp.write_text(json.dumps(artifact_payload, indent=2) + "\n")
os.replace(artifact_tmp, artifact_manifest)
PY
}

assert_final_qa_contract_compatible() {
  "$PYTHON" - \
    "$ROOT/tools/qtail_verify_droid_page.mjs" \
    "$FINAL_QA_CONTRACT_BLOCKER" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

verifier = Path(sys.argv[1])
blocker = Path(sys.argv[2])
source = verifier.read_text(encoding="utf-8")
required_tokens = (
    'status: "preview_active"',
    "owner_pid: process.ppid",
    "previewCreated = true",
    '"--commit-bootstrap"',
    '"droid_final_page_qa_bootstrap_v1"',
    'expectedCompletion: "8 / 9"',
    'status: "qa_complete_waiting_final_commit"',
    'status: "already_complete_read_only"',
    '"--post-commit-read-only"',
    'expectedCompletion: "9 / 9"',
    'status: "postcommit_page_qa_complete"',
    "non-smoke final QA must be owned by qtail_orico_full_pipeline.sh",
    "the parent pipeline must invalidate it before QA",
    "await unlink(previewMarker).catch(() => {})",
)
missing = [token for token in required_tokens if token not in source]
if not missing:
    blocker.unlink(missing_ok=True)
    raise SystemExit(0)
payload = {
    "status": "blocked",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "reason": (
        "The page verifier lacks the lease-bound 8/9 sealing-to-committed "
        "marker contract required to prevent premature completion."
    ),
    "self_heal": (
        "The blocker is removed automatically when all required two-phase "
        "marker operations are present."
    ),
    "missing_contract_tokens": missing,
    "verifier": str(verifier),
    "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
}
temporary = blocker.with_name(f".{blocker.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, blocker)
raise SystemExit(1)
PY
}

terminate_process_tree() {
  local parent_pid="$1"
  local child_pid
  for child_pid in $(pgrep -P "$parent_pid" 2>/dev/null); do
    terminate_process_tree "$child_pid"
  done
  kill -TERM "$parent_pid" 2>/dev/null || true
}

wait_for_volume
mkdir -p "$DATA_ROOT" "$RESULT_ROOT" "$MARKER_ROOT" "$LOG_ROOT" "$JOB_ROOT/code" "$JOB_ROOT/data" || exit 10
LIVE_LOG_LINK="$RESULT_ROOT/live_logs"
if [ -L "$LIVE_LOG_LINK" ]; then
  if [ "$(readlink "$LIVE_LOG_LINK")" != "../../logs" ]; then
    log "live log link points to an unexpected target"
    exit 19
  fi
elif [ -e "$LIVE_LOG_LINK" ]; then
  log "live log path exists but is not a symlink"
  exit 19
else
  ln -s ../../logs "$LIVE_LOG_LINK" || exit 19
fi
if [ -L "$DOWNLOAD_MARKER_LINK" ]; then
  if [ "$(readlink "$DOWNLOAD_MARKER_LINK")" != "../../manifests/DROID_DOWNLOAD_COMPLETE" ]; then
    log "download marker link points to an unexpected target"
    exit 19
  fi
elif [ -e "$DOWNLOAD_MARKER_LINK" ]; then
  log "download marker link path exists but is not a symlink"
  exit 19
else
  ln -s ../../manifests/DROID_DOWNLOAD_COMPLETE "$DOWNLOAD_MARKER_LINK" || exit 19
fi

while ! ln -s "$$" "$LOCK_DIR" 2>/dev/null; do
  if [ -L "$LOCK_DIR" ]; then
    lock_pid="$(readlink "$LOCK_DIR" 2>/dev/null || true)"
  else
    lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  lock_command="$(ps -p "$lock_pid" -o command= 2>/dev/null || true)"
  if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
    printf '[%s] pipeline lock is owned by live pid=%s command=%s; refusing to steal\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$lock_pid" "$lock_command" >> "$LOG"
    exit 0
  fi
  rm -rf "$LOCK_DIR" || exit 18
done
cleanup_lock() {
  if [ -L "$LOCK_DIR" ] && [ "$(readlink "$LOCK_DIR" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$LOCK_DIR"
  fi
}
terminate_pipeline() {
  cleanup_lock
  exit 143
}
trap cleanup_lock EXIT
trap terminate_pipeline INT TERM HUP

if ! write_pipeline_started_marker; then
  log "semantic pipeline-start marker could not be committed"
  exit 18
fi
if ! "$PYTHON" "$ORCHESTRATION_SNAPSHOT_PUBLISHER" \
  --repo-root "$ROOT" \
  --job-root "$JOB_ROOT" \
  --include "$ROOT/tools/qtail_prewarm_status_contract_selftest.py" \
  >> "$LOG" 2>&1; then
  log "atomic ORICO orchestration snapshot publication failed"
  exit 18
fi
refresh_status

if [ -f "$MARKER_ROOT/DROID_TRAINING_COMPLETE" ]; then
  if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
    --job-root "$JOB_ROOT" --stage training >> "$LOG" 2>&1; then
    log "stored training marker is stale or unbound; invalidating dependent markers"
    rm -f \
      "$MARKER_ROOT/DROID_TRAINING_COMPLETE" \
      "$MARKER_ROOT/FINAL_PAGE_QA_PREVIEW" \
      "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" \
      "$MARKER_ROOT/DROID_PUBLIC_PROJECTION_COMMITTED"
  fi
fi
if [ -f "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" ]; then
  if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
    --job-root "$JOB_ROOT" --stage final >> "$LOG" 2>&1; then
    if "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
      --job-root "$JOB_ROOT" --stage final \
      --validate-projection >> "$LOG" 2>&1; then
      log "stored 9/9 projection is valid; resuming postcommit browser QA"
      invalidate_postcommit_qa
    else
      log "stored final QA marker is stale or unbound; invalidating it"
      invalidate_final_qa
    fi
  fi
elif [ -f "$MARKER_ROOT/FINAL_PAGE_QA_PREVIEW" ] \
  || [ -f "$RESULT_ROOT/final_page_qa.json" ]; then
  log "incomplete prior final QA state detected; invalidating it for a clean rerun"
  invalidate_final_qa
fi
refresh_status

if [ -f "$MARKER_ROOT/DROID_TRAINING_COMPLETE" ] \
  && [ -f "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" ] \
  && [ -f "$MARKER_ROOT/DROID_PUBLIC_PROJECTION_COMMITTED" ] \
  && [ -f "$MARKER_ROOT/DROID_POSTCOMMIT_PAGE_QA_COMPLETE" ] \
  && "$PYTHON" - "$RESULT_ROOT/completion_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("status") != "complete" or audit.get("passed_requirements") != 9:
    raise SystemExit(1)
PY
then
  log "pipeline already complete; keeping status service alive"
  while true; do
    refresh_status
    sleep 300
  done
fi

if [ -f "$MARKER_ROOT/OPENX_MIGRATION_COMPLETE" ] \
  && ! validate_openx_migration_marker; then
  log "stored Open X migration marker is legacy, stale, or unbound; rebuilding it from live ORICO state"
  rm -f "$MARKER_ROOT/OPENX_MIGRATION_COMPLETE"
fi

if [ ! -f "$MARKER_ROOT/OPENX_MIGRATION_COMPLETE" ]; then
  migration_source=""
  if [ -L "$OPENX_SOURCE" ]; then
    if [ "$(readlink "$OPENX_SOURCE")" != "$OPENX_TARGET" ]; then
      log "Open X source is an unexpected symlink"
      exit 11
    fi
    if [ -d "$OPENX_BACKUP" ]; then
      migration_source="$OPENX_BACKUP"
      log "resuming interrupted Open X move from local migration backup"
    elif [ ! -d "$OPENX_TARGET" ]; then
      log "Open X source is a broken symlink and no migration backup exists"
      exit 12
    fi
  else
    if [ -d "$OPENX_BACKUP" ] && [ ! -e "$OPENX_SOURCE" ]; then
      migration_source="$OPENX_BACKUP"
      log "resuming interrupted Open X move from local migration backup"
    elif [ -d "$OPENX_SOURCE" ]; then
      migration_source="$OPENX_SOURCE"
      log "moving existing Open X Strong data to ORICO"
    elif [ -d "$OPENX_TARGET" ]; then
      migration_source=""
      log "Open X data is already on ORICO; restoring the workspace symlink"
    else
      log "Open X source data is missing"
      exit 13
    fi
  fi

  if [ -n "$migration_source" ]; then
    mkdir -p "$OPENX_TARGET" || exit 14
    /usr/bin/rsync -a --partial "$migration_source/" "$OPENX_TARGET/" >> "$LOG" 2>&1
    if [ "$?" -ne 0 ]; then
      log "Open X rsync failed; local migration backup remains intact"
      exit 15
    fi
    if /usr/bin/rsync -a --dry-run --delete "$migration_source/" "$OPENX_TARGET/" | grep -q .; then
      log "Open X migration verification found differences; retrying later"
      exit 16
    fi
  fi

  if [ -d "$OPENX_SOURCE" ] && [ ! -L "$OPENX_SOURCE" ]; then
    mv "$OPENX_SOURCE" "$OPENX_BACKUP"
  fi
  if [ ! -L "$OPENX_SOURCE" ]; then
    ln -s "$OPENX_TARGET" "$OPENX_SOURCE"
  fi
  if [ "$(readlink "$OPENX_SOURCE")" != "$OPENX_TARGET" ] || [ ! -d "$OPENX_SOURCE" ]; then
    log "Open X workspace symlink verification failed"
    exit 17
  fi
  if [ -d "$OPENX_BACKUP" ]; then
    rm -rf "$OPENX_BACKUP"
  fi
  if ! write_openx_migration_marker; then
    log "Open X semantic migration marker could not be committed"
    exit 18
  fi
  log "Open X move verified; workspace path now points to ORICO"
  log "Open X and DROID remain separate datasets; DROID rsync will skip only matching DROID objects"
fi
refresh_status

if [ -f "$MARKER_ROOT/DROID_BACKEND_READY" ] \
  && ! validate_droid_backend_marker; then
  log "stored DROID backend marker is legacy, stale, or unbound; rebuilding the audited ORICO checkout"
  rm -f "$MARKER_ROOT/DROID_BACKEND_READY"
fi

if [ ! -f "$MARKER_ROOT/DROID_BACKEND_READY" ]; then
  log "copying DROID policy-learning repository to ORICO"
  if ! rsync -a --delete \
    "$ROOT/external_data/embodied_full/training_backends/droid_policy_learning/" \
    "$JOB_ROOT/code/droid_policy_learning/" >> "$LOG" 2>&1; then
    log "DROID backend copy failed; readiness marker withheld"
    exit 19
  fi
  if ! git -C "$JOB_ROOT/code/droid_policy_learning" fsck --no-progress >> "$LOG" 2>&1; then
    log "DROID backend git fsck failed; readiness marker withheld"
    exit 20
  fi
  if ! backend_commit="$(
    git -C "$JOB_ROOT/code/droid_policy_learning" rev-parse HEAD 2>> "$LOG"
  )"; then
    log "DROID backend commit lookup failed; readiness marker withheld"
    exit 21
  fi
  if ! printf '%s\n' "$backend_commit" | grep -Eq '^[0-9a-f]{40}$'; then
    log "DROID backend commit audit failed; readiness marker withheld"
    exit 21
  fi
  printf '%s\n' "$backend_commit" > "$MARKER_ROOT/droid_policy_learning_commit.txt.tmp"
  mv "$MARKER_ROOT/droid_policy_learning_commit.txt.tmp" \
    "$MARKER_ROOT/droid_policy_learning_commit.txt"
  if ! write_droid_backend_marker; then
    log "DROID backend semantic readiness marker could not be committed"
    exit 21
  fi
  log "DROID backend copied and git fsck passed"
fi
refresh_status

if [ ! -x "$GSUTIL" ]; then
  log "external gsutil is missing: $GSUTIL"
  exit 13
fi

if [ -f "$MARKER_ROOT/DROID_SOURCE_PROBED" ] \
  && ! validate_droid_source_probe_marker; then
  log "stored DROID source marker is legacy, stale, or unbound; rebuilding it without a network probe when the report remains valid"
  rm -f "$MARKER_ROOT/DROID_SOURCE_PROBED"
fi

if [ ! -f "$MARKER_ROOT/DROID_SOURCE_PROBED" ] \
  && [ -f "$SOURCE_PROBE_REPORT" ]; then
  if ! "$PYTHON" "$ROOT/tools/qtail_probe_droid_source.py" \
    --gsutil "$GSUTIL" \
    --source "$REMOTE_URI" \
    --job-root "$JOB_ROOT" \
    --out "$SOURCE_PROBE_REPORT" \
    --expected-bytes "$REMOTE_BYTES" \
    --marker-dir "$MARKER_ROOT" \
    --seal-existing >> "$LOG" 2>&1; then
    log "stored DROID source report could not be sealed; a fresh direct-route probe is required"
  fi
fi

if [ ! -f "$MARKER_ROOT/DROID_SOURCE_PROBED" ]; then
  log "probing official DROID source size and ORICO capacity"
  if ! "$PYTHON" "$ROOT/tools/qtail_probe_droid_source.py" \
    --gsutil "$GSUTIL" \
    --source "$REMOTE_URI" \
    --job-root "$JOB_ROOT" \
    --out "$SOURCE_PROBE_REPORT" \
    --expected-bytes "$REMOTE_BYTES" \
    --marker-dir "$MARKER_ROOT" >> "$LOG" 2>&1; then
    log "official DROID source probe failed; source marker withheld"
    exit 22
  fi
  log "official DROID source probe passed"
fi
if ! validate_droid_source_probe_marker; then
  log "semantic DROID source marker validation failed"
  exit 22
fi
refresh_status

log "running immutable DROID download-marker positive/negative controls"
if ! "$PYTHON" "$DOWNLOAD_MARKER_SELFTEST" \
  --verifier "$DOWNLOAD_MARKER_VERIFIER" \
  --python "$PYTHON" \
  --out "$DOWNLOAD_MARKER_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "DROID download-marker self-test failed; download handoff withheld"
  exit 22
fi

log "running final mirror-verifier positive/negative controls"
if ! "$PYTHON" "$MIRROR_VERIFIER_SELFTEST" \
  --verifier "$MIRROR_VERIFIER" \
  --python "$PYTHON" \
  --out "$MIRROR_VERIFIER_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "DROID mirror-verifier self-test failed; checksum handoff withheld"
  exit 22
fi

log "running runtime process and generation-handoff destructive controls"
if ! "$PYTHON" "$RUNTIME_PROCESS_SELFTEST" \
  --out "$RUNTIME_PROCESS_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "DROID runtime process contract self-test failed; pipeline withheld"
  exit 22
fi

log "running UniClash pre-checksum transport-gate destructive controls"
if ! "$PYTHON" "$UNICLASH_GATE_SELFTEST" \
  --out "$UNICLASH_GATE_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "UniClash pre-checksum gate self-test failed; pipeline withheld"
  exit 22
fi

log "rebuilding deterministic downloader single-writer controls"
if ! "$PYTHON" "$DOWNLOADER_SELFTEST" \
  --downloader "$ROOT/tools/qtail_parallel_gcs_download.py" \
  --out "$DOWNLOADER_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "DROID downloader single-writer controls failed; pipeline withheld"
  exit 22
fi

log "rebuilding deterministic UniClash v6 interface-bound classifier controls"
if ! "$PYTHON" "$ROOT/tools/qtail_uniclash_transport_guard.py" \
  --status "$UNICLASH_GUARD_STATUS" \
  --expected-interface en1 \
  --once \
  --no-terminate \
  --no-restart-uniclash \
  --classifier-selftest-out "$CLASSIFIER_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "UniClash v6 classifier controls failed; pipeline withheld"
  exit 22
fi

log "rebuilding pipeline hardening positive/negative controls"
if ! "$PYTHON" "$STAGE_HARDENING_SELFTEST" \
  --out "$STAGE_HARDENING_SELFTEST_REPORT" >> "$LOG" 2>&1 \
  || ! "$PYTHON" "$PREVIEW_SELFTEST" \
  --out "$PREVIEW_SELFTEST_REPORT" >> "$LOG" 2>&1 \
  || ! "$PYTHON" "$MANIFEST_SELFTEST" \
  --out "$MANIFEST_SELFTEST_REPORT" >> "$LOG" 2>&1 \
  || ! "$PYTHON" "$SHELL_CONTRACT_SELFTEST" \
  --pipeline "$ROOT/scripts/qtail_orico_full_pipeline.sh" \
  --out "$SHELL_CONTRACT_SELFTEST_REPORT" >> "$LOG" 2>&1; then
  log "pipeline hardening controls failed; pipeline withheld"
  exit 22
fi

if [ -f "$DOWNLOAD_MARKER" ]; then
  if ! "$PYTHON" "$DOWNLOAD_MARKER_VERIFIER" \
    --data-dir "$DATA_ROOT" \
    --manifest "$OBJECT_MANIFEST" \
    --checksum-manifest "$CHECKSUM_MANIFEST" \
    --checksum-ledger "$CHECKSUM_LEDGER" \
    --transport-status "$RESULT_ROOT/parallel_download_status.json" \
    --marker "$DOWNLOAD_MARKER" \
    --expected-bytes "$REMOTE_BYTES" >> "$LOG" 2>&1; then
    log "stored download marker is stale or unbound; invalidating dependent markers"
    rm -f \
      "$DOWNLOAD_MARKER" \
      "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" \
      "$MARKER_ROOT/DROID_TRAINING_COMPLETE" \
      "$MARKER_ROOT/FINAL_PAGE_QA_PREVIEW" \
      "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" \
      "$MARKER_ROOT/DROID_PUBLIC_PROJECTION_COMMITTED"
  fi
fi

if [ ! -f "$DOWNLOAD_MARKER" ]; then
  if [ ! -f "$OBJECT_MANIFEST" ]; then
    log "building official DROID object manifest"
    if ! "$PYTHON" "$ROOT/tools/qtail_build_droid_object_manifest.py" \
      --gsutil "$GSUTIL" \
      --source "$REMOTE_URI" \
      --out "$OBJECT_MANIFEST" \
      --expected-bytes "$REMOTE_BYTES" >> "$LOG" 2>&1; then
      log "official DROID object-manifest build failed"
      exit 23
    fi
  fi
  if [ ! -f "$CHECKSUM_MANIFEST" ]; then
    log "building official DROID MD5/CRC32C manifest"
    if ! "$PYTHON" "$ROOT/tools/qtail_build_droid_checksum_manifest.py" \
      --size-manifest "$OBJECT_MANIFEST" \
      --out "$CHECKSUM_MANIFEST" >> "$LOG" 2>&1; then
      log "official DROID checksum-manifest build failed"
      exit 24
    fi
  fi
  if [ ! -f "$LIVE_PARTIAL_SELFTEST_REPORT" ] \
    && [ -f "$CHECKSUM_LEDGER" ] \
    && [ -f "$RESULT_ROOT/parallel_download_status.json" ]; then
    log "capturing live partial-mirror completion-marker rejection"
    if ! "$PYTHON" "$LIVE_PARTIAL_SELFTEST" \
      --verifier "$DOWNLOAD_MARKER_VERIFIER" \
      --python "$PYTHON" \
      --data-dir "$DATA_ROOT" \
      --manifest "$OBJECT_MANIFEST" \
      --checksum-manifest "$CHECKSUM_MANIFEST" \
      --checksum-ledger "$CHECKSUM_LEDGER" \
      --transport-status "$RESULT_ROOT/parallel_download_status.json" \
      --expected-bytes "$REMOTE_BYTES" \
      --out "$LIVE_PARTIAL_SELFTEST_REPORT" >> "$LOG" 2>&1; then
      log "live partial-mirror rejection control could not be rebuilt"
      exit 24
    fi
  elif [ ! -f "$LIVE_PARTIAL_SELFTEST_REPORT" ]; then
    log "live partial-mirror rejection history is unavailable before the first downloader heartbeat; it is optional and does not block formal closure"
  fi
  log "starting resumable parallel HTTPS download from official DROID manifest"
  log "download route=$DOWNLOAD_PROXY interface=$DOWNLOAD_INTERFACE workers=$DOWNLOAD_WORKERS primary_endpoints=$DOWNLOAD_PRIMARY_ENDPOINTS reserve_free_bytes=$DOWNLOAD_RESERVE_FREE_BYTES"
  while true; do
    wait_for_volume
    "$PYTHON" "$ROOT/tools/qtail_parallel_gcs_download.py" \
      --manifest "$OBJECT_MANIFEST" \
      --checksum-manifest "$CHECKSUM_MANIFEST" \
      --checksum-ledger "$CHECKSUM_LEDGER" \
      --checksum-quarantine "$CHECKSUM_QUARANTINE" \
      --target "$DATA_ROOT" \
      --status "$RESULT_ROOT/parallel_download_status.json" \
      --process-lock "$RESULT_ROOT/.qtail_parallel_gcs_download.lock" \
      --workers "$DOWNLOAD_WORKERS" \
      --heartbeat-seconds 15 \
      --stall-timeout-seconds 1800 \
      --chunk-mib 64 \
      --primary-endpoints "$DOWNLOAD_PRIMARY_ENDPOINTS" \
      --required-mount /Volumes/ORICO \
      --reserve-free-bytes "$DOWNLOAD_RESERVE_FREE_BYTES" \
      --forbid-tunnel-route \
      --expected-interface "$DOWNLOAD_INTERFACE" \
      --proxy "$DOWNLOAD_PROXY" >> "$LOG" 2>&1
    code=$?
    refresh_status
    if [ "$code" -eq 0 ]; then
      if "$PYTHON" "$DOWNLOAD_MARKER_VERIFIER" \
        --data-dir "$DATA_ROOT" \
        --manifest "$OBJECT_MANIFEST" \
        --checksum-manifest "$CHECKSUM_MANIFEST" \
        --checksum-ledger "$CHECKSUM_LEDGER" \
        --transport-status "$RESULT_ROOT/parallel_download_status.json" \
        --marker "$DOWNLOAD_MARKER" \
        --expected-bytes "$REMOTE_BYTES" \
        --write >> "$LOG" 2>&1; then
        log "initial DROID download completed with immutable 4,102-object binding"
        break
      else
        code=$?
        log "DROID downloader returned 0 but immutable handoff marker failed with code $code"
      fi
    fi
    log "DROID resumable HTTPS downloader exited $code; retrying in 120 seconds"
    sleep 120
  done
fi

if [ ! -f "$MARKER_ROOT/DROID_TRAINING_COMPLETE" ]; then
  if ! require_pipeline_generation_marker "pre-checksum"; then
    log "pipeline PID/lock/script generation gate blocked checksum"
    exit 88
  fi
fi

log "auditing official DROID 1.0.0/1.0.1 release metadata and shared schema"
if ! "$PYTHON" "$ROOT/tools/qtail_audit_droid_release_metadata.py" \
  --data-dir "$DATA_ROOT" \
  --checksum-manifest "$CHECKSUM_MANIFEST" \
  --out "$RELEASE_METADATA_AUDIT" >> "$LOG" 2>&1; then
  log "official DROID release metadata audit failed; checksum and training withheld"
  exit 25
fi
log "official DROID release metadata audit passed: 4,096 shards / 187,891 records"
refresh_status

if [ -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then
  wait_for_volume
  if ! "$PYTHON" - "$JOB_ROOT" "$ROOT/tools" <<'PY' >> "$LOG" 2>&1
import json
import sys
from pathlib import Path

job_root = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from qtail_verify_droid_stage_markers import validate_checksum_marker

errors = validate_checksum_marker(job_root)
verification = json.loads(
    (
        job_root
        / "results"
        / "qtail_droid_full"
        / "download_verification.json"
    ).read_text(encoding="utf-8")
)
byte_checksum_verified = bool(
    verification.get("local_md5_rehash_complete") is True
    or int(verification.get("checksum_rsync_returncode", -1)) == 0
)
if (
    verification.get("status") != "complete"
    or verification.get("ready_for_full_allocation_training") is not True
    or not byte_checksum_verified
    or int(verification.get("checksum_error_count", -1)) != 0
):
    errors.append("bound prior byte-checksum verification is incomplete")
if errors:
    raise SystemExit("; ".join(errors))
PY
  then
    log "stored checksum marker or its prior full-MD5 evidence is invalid; invalidating it"
    rm -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED"
  elif ! "$PYTHON" "$MIRROR_VERIFIER" \
    --data-dir "$DATA_ROOT" \
    --manifest "$OBJECT_MANIFEST" \
    --checksum-manifest "$CHECKSUM_MANIFEST" \
    --checksum-ledger "$CHECKSUM_LEDGER" \
    --expected-bytes "$REMOTE_BYTES" \
    --checksum-returncode 0 \
    --out "$CHECKSUM_STAT_CONTINUITY_REPORT" >> "$LOG" 2>&1; then
    log "stored byte-checksum evidence no longer has 4,102-file stat continuity; invalidating it"
    rm -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED"
  elif ! "$PYTHON" - \
    "$CHECKSUM_STAT_CONTINUITY_REPORT" \
    "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" \
    "$RESULT_ROOT/download_verification.json" <<'PY' >> "$LOG" 2>&1
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

continuity_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
verification_path = Path(sys.argv[3])
payload = json.loads(continuity_path.read_text(encoding="utf-8"))
payload["continuity_contract"] = {
    "version": "bound_byte_checksum_plus_4102_stat_identity_v1",
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "prior_checksum_marker": str(marker_path),
    "prior_checksum_marker_sha256": hashlib.sha256(
        marker_path.read_bytes()
    ).hexdigest(),
    "prior_download_verification": str(verification_path),
    "prior_download_verification_sha256": hashlib.sha256(
        verification_path.read_bytes()
    ).hexdigest(),
    "current_stat_identity_passed": payload.get("status") == "complete",
}
temporary = continuity_path.with_name(
    f".{continuity_path.name}.tmp.{os.getpid()}"
)
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, continuity_path)
PY
  then
    log "checksum stat-continuity evidence could not be bound; invalidating marker"
    rm -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED"
  else
    require_fresh_direct_gate \
      "existing checksum-marker fast path" \
      "$UNICLASH_CHECKSUM_HANDOFF_GATE_REPORT"
    log "QTAIL_TERMINAL checksum_complete path=bound_byte_checksum_plus_stat_continuity"
  fi
fi

if [ ! -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then
  log "running checksum comparison against official DROID bucket"
  while true; do
    wait_for_volume
    if ! "$PYTHON" "$UNICLASH_GATE" \
      --guard "$UNICLASH_GUARD_STATUS" \
      --out "$UNICLASH_GATE_REPORT" \
      --expected-interface en1 \
      --max-age-seconds 10 \
      --quiet >> "$LOG" 2>&1; then
      log "UniClashCore/direct-route pre-checksum gate is not ready; gsutil remains withheld for 30 seconds"
      refresh_status
      sleep 30
      continue
    fi
    log "UniClashCore is running and checksum transport preflight passed on en1"
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      NO_PROXY="*" no_proxy="*" TMPDIR="$JOB_ROOT/tmp" "$GSUTIL" -m \
      -o "GSUtil:parallel_process_count=1" \
      -o "GSUtil:parallel_thread_count=12" \
      rsync -r -c "$REMOTE_URI" "$DATA_ROOT" >> "$LOG" 2>&1 &
    checksum_pid=$!
    checksum_guard_failed=0
    while kill -0 "$checksum_pid" 2>/dev/null; do
      sleep 2
      if ! "$PYTHON" "$UNICLASH_GATE" \
        --guard "$UNICLASH_GUARD_STATUS" \
        --out "$UNICLASH_GATE_REPORT" \
        --expected-interface en1 \
        --max-age-seconds 6 \
        --quiet >> "$LOG" 2>&1; then
        log "UniClash transport guard heartbeat failed during checksum; terminating gsutil process tree"
        terminate_process_tree "$checksum_pid"
        checksum_guard_failed=1
        break
      fi
    done
    wait "$checksum_pid"
    checksum_code=$?
    if [ "$checksum_guard_failed" -eq 1 ]; then
      checksum_code=86
    fi
    refresh_status
    if [ "$checksum_code" -eq 0 ]; then
      break
    fi
    log "DROID checksum rsync exited $checksum_code; retrying in 120 seconds"
    sleep 120
  done

  wait_for_volume
  "$PYTHON" "$ROOT/tools/qtail_cleanup_droid_transport_artifacts.py" \
    --data-dir "$DATA_ROOT" \
    --manifest "$RESULT_ROOT/droid_object_manifest.json" \
    --quarantine-dir "$RESULT_ROOT/transport_quarantine" \
    --out "$RESULT_ROOT/droid_transport_cleanup_audit.json" >> "$LOG" 2>&1
  cleanup_code=$?
  if [ "$cleanup_code" -ne 0 ]; then
    log "DROID transport-artifact cleanup audit exited $cleanup_code"
    exit "$cleanup_code"
  fi

  "$PYTHON" "$MIRROR_VERIFIER" \
    --data-dir "$DATA_ROOT" \
    --manifest "$OBJECT_MANIFEST" \
    --checksum-manifest "$CHECKSUM_MANIFEST" \
    --checksum-ledger "$CHECKSUM_LEDGER" \
    --expected-bytes "$REMOTE_BYTES" \
    --checksum-returncode "$checksum_code" \
    --out "$RESULT_ROOT/download_verification.json"
  verification_code=$?
  if [ "$verification_code" -ne 0 ]; then
    rm -f "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE"
    log "DROID manifest verification failed with code $verification_code; download marker invalidated so the official MD5 ledger is rebuilt on restart"
    exit "$verification_code"
  fi
  if ! commit_checksum_marker; then
    log "atomic checksum marker commit failed; training handoff aborted"
    exit 87
  fi
  require_fresh_direct_gate \
    "checksum completion handoff" \
    "$UNICLASH_CHECKSUM_HANDOFF_GATE_REPORT"
  log "DROID checksum and local completeness gates passed"
  log "QTAIL_TERMINAL checksum_complete path=full_checksum"
fi
refresh_status

log "running DROID timeline monotonic-counter controls"
"$PYTHON" "$ROOT/tools/qtail_droid_timeline_monotonic_selftest.py" \
  --out "$RESULT_ROOT/droid_timeline_monotonic_selftest.json" >> "$LOG" 2>&1
timeline_selftest_code=$?
if [ "$timeline_selftest_code" -ne 0 ]; then
  log "DROID timeline monotonic-counter self-test exited $timeline_selftest_code; training withheld"
  exit "$timeline_selftest_code"
fi

log "running deterministic DROID protocol positive/negative controls"
"$PYTHON" "$ROOT/tools/qtail_droid_protocol_selftest.py" \
  --pt-source "$ROOT/data/uploaded_data.csv" \
  --out "$RESULT_ROOT/droid_protocol_selftest.json" >> "$LOG" 2>&1
selftest_code=$?
if [ "$selftest_code" -ne 0 ]; then
  log "DROID protocol self-test exited $selftest_code; training withheld"
  exit "$selftest_code"
fi

log "running DROID prewarm status-scope positive/negative controls"
"$PYTHON" "$ROOT/tools/qtail_prewarm_status_contract_selftest.py" \
  --out "$RESULT_ROOT/droid_prewarm_status_contract_selftest.json" \
  >> "$LOG" 2>&1
prewarm_status_selftest_code=$?
if [ "$prewarm_status_selftest_code" -ne 0 ]; then
  log "DROID prewarm status-scope self-test exited $prewarm_status_selftest_code; training withheld"
  exit "$prewarm_status_selftest_code"
fi

log "running formal DROID pre-optimizer gate-order controls"
"$PYTHON" "$TRAINING_GATE_ORDER_SELFTEST" \
  --trainer "$ROOT/tools/qtail_train_droid_full.py" \
  --out "$TRAINING_GATE_ORDER_SELFTEST_REPORT" >> "$LOG" 2>&1
training_gate_order_code=$?
if [ "$training_gate_order_code" -ne 0 ]; then
  log "DROID training gate-order self-test exited $training_gate_order_code; training withheld"
  exit "$training_gate_order_code"
fi

if [ ! -f "$MARKER_ROOT/DROID_TRAINING_COMPLETE" ]; then
  log "running DROID environment contract positive/negative controls"
  "$PYTHON" "$ROOT/tools/qtail_droid_environment_contract_selftest.py" \
    --repo-root "$ROOT" \
    --job-root "$JOB_ROOT" \
    --out "$RESULT_ROOT/droid_environment_contract_selftest.json" \
    --pt-source "$ROOT/data/uploaded_data.csv" \
    --object-manifest "$RESULT_ROOT/droid_object_manifest.json" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --transport-status "$RESULT_ROOT/parallel_download_status.json" \
    --uniclash-guard-status "$RESULT_ROOT/uniclash_transport_guard.json" \
    --orchestration-snapshot-manifest "$ORCHESTRATION_SNAPSHOT_MANIFEST" \
    --backend-root "$JOB_ROOT/code/droid_policy_learning" \
    >> "$LOG" 2>&1
  environment_selftest_code=$?
  if [ "$environment_selftest_code" -ne 0 ]; then
    log "DROID environment contract self-test exited $environment_selftest_code; training withheld"
    exit "$environment_selftest_code"
  fi

  if ! require_pipeline_generation_marker "pre-environment"; then
    log "pipeline PID/lock/script generation gate blocked environment capture"
    exit 88
  fi
  require_fresh_direct_gate \
    "environment capture" \
    "$UNICLASH_PRE_ENVIRONMENT_GATE_REPORT"
  log "capturing secret-free DROID training environment manifest"
  "$PYTHON" "$ROOT/tools/qtail_capture_droid_environment.py" \
    --repo-root "$ROOT" \
    --job-root "$JOB_ROOT" \
    --out "$RESULT_ROOT/droid_environment_manifest.json" \
    --pt-source "$ROOT/data/uploaded_data.csv" \
    --object-manifest "$RESULT_ROOT/droid_object_manifest.json" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --download-verification "$RESULT_ROOT/download_verification.json" \
    --transport-status "$RESULT_ROOT/parallel_download_status.json" \
    --uniclash-guard-status "$RESULT_ROOT/uniclash_transport_guard.json" \
    --backend-root "$JOB_ROOT/code/droid_policy_learning" \
    --orchestration-snapshot-manifest "$ORCHESTRATION_SNAPSHOT_MANIFEST" \
    --require-final-inputs >> "$LOG" 2>&1
  environment_code=$?
  if [ "$environment_code" -ne 0 ]; then
    log "DROID environment manifest exited $environment_code; training withheld"
    exit "$environment_code"
  fi
fi

PREWARM_LOOP="$ROOT/scripts/qtail_droid_feature_prewarm_loop.sh"
while pgrep -f -x "/bin/zsh $PREWARM_LOOP" >/dev/null 2>&1; do
  log "waiting for feature prewarm loop to exit before formal training"
  sleep 60
done
while pgrep -f "$ROOT/tools/qtail_train_droid_full.py" >/dev/null 2>&1; do
  log "waiting for every feature-prewarm trainer process to exit before formal training"
  sleep 60
done

if [ ! -f "$MARKER_ROOT/DROID_TRAINING_COMPLETE" ]; then
  if ! require_pipeline_generation_marker "pre-formal-training"; then
    log "pipeline PID/lock/script generation gate blocked formal training"
    exit 88
  fi
  require_fresh_direct_gate \
    "formal training launch" \
    "$UNICLASH_PRE_TRAINING_GATE_REPORT"
  log "starting equal-compute source vs Q-Tail DROID full allocation training"
  "$PYTHON" "$ROOT/tools/qtail_train_droid_full.py" \
    --data-dir "$DATA_ROOT" \
    --out "$RESULT_ROOT" \
    --marker-dir "$MARKER_ROOT" \
    --object-manifest "$RESULT_ROOT/droid_object_manifest.json" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --checksum-ledger "$RESULT_ROOT/droid_object_checksum_ledger.json" \
    --checksum-stat-continuity "$CHECKSUM_STAT_CONTINUITY_REPORT" \
    --transport-status "$RESULT_ROOT/parallel_download_status.json" \
    --download-marker "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE" \
    --download-verification "$RESULT_ROOT/download_verification.json" \
    --environment-manifest "$RESULT_ROOT/droid_environment_manifest.json" \
    --require-verified-mirror \
    --process-lock "$RESULT_ROOT/.qtail_train_droid_full.lock" \
    --required-mount /Volumes/ORICO \
    --steps 20000 \
    --records-per-shard 0 \
    --min-shards 64 \
    --min-record-parse-rate 1.0 \
    --min-record-scan-complete-rate 1.0 \
    --status-every-shards 10 \
    --checkpoint-every-steps 5000 \
    --seed 11 \
    --bootstrap-samples 5000 \
    --holdout-fraction 0.20 \
    --pt-source "$ROOT/data/uploaded_data.csv" >> "$LOG" 2>&1
  training_code=$?
  if [ "$training_code" -ne 0 ]; then
    log "DROID full allocation training exited $training_code; completion marker withheld"
    exit "$training_code"
  fi
  "$PYTHON" - \
    "$RESULT_ROOT/droid_feature_cache_manifest.json" \
    "$RESULT_ROOT/droid_feature_cache_verification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
verification = Path(sys.argv[2])
if not manifest.is_file() or not verification.is_file():
    raise SystemExit(1)
audit = json.loads(verification.read_text(encoding="utf-8"))
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
if not (
    audit.get("status") == "verified"
    and audit.get("cache_manifest_sha256") == manifest_sha256
    and audit.get("all_official_tfrecords") is True
    and audit.get("full_official_record_count_match") is True
    and audit.get("all_feature_values_recomputed") is True
    and int(audit.get("official_tfrecord_count", -1)) == 4_096
    and int(audit.get("verified_cache_count", -1)) == 4_096
    and int(audit.get("recomputed_feature_count", -1)) == 4_096
    and int(audit.get("official_expected_records", -1)) == 187_891
    and int(audit.get("verified_decoded_records", -1)) == 187_891
    and int(audit.get("error_count", -1)) == 0
):
    raise SystemExit(1)
PY
  cache_verification_code=$?
  if [ "$cache_verification_code" -eq 0 ]; then
    log "reusing full feature-value recomputation bound to the unchanged cache manifest"
  else
    "$PYTHON" "$ROOT/tools/qtail_verify_droid_feature_cache.py" \
      --data-dir "$DATA_ROOT" \
      --object-manifest "$RESULT_ROOT/droid_object_manifest.json" \
      --cache-manifest "$RESULT_ROOT/droid_feature_cache_manifest.json" \
      --out "$RESULT_ROOT/droid_feature_cache_verification.json" \
      --artifact-manifest "$RESULT_ROOT/droid_artifact_manifest.json" \
      --require-all-official-tfrecords \
      --recompute-feature-values >> "$LOG" 2>&1
    cache_verification_code=$?
  fi
  if [ "$cache_verification_code" -ne 0 ]; then
    log "DROID full feature-cache verification exited $cache_verification_code"
    exit "$cache_verification_code"
  fi
  "$PYTHON" "$ROOT/tools/qtail_audit_droid_incremental_closure.py" \
    --data-dir "$DATA_ROOT" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --checksum-ledger "$RESULT_ROOT/droid_object_checksum_ledger.json" \
    --cache-manifest "$RESULT_ROOT/droid_feature_cache_manifest.json" \
    --record-audit "$RESULT_ROOT/droid_feature_cache_verification.json" \
    --require-formal \
    --out "$RESULT_ROOT/droid_incremental_closure_audit.json" >> "$LOG" 2>&1
  closure_code=$?
  if [ "$closure_code" -ne 0 ]; then
    log "DROID full MD5/record/cache closure exited $closure_code"
    exit "$closure_code"
  fi
  log "QTAIL_TERMINAL record_closure_complete records=187891"
  "$PYTHON" "$ROOT/tools/qtail_droid_incremental_closure_selftest.py" \
    --auditor "$ROOT/tools/qtail_audit_droid_incremental_closure.py" \
    --python "$PYTHON" \
    --data-dir "$DATA_ROOT" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --checksum-ledger "$RESULT_ROOT/droid_object_checksum_ledger.json" \
    --cache-manifest "$RESULT_ROOT/droid_feature_cache_manifest.json" \
    --record-audit "$RESULT_ROOT/droid_feature_cache_verification.json" \
    --out "$RESULT_ROOT/droid_incremental_closure_selftest.json" >> "$LOG" 2>&1
  closure_selftest_code=$?
  if [ "$closure_selftest_code" -ne 0 ]; then
    log "DROID full closure controls exited $closure_selftest_code"
    exit "$closure_selftest_code"
  fi
  "$PYTHON" "$ROOT/tools/qtail_seal_droid_release_milestones.py" \
    --data-dir "$DATA_ROOT" \
    --checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json" \
    --checksum-ledger "$RESULT_ROOT/droid_object_checksum_ledger.json" \
    --closure "$RESULT_ROOT/droid_incremental_closure_audit.json" \
    --milestone-dir "$RESULT_ROOT/release_milestones" \
    --out "$RESULT_ROOT/droid_release_milestone_status.json" >> "$LOG" 2>&1
  milestone_code=$?
  if [ "$milestone_code" -ne 0 ]; then
    log "DROID official release milestone seal exited $milestone_code"
    exit "$milestone_code"
  fi
  "$PYTHON" "$ROOT/tools/qtail_merge_droid_artifact_manifest.py" \
    --manifest "$RESULT_ROOT/droid_artifact_manifest.json" \
    --formal-droid-root "$RESULT_ROOT" \
    >> "$LOG" 2>&1
  artifact_merge_code=$?
  if [ "$artifact_merge_code" -ne 0 ]; then
    log "DROID artifact-manifest closure merge exited $artifact_merge_code"
    exit "$artifact_merge_code"
  fi
  cp "$RESULT_ROOT/droid_artifact_manifest.json" \
    "$RESULT_ROOT/droid_training_artifact_manifest.json.tmp"
  mv "$RESULT_ROOT/droid_training_artifact_manifest.json.tmp" \
    "$RESULT_ROOT/droid_training_artifact_manifest.json"
  "$PYTHON" - \
    "$RESULT_ROOT/droid_full_training_report.json" \
    "$RESULT_ROOT/droid_feature_cache_verification.json" \
    "$RESULT_ROOT/droid_environment_manifest.json" \
    "$ROOT/tools" <<'PY'
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[4])
from qtail_train_droid_full import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_STRATA,
    FORMAL_CHECKPOINT_EVERY_STEPS,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_HOLDOUT_FRACTION,
    FORMAL_HOLDOUT_RELATIVE_PATH_SHA256,
    FORMAL_HOLDOUT_SHARDS_PER_RELEASE,
    FORMAL_MIN_RECORD_PARSE_RATE,
    FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE,
    FORMAL_PT_SOURCE_SHA256,
    FORMAL_RANDOMIZATION_SAMPLES,
    FORMAL_SEED,
    FORMAL_STEPS_PER_STAGE,
    OPTIMIZER_SIGNATURE,
    OPTIMIZER_UPDATE_SEMANTICS,
    RARE_COVERAGE_BUDGETS,
    RARE_COVERAGE_THRESHOLDS,
    RARE_INSTRUCTION_MAX_TRAIN_DF,
    deterministic_release_stratified_split,
    rare_instruction_fingerprint_coverage,
)

report_path = Path(sys.argv[1])
cache_verification_path = Path(sys.argv[2])
environment_path = Path(sys.argv[3])
if not report_path.is_file():
    raise SystemExit("DROID training report is missing")
if not cache_verification_path.is_file():
    raise SystemExit("DROID feature-cache verification is missing")
if not environment_path.is_file():
    raise SystemExit("DROID environment manifest is missing")
report = json.loads(report_path.read_text(encoding="utf-8"))
cache_verification = json.loads(
    cache_verification_path.read_text(encoding="utf-8")
)
environment = json.loads(environment_path.read_text(encoding="utf-8"))
if (
    environment.get("status") != "complete"
    or not environment.get("gates")
    or not all(environment["gates"].values())
):
    raise SystemExit("DROID environment manifest gates are not satisfied")
if report.get("status") != "complete":
    raise SystemExit("DROID training report does not declare complete")
if report.get("training_scope") != "all_complete_shards_all_decodable_records":
    raise SystemExit("DROID training report is not an all-record full-shard run")
formal = report.get("formal_protocol", {})
if (
    formal.get("locked") is not True
    or int(formal.get("seed", -1)) != FORMAL_SEED
    or int(formal.get("steps_per_stage", -1)) != FORMAL_STEPS_PER_STAGE
    or float(formal.get("holdout_fraction", -1.0))
    != FORMAL_HOLDOUT_FRACTION
    or int(formal.get("holdout_shards_per_release", -1))
    != FORMAL_HOLDOUT_SHARDS_PER_RELEASE
    or formal.get("holdout_relative_path_sha256")
    != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
    or formal.get("holdout_membership_path_scope")
    != "official_release_relative_path"
    or int(formal.get("bootstrap_samples", -1))
    != FORMAL_BOOTSTRAP_SAMPLES
    or int(formal.get("randomization_samples", -1))
    != FORMAL_RANDOMIZATION_SAMPLES
    or int(formal.get("checkpoint_every_steps", -1))
    != FORMAL_CHECKPOINT_EVERY_STEPS
    or float(formal.get("min_record_parse_rate", -1.0))
    != FORMAL_MIN_RECORD_PARSE_RATE
    or float(formal.get("min_record_scan_complete_rate", -1.0))
    != FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE
    or formal.get("require_verified_mirror") is not True
    or formal.get("pt_source_sha256") != FORMAL_PT_SOURCE_SHA256
    or int(report.get("seed", -1)) != FORMAL_SEED
):
    raise SystemExit("DROID formal protocol lock is not satisfied")
audit = report.get("compute_audit", {})
checkpoint_audit = report.get("intermediate_checkpoint_audit", {})
steps = int(report.get("steps", -1))
if (
    steps != 20_000
    or int(report.get("total_steps_per_arm", -1)) != 40_000
    or int(audit.get("source_steps", -2)) != 40_000
    or int(audit.get("qtail_steps", -3)) != 40_000
    or int(audit.get("evaluation_source_steps", -1)) != steps
    or int(audit.get("evaluation_qtail_steps", -1)) != steps
    or int(audit.get("deployment_source_steps", -1)) != steps
    or int(audit.get("deployment_qtail_steps", -1)) != steps
    or int(audit.get("evaluation_source_optimizer_updates", -1)) != steps
    or int(audit.get("evaluation_qtail_optimizer_updates", -1)) != steps
    or int(audit.get("deployment_source_optimizer_updates", -1)) != steps
    or int(audit.get("deployment_qtail_optimizer_updates", -1)) != steps
    or int(audit.get("source_optimizer_updates", -1)) != steps * 2
    or int(audit.get("qtail_optimizer_updates", -1)) != steps * 2
    or audit.get("optimizer_update_semantics")
    != OPTIMIZER_UPDATE_SEMANTICS
    or set(audit.get("resume", {}))
    != {
        "evaluation_source",
        "evaluation_qtail",
        "deployment_source",
        "deployment_qtail",
    }
    or not all(
        int(item.get("target_step", -1)) == steps
        and int(item.get("optimizer_updates_completed", -1)) == steps
        and item.get("device") == audit.get("training_device")
        and item.get("optimizer") == OPTIMIZER_SIGNATURE
        and item.get("environment_fingerprint")
        == audit.get("checkpoint_environment_fingerprint")
        and (
            not item.get("resumed")
            or (
                item.get("checkpoint_device")
                == audit.get("training_device")
                and item.get("checkpoint_optimizer")
                == OPTIMIZER_SIGNATURE
                and item.get("checkpoint_environment_fingerprint")
                == audit.get("checkpoint_environment_fingerprint")
            )
        )
        and item.get("step_semantics")
        == "Checkpoint step k is the state after exactly k optimizer updates."
        for item in audit.get("resume", {}).values()
    )
    or audit.get("architecture") != "AllocationHead(10→32→16→1)"
    or not audit.get("same_architecture")
    or audit.get("same_optimizer") != OPTIMIZER_SIGNATURE
    or not audit.get("same_seed")
    or not audit.get("same_features")
    or not audit.get("same_device")
    or audit.get("same_environment_fingerprint") is not True
    or len(str(audit.get("runtime_environment_fingerprint", ""))) != 64
    or len(str(audit.get("checkpoint_environment_fingerprint", ""))) != 64
    or not audit.get("same_parameter_count")
    or int(audit.get("source_parameter_count", -1)) <= 0
    or int(audit.get("source_parameter_count", -1))
    != int(audit.get("qtail_parameter_count", -2))
    or checkpoint_audit.get("status") != "complete"
    or checkpoint_audit.get(
        "paired_feature_signatures_equal"
    )
    is not True
    or checkpoint_audit.get(
        "initialized_state_signatures_equal"
    )
    is not True
):
    raise SystemExit("DROID same-compute contract is not satisfied")
holdout = report.get("holdout_evaluation", {})
release_holdout = {
    str(item.get("release")): item
    for item in holdout.get("per_release", [])
    if isinstance(item, dict)
}
holdout_relative_paths_raw = holdout.get("holdout_relative_paths", [])
holdout_relative_paths_typed = bool(
    isinstance(holdout_relative_paths_raw, list)
    and all(isinstance(value, str) for value in holdout_relative_paths_raw)
)
holdout_relative_paths = (
    holdout_relative_paths_raw if holdout_relative_paths_typed else []
)
holdout_relative_path_sha256 = hashlib.sha256(
    "\n".join(holdout_relative_paths).encode("utf-8")
).hexdigest()
if (
    holdout.get("version")
    != "release_stratified_official_relative_path_hash_v2"
    or holdout.get("membership_path_scope")
    != "official_release_relative_path"
    or holdout.get("holdout_membership_locked") is not True
    or not holdout_relative_paths_typed
    or len(holdout_relative_paths)
    != FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
    or holdout_relative_paths != sorted(holdout_relative_paths)
    or len(set(holdout_relative_paths))
    != FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
    or holdout_relative_path_sha256
    != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
    or holdout.get("holdout_relative_path_sha256")
    != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
    or holdout.get("normalization_fit") != "training_shards_only"
    or holdout.get("tail_taxonomy_scope")
    != "training_shards_fit_applied_to_holdout"
    or holdout.get("instruction_rarity_fit") != "training_shards_only"
    or holdout.get("pt_allocation_fit") != "training_shards_only"
    or holdout.get("evaluation_predictions_scope") != "holdout_shards_only"
    or int(holdout.get("training_shards", 0))
    + int(holdout.get("holdout_shards", 0))
    != 4_096
    or int(holdout.get("holdout_shards", 0)) <= 0
    or int(holdout.get("holdout_shards", -1))
    != FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
    or float(holdout.get("requested_holdout_fraction", -1.0))
    != FORMAL_HOLDOUT_FRACTION
    or int(holdout.get("seed", -1)) != FORMAL_SEED
    or set(release_holdout) != {"1.0.0", "1.0.1"}
    or not all(
        int(item.get("training_shards", 0)) > 0
        and int(item.get("holdout_shards", 0))
        == FORMAL_HOLDOUT_SHARDS_PER_RELEASE
        and int(item.get("training_shards", 0))
        + int(item.get("holdout_shards", 0))
        == 2_048
        for item in release_holdout.values()
    )
):
    raise SystemExit("DROID deterministic release-stratified holdout gate failed")
tail_contract = report.get("tail_score_contract", {})
if (
    int(tail_contract.get("transform_fit_row_count", -1))
    != int(holdout.get("training_shards", -2))
    or int(tail_contract.get("allocation_fit_row_count", -1))
    != int(holdout.get("training_shards", -2))
    or tail_contract.get("instruction_document_frequency_fit")
    != "normalization_fit_rows_only"
):
    raise SystemExit("DROID held-out tail transform leakage gate failed")
pt_source = report.get("pt_source_audit", {})
if (
    int(pt_source.get("count", 0)) < 4_096
    or pt_source.get("sha256") != FORMAL_PT_SOURCE_SHA256
    or not math.isfinite(float(pt_source.get("coefficient_of_variation", math.nan)))
):
    raise SystemExit("DROID empirical PT source audit failed")
expected_releases = {
    "1.0.0": ("r2d2_faceblur", 2_048, 92_233),
    "1.0.1": ("droid_101", 2_048, 95_658),
}
release_rows = {
    str(item.get("release")): item
    for item in report.get("release_composition", [])
    if isinstance(item, dict)
}
try:
    release_composition_valid = (
        set(release_rows) == set(expected_releases)
        and all(
            release_rows[release].get("official_dataset_name") == dataset
            and release_rows[release].get("metadata_status") == "verified"
            and int(release_rows[release].get("observed_tfrecord_shards", -1))
            == shards
            and int(release_rows[release].get("official_tfrecord_shards", -1))
            == shards
            and int(release_rows[release].get("observed_records_decoded", -1))
            == records
            and int(release_rows[release].get("official_expected_records", -1))
            == records
            and release_rows[release].get("full_shard_coverage") is True
            and release_rows[release].get("full_record_count_match") is True
            for release, (dataset, shards, records) in expected_releases.items()
        )
        and sum(
            int(item.get("observed_tfrecord_shards", 0))
            for item in release_rows.values()
        )
        == 4_096
        and sum(
            int(item.get("observed_records_decoded", 0))
            for item in release_rows.values()
        )
        == 187_891
        and sum(
            int(item.get("observed_tfrecord_bytes", 0))
            for item in release_rows.values()
        )
        == int(report.get("total_bytes", -1))
    )
except (TypeError, ValueError):
    release_composition_valid = False
if not release_composition_valid:
    raise SystemExit("DROID release-level shard and record audit is not satisfied")
evidence = report.get("trajectory_evidence", {})
if not evidence.get("full_record_mode"):
    raise SystemExit("DROID report does not prove full-record mode")
if float(evidence.get("record_parse_rate", 0.0)) != 1.0:
    raise SystemExit("DROID record parse rate is not 100%")
if float(evidence.get("record_scan_complete_rate", 0.0)) != 1.0:
    raise SystemExit("DROID full-record scan completion gate failed")
input_audit = report.get("input_audit", {})
if (
    not input_audit.get("required")
    or not input_audit.get("verified")
    or int(input_audit.get("expected_tfrecord_shards", -1))
    != int(input_audit.get("actual_tfrecord_shards", -2))
):
    raise SystemExit("DROID verified-mirror input gate is not satisfied")
if (
    cache_verification.get("status") != "verified"
    or cache_verification.get("all_official_tfrecords") is not True
    or cache_verification.get("full_official_record_count_match") is not True
    or int(cache_verification.get("verified_cache_count", -1))
    != int(cache_verification.get("official_tfrecord_count", -2))
    or int(cache_verification.get("error_count", -1)) != 0
    or cache_verification.get("feature_values_recomputed") is not True
    or cache_verification.get("all_feature_values_recomputed") is not True
    or int(cache_verification.get("recomputed_feature_count", -1))
    != 4_096
):
    raise SystemExit("DROID full feature-cache verification gate failed")
effect = report.get("effect_metrics", {})
bootstrap = effect.get("paired_bootstrap", {})
randomization = effect.get("paired_arm_swap_randomization", {})
hypothesis_gate = effect.get("hypothesis_gate", {})
bootstrap_strata_counts = bootstrap.get("strata_counts", {})
try:
    bootstrap_strata_valid = (
        bootstrap.get("strata") == list(BOOTSTRAP_STRATA)
        and set(bootstrap_strata_counts) == set(BOOTSTRAP_STRATA)
        and all(
            int(bootstrap_strata_counts[release])
            == int(release_holdout[release].get("holdout_shards", -1))
            for release in BOOTSTRAP_STRATA
        )
        and sum(
            int(bootstrap_strata_counts[release])
            for release in BOOTSTRAP_STRATA
        )
        == int(holdout.get("holdout_shards", -1))
    )
except (TypeError, ValueError, KeyError):
    bootstrap_strata_valid = False
finite_effect_values = [
    effect.get("source_pred_tail_share"),
    effect.get("qtail_pred_tail_share"),
    effect.get("predicted_tail_share_gain_pp"),
    effect.get("source_extreme_underallocation_rate"),
    effect.get("qtail_extreme_underallocation_rate"),
    effect.get("extreme_underallocation_reduction_pp"),
    bootstrap.get("mean_gain_pp"),
    bootstrap.get("ci95_low_pp"),
    bootstrap.get("ci95_high_pp"),
    bootstrap.get("descriptive_fraction_gain_le_zero"),
    randomization.get("observed_gain_pp"),
    randomization.get("diagnostic_exceedance_fraction"),
]
if not all(
    isinstance(value, (int, float)) and math.isfinite(float(value))
    for value in finite_effect_values
):
    raise SystemExit("DROID effect metrics contain non-finite values")
if (
    effect.get("tail_definition")
    != "heldout_top_30_percent_by_record_informed_tail_score_v2"
    or effect.get("extreme_definition")
    != "heldout_top_10_percent_by_record_informed_tail_score_v2"
    or effect.get("evaluation_scope")
    != "deterministic_release_stratified_heldout_shards"
    or int(bootstrap.get("samples", -1)) != 5_000
    or bootstrap.get("method") != BOOTSTRAP_METHOD
    or not bootstrap_strata_valid
    or bootstrap.get("p_gain_le_zero_is_p_value") is not False
    or randomization.get("version")
    != "paired_shard_arm_swap_diagnostic_v2"
    or int(randomization.get("samples", -1))
    != FORMAL_RANDOMIZATION_SAMPLES
    or randomization.get("unit")
    != "non_independent_heldout_shard_weight"
    or randomization.get("finite_sample_correction") != "(k+1)/(B+1)"
    or randomization.get(
        "exchangeability_justified_by_experiment_design"
    )
    is not False
    or randomization.get("inference_role")
    != "dependency_sensitive_descriptive_diagnostic_only"
    or randomization.get("conditional_p_value_is_valid_p_value")
    is not False
    or hypothesis_gate.get("name")
    != "heldout_tail_allocation_outcome_v4"
    or float(hypothesis_gate.get("minimum_tail_share_gain_pp", -1.0)) != 2.0
    or hypothesis_gate.get("requires_ci95_low_at_least_minimum")
    is not True
    or hypothesis_gate.get(
        "requires_positive_extreme_underallocation_reduction"
    )
    is not True
    or hypothesis_gate.get("completion_role")
    != "outcome_only_not_experiment_execution_gate"
    or hypothesis_gate.get(
        "randomization_diagnostic_is_valid_p_value"
    )
    is not False
    or not isinstance(hypothesis_gate.get("passed"), bool)
):
    raise SystemExit("DROID effect metric definitions or bootstrap scope are invalid")
for key in (
    "source_pred_tail_share",
    "qtail_pred_tail_share",
    "source_extreme_underallocation_rate",
    "qtail_extreme_underallocation_rate",
):
    if not 0.0 <= float(effect[key]) <= 1.0:
        raise SystemExit(f"DROID effect metric {key} is outside [0, 1]")
if float(bootstrap["ci95_low_pp"]) > float(bootstrap["ci95_high_pp"]):
    raise SystemExit("DROID paired bootstrap CI is reversed")
if not 0.0 <= float(
    bootstrap["descriptive_fraction_gain_le_zero"]
) <= 1.0:
    raise SystemExit(
        "DROID paired bootstrap descriptive fraction is outside [0, 1]"
    )
if not 0.0 < float(
    randomization["diagnostic_exceedance_fraction"]
) <= 1.0:
    raise SystemExit(
        "DROID arm-swap diagnostic fraction is outside (0, 1]"
    )
if (
    float(randomization["conditional_p_value"])
    != float(randomization["diagnostic_exceedance_fraction"])
):
    raise SystemExit("DROID arm-swap diagnostic compatibility alias differs")
tail_gain = (
    float(effect["qtail_pred_tail_share"])
    - float(effect["source_pred_tail_share"])
) * 100.0
extreme_reduction = (
    float(effect["source_extreme_underallocation_rate"])
    - float(effect["qtail_extreme_underallocation_rate"])
) * 100.0
expected_supported = bool(
    tail_gain >= 2.0
    and float(bootstrap["ci95_low_pp"]) >= 2.0
    and extreme_reduction > 0.0
)
expected_not_supported = bool(
    float(bootstrap["ci95_high_pp"]) < 2.0
    or extreme_reduction <= 0.0
)
expected_outcome = (
    "supported"
    if expected_supported
    else "not_supported"
    if expected_not_supported
    else "inconclusive"
)
if (
    not math.isclose(
        float(effect["predicted_tail_share_gain_pp"]),
        tail_gain,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    or not math.isclose(
        float(effect["extreme_underallocation_reduction_pp"]),
        extreme_reduction,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    or int(effect.get("tail_selected_shards", -1)) != 246
    or int(effect.get("tail_total_holdout_shards", -1)) != 820
    or int(effect.get("extreme_selected_shards", -1)) != 82
    or int(effect.get("extreme_total_holdout_shards", -1)) != 820
    or hypothesis_gate.get("outcome") != expected_outcome
    or hypothesis_gate.get("supported") is not expected_supported
    or hypothesis_gate.get("passed") is not expected_supported
):
    raise SystemExit("DROID recomputed effect metrics or outcome contract failed")

coverage = report.get("rare_instruction_fingerprint_coverage", {})
coverage_path = (
    report_path.parent / "droid_rare_instruction_fingerprint_coverage.json"
)
training_rows_path = report_path.parent / "droid_shard_training_rows.csv"
if not coverage_path.is_file() or not training_rows_path.is_file():
    raise SystemExit("DROID rare-fingerprint coverage artifacts are missing")
coverage_artifact = json.loads(coverage_path.read_text(encoding="utf-8"))
if coverage_artifact != coverage:
    raise SystemExit("DROID rare-fingerprint report and artifact differ")
with training_rows_path.open(newline="", encoding="utf-8") as handle:
    coverage_rows = list(csv.DictReader(handle))
if len(coverage_rows) != 4_096:
    raise SystemExit("DROID coverage recomputation requires 4,096 training rows")
for row in coverage_rows:
    try:
        hashes = ast.literal_eval(row.get("instruction_hashes", "[]"))
    except (SyntaxError, ValueError) as exc:
        raise SystemExit(
            f"Invalid structured instruction hashes for {row.get('path')}: {exc}"
        )
    if not isinstance(hashes, list) or not all(
        isinstance(value, str) for value in hashes
    ):
        raise SystemExit(
            f"Instruction hashes are not a string list for {row.get('path')}"
        )
    row["instruction_hashes"] = hashes
coverage_train, coverage_holdout, _ = deterministic_release_stratified_split(
    coverage_rows,
    holdout_fraction=FORMAL_HOLDOUT_FRACTION,
    seed=FORMAL_SEED,
)
source_predictions = np.asarray(
    [
        float(coverage_rows[int(index)]["holdout_source_pred"])
        for index in coverage_holdout
    ],
    dtype=np.float64,
)
qtail_predictions = np.asarray(
    [
        float(coverage_rows[int(index)]["holdout_qtail_pred"])
        for index in coverage_holdout
    ],
    dtype=np.float64,
)
recomputed_coverage = rare_instruction_fingerprint_coverage(
    coverage_rows,
    coverage_train,
    coverage_holdout,
    source_predictions,
    qtail_predictions,
)
if recomputed_coverage != coverage:
    raise SystemExit("DROID rare-fingerprint coverage recomputation differs")
coverage_curve = coverage.get("curve", [])
coverage_time = coverage.get("time_to_coverage", [])
coverage_status = coverage.get("status")
coverage_shape_valid = (
    (
        coverage_status == "complete"
        and int(coverage.get("rare_holdout_fingerprint_count", 0)) > 0
        and [int(item.get("draw_budget", -1)) for item in coverage_curve]
        == list(RARE_COVERAGE_BUDGETS)
        and [
            float(item.get("coverage_threshold", -1.0))
            for item in coverage_time
        ]
        == list(RARE_COVERAGE_THRESHOLDS)
    )
    or (
        coverage_status == "no_eligible_fingerprints"
        and int(coverage.get("rare_holdout_fingerprint_count", -1)) == 0
        and int(
            coverage.get("unseen_in_training_fingerprint_count", -1)
        )
        == 0
        and coverage_curve == []
        and coverage_time == []
        and bool(coverage.get("status_reason"))
    )
)
if (
    coverage.get("version")
    != "heldout_instruction_fingerprint_coverage_v1"
    or coverage_status
    not in {"complete", "no_eligible_fingerprints"}
    or not coverage_shape_valid
    or coverage.get("metric_role")
    != "auxiliary_descriptive_metric_not_a_completion_gate"
    or coverage.get("rarity_fit_scope") != "training_shards_only"
    or coverage.get("evaluation_scope") != "holdout_shards_only"
    or "not semantic task coverage"
    not in str(coverage.get("claim_boundary", ""))
    or int(coverage.get("training_shards", -1))
    != 4_096 - FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
    or int(coverage.get("holdout_shards", -1))
    != FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
    or int(coverage.get("max_training_shard_document_frequency", -1))
    != RARE_INSTRUCTION_MAX_TRAIN_DF
    or len(str(coverage.get("training_document_frequency_sha256", ""))) != 64
):
    raise SystemExit("DROID rare-fingerprint coverage contract is invalid")
for item in coverage_curve:
    source_value = float(item["source_expected_coverage"])
    qtail_value = float(item["qtail_expected_coverage"])
    if (
        not 0.0 <= source_value <= 1.0
        or not 0.0 <= qtail_value <= 1.0
        or not math.isclose(
            float(item["gain_pp"]),
            (qtail_value - source_value) * 100.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise SystemExit("DROID rare-fingerprint coverage values are invalid")
PY
  report_code=$?
  if [ "$report_code" -ne 0 ]; then
    log "DROID final training report audit exited $report_code; completion marker withheld"
    exit "$report_code"
  fi
  "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
    --job-root "$JOB_ROOT" \
    --stage training \
    --commit >> "$LOG" 2>&1
  marker_code=$?
  if [ "$marker_code" -ne 0 ]; then
    log "DROID training completion marker binding failed with code $marker_code"
    exit "$marker_code"
  fi
  log "DROID full allocation training complete"
  log "QTAIL_TERMINAL training_complete"
fi

log "re-sealing training artifact manifest and marker after regenerated controls"
if ! "$PYTHON" "$ROOT/tools/qtail_merge_droid_artifact_manifest.py" \
  --manifest "$RESULT_ROOT/droid_artifact_manifest.json" \
  --formal-droid-root "$RESULT_ROOT" >> "$LOG" 2>&1; then
  log "post-control artifact-manifest re-seal failed"
  exit 69
fi
cp "$RESULT_ROOT/droid_artifact_manifest.json" \
  "$RESULT_ROOT/droid_training_artifact_manifest.json.tmp"
mv "$RESULT_ROOT/droid_training_artifact_manifest.json.tmp" \
  "$RESULT_ROOT/droid_training_artifact_manifest.json"
if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
  --job-root "$JOB_ROOT" \
  --stage training \
  --commit >> "$LOG" 2>&1; then
  log "post-control training marker re-seal failed"
  exit 69
fi
refresh_status

if [ ! -f "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" ]; then
  if ! assert_final_qa_contract_compatible; then
    log "final QA fail-fast: legacy verifier still promotes preview to formal 9/9; blocker will self-heal after that forbidden module is separately updated"
    exit 72
  fi
  log "running final desktop/mobile page QA in honest 8/9 sealing state"
  /usr/bin/env node "$ROOT/tools/qtail_verify_droid_page.mjs" \
    --repo-root "$ROOT" \
    --job-root "$JOB_ROOT" \
    --page-url "http://127.0.0.1:54655/qtail-droid-full-training" >> "$LOG" 2>&1
  qa_code=$?
  if [ "$qa_code" -ne 0 ] || [ ! -f "$MARKER_ROOT/FINAL_PAGE_QA_COMPLETE" ]; then
    log "final page QA exited $qa_code; completion marker withheld"
    if [ "$qa_code" -eq 0 ]; then
      exit 70
    fi
    exit "$qa_code"
  fi
  qa_sha256="$(shasum -a 256 "$RESULT_ROOT/final_page_qa.json" | awk '{print $1}')"
  if [ "${#qa_sha256}" -ne 64 ]; then
    log "final QA SHA-256 capture failed"
    invalidate_final_qa
    exit 71
  fi
  log "QTAIL_TERMINAL qa_sealing_complete qa_sha256=$qa_sha256"
  if ! refresh_final_process_logs; then
    log "final process-log terminal snapshot failed; final marker withheld"
    invalidate_final_qa
    exit 71
  fi
  if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
    --job-root "$JOB_ROOT" --stage final --commit >> "$LOG" 2>&1; then
    invalidate_final_qa
    log "final page QA marker binding verification failed"
    exit 71
  fi
  rm -f "$MARKER_ROOT/FINAL_PAGE_QA_PREVIEW"
  refresh_status
fi

if [ ! -f "$MARKER_ROOT/DROID_PUBLIC_PROJECTION_COMMITTED" ]; then
  if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
    --job-root "$JOB_ROOT" --stage final \
    --commit-public-projection >> "$LOG" 2>&1; then
    invalidate_final_qa
    refresh_status
    log "public 9/9 projection snapshot binding failed"
    exit 71
  fi
fi

if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
  --job-root "$JOB_ROOT" --stage final \
  --validate-projection >> "$LOG" 2>&1; then
  invalidate_final_qa
  refresh_status
  log "committed final marker and public 9/9 projection did not verify"
  exit 71
fi

if [ ! -f "$MARKER_ROOT/DROID_POSTCOMMIT_PAGE_QA_COMPLETE" ]; then
  log "running read-only desktop/mobile browser QA against committed 9/9"
  /usr/bin/env node "$ROOT/tools/qtail_verify_droid_page.mjs" \
    --repo-root "$ROOT" \
    --job-root "$JOB_ROOT" \
    --page-url "http://127.0.0.1:54655/qtail-droid-full-training" \
    --post-commit-read-only >> "$LOG" 2>&1
  postcommit_code=$?
  if [ "$postcommit_code" -ne 0 ] \
    || [ ! -f "$MARKER_ROOT/DROID_POSTCOMMIT_PAGE_QA_COMPLETE" ]; then
    invalidate_postcommit_qa
    log "postcommit browser QA exited $postcommit_code; final completion withheld"
    if [ "$postcommit_code" -eq 0 ]; then
      exit 73
    fi
    exit "$postcommit_code"
  fi
fi

if ! "$PYTHON" "$ROOT/tools/qtail_verify_droid_stage_markers.py" \
  --job-root "$JOB_ROOT" --stage final >> "$LOG" 2>&1; then
  invalidate_postcommit_qa
  refresh_status
  log "postcommit browser evidence did not verify"
  exit 73
fi
log "QTAIL_TERMINAL qa_commit_complete"
log "final page QA committed; browser-rendered completion reached 9/9"
refresh_status

while true; do
  refresh_status
  sleep 300
done
