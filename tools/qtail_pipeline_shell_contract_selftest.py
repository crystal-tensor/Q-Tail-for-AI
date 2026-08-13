#!/usr/bin/env python3
"""Static positive/negative controls for irreversible shell handoffs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SNIPPETS = (
    '"version": "droid_checksum_verified_marker_v2"',
    '"download_completion_marker_sha256"',
    "if ! commit_checksum_marker; then",
    '"running final mirror-verifier positive/negative controls"',
    '"$MIRROR_VERIFIER_SELFTEST"',
    '"running formal DROID pre-optimizer gate-order controls"',
    '"$TRAINING_GATE_ORDER_SELFTEST"',
    '"existing checksum-marker fast path"',
    '"checksum completion handoff"',
    '"environment capture"',
    '"formal training launch"',
    "QTAIL_TERMINAL checksum_complete",
    "QTAIL_TERMINAL record_closure_complete records=187891",
    "QTAIL_TERMINAL training_complete",
    "QTAIL_TERMINAL qa_sealing_complete qa_sha256=$qa_sha256",
    "QTAIL_TERMINAL qa_commit_complete",
    "post-control training marker re-seal failed",
    "refresh_final_process_logs",
    "required_sources = {",
    '"qtail-web-services.log": (',
    '"dual-port DROID page supervision and recovery"',
    "optional_sources = {",
    '"qtail_droid_terminal_launcher.log": (',
    '"qtail_uniclash_guard_stderr.log": (',
    '"optional_log_count": len(optional_sources)',
    "--job-root \"$JOB_ROOT\" --stage final --commit",
    "--commit-public-projection",
    "DROID_PUBLIC_PROJECTION_COMMITTED",
    "--post-commit-read-only",
    "DROID_POSTCOMMIT_PAGE_QA_COMPLETE",
    "invalidate_postcommit_qa",
    "assert_final_qa_contract_compatible",
    "final_qa_contract_blocked.json",
    "invalidate_final_qa",
    "already_complete_read_only",
    "non-smoke final QA must be owned by qtail_orico_full_pipeline.sh",
    "the parent pipeline must invalidate it before QA",
    'DOWNLOAD_INTERFACE="${QTAIL_DROID_DOWNLOAD_INTERFACE:-en1}"',
    'ORCHESTRATION_SNAPSHOT_MANIFEST="$JOB_ROOT/code/qtail_orchestration/SHA256SUMS"',
    '"$PYTHON" "$ORCHESTRATION_SNAPSHOT_PUBLISHER"',
    '--orchestration-snapshot-manifest "$ORCHESTRATION_SNAPSHOT_MANIFEST"',
    '--forbid-tunnel-route',
    '--expected-interface "$DOWNLOAD_INTERFACE"',
    '--proxy "$DOWNLOAD_PROXY"',
    "pipeline lock is owned by live pid=%s command=%s; refusing to steal",
    "--require-formal",
    "qtail_openx_migration_marker_v2",
    "validate_openx_migration_marker",
    "write_openx_migration_marker",
    'payload.get("resolved_source") != str(target)',
    "qtail_droid_backend_marker_v2",
    "validate_droid_backend_marker",
    "write_droid_backend_marker",
    'payload.get("source_root") != str(source)',
    'payload.get("commit_marker_sha256")',
    "hashlib.sha256(commit_marker.read_bytes()).hexdigest()",
    'payload.get("git_fsck_passed") is not True',
    "qtail_pipeline_started_marker_v2",
    "write_pipeline_started_marker",
    '"script_sha256"',
    '"lock_owner_pid"',
    "pipeline lock owner differs from marker pid",
    "qtail_pipeline_generation_gate_v1",
    "require_pipeline_generation_marker",
    '"pre-checksum"',
    '"pre-environment"',
    '"pre-formal-training"',
    "pipeline_generation_gate.json",
    '"marker_sha_matches_current_source"',
    '"live_command_matches"',
    "qtail_droid_source_probe_marker_v2",
    "validate_droid_source_probe_marker",
    "--seal-existing",
    '"report_sha256"',
    'audit.get("checkpoint_environment_fingerprint")',
    "bound_byte_checksum_plus_4102_stat_identity_v1",
    'CHECKSUM_STAT_CONTINUITY_REPORT="$RESULT_ROOT/droid_checksum_stat_continuity.json"',
    "QTAIL_TERMINAL checksum_complete path=bound_byte_checksum_plus_stat_continuity",
)

FORMAL_TRAINING_SNIPPETS = (
    '"$PYTHON" "$ROOT/tools/qtail_train_droid_full.py"',
    '--data-dir "$DATA_ROOT"',
    '--out "$RESULT_ROOT"',
    '--marker-dir "$MARKER_ROOT"',
    '--object-manifest "$RESULT_ROOT/droid_object_manifest.json"',
    '--checksum-manifest "$RESULT_ROOT/droid_object_checksum_manifest.json"',
    '--checksum-ledger "$RESULT_ROOT/droid_object_checksum_ledger.json"',
    '--transport-status "$RESULT_ROOT/parallel_download_status.json"',
    '--download-marker "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE"',
    '--download-verification "$RESULT_ROOT/download_verification.json"',
    '--environment-manifest "$RESULT_ROOT/droid_environment_manifest.json"',
    "--require-verified-mirror",
    '--process-lock "$RESULT_ROOT/.qtail_train_droid_full.lock"',
    "--required-mount /Volumes/ORICO",
    "--steps 20000",
    "--records-per-shard 0",
    "--min-record-parse-rate 1.0",
    "--min-record-scan-complete-rate 1.0",
    "--checkpoint-every-steps 5000",
    "--seed 11",
    "--bootstrap-samples 5000",
    "--holdout-fraction 0.20",
    '--pt-source "$ROOT/data/uploaded_data.csv"',
)

FORMAL_ARTIFACT_PRODUCER_SNIPPETS = (
    '"$ROOT/tools/qtail_build_droid_object_manifest.py"',
    '"$ROOT/tools/qtail_build_droid_checksum_manifest.py"',
    '"$ROOT/tools/qtail_audit_droid_release_metadata.py"',
    '"$ROOT/tools/qtail_droid_timeline_monotonic_selftest.py"',
    '"$ROOT/tools/qtail_droid_protocol_selftest.py"',
    '"$ROOT/tools/qtail_capture_droid_environment.py"',
    '"$ORCHESTRATION_SNAPSHOT_PUBLISHER"',
    '"$ROOT/tools/qtail_verify_droid_feature_cache.py"',
    '"$ROOT/tools/qtail_audit_droid_incremental_closure.py"',
    '"$ROOT/tools/qtail_droid_incremental_closure_selftest.py"',
    '"$ROOT/tools/qtail_seal_droid_release_milestones.py"',
    '"$ROOT/tools/qtail_merge_droid_artifact_manifest.py"',
    '"$DOWNLOAD_MARKER_SELFTEST"',
    '"$MIRROR_VERIFIER_SELFTEST"',
    '"$RUNTIME_PROCESS_SELFTEST"',
    '"$UNICLASH_GATE_SELFTEST"',
    '"$DOWNLOADER_SELFTEST"',
    '"$STAGE_HARDENING_SELFTEST"',
    '"$PREVIEW_SELFTEST"',
    '"$MANIFEST_SELFTEST"',
    '"$SHELL_CONTRACT_SELFTEST"',
    '"$TRAINING_GATE_ORDER_SELFTEST"',
    '"$CLASSIFIER_SELFTEST_REPORT"',
    '"$UNICLASH_CHECKSUM_HANDOFF_GATE_REPORT"',
    '"$UNICLASH_PRE_ENVIRONMENT_GATE_REPORT"',
    '"$UNICLASH_PRE_TRAINING_GATE_REPORT"',
)

ORDERED_SNIPPETS = (
    "QTAIL_TERMINAL checksum_complete",
    '"running formal DROID pre-optimizer gate-order controls"',
    '"environment capture"',
    '"waiting for feature prewarm loop to exit before formal training"',
    '"formal training launch"',
    '"$PYTHON" "$ROOT/tools/qtail_train_droid_full.py"',
    "--require-verified-mirror",
    "QTAIL_TERMINAL record_closure_complete records=187891",
    "QTAIL_TERMINAL training_complete",
    "QTAIL_TERMINAL qa_sealing_complete qa_sha256=$qa_sha256",
    "if ! refresh_final_process_logs; then",
    '--job-root "$JOB_ROOT" --stage final --commit',
    "--commit-public-projection",
    '"running read-only desktop/mobile browser QA against committed 9/9"',
    "--post-commit-read-only",
    "QTAIL_TERMINAL qa_commit_complete",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contract_errors(source: str) -> list[str]:
    errors = [
        f"missing shell contract: {snippet}"
        for snippet in (
            *REQUIRED_SNIPPETS,
            *FORMAL_TRAINING_SNIPPETS,
            *FORMAL_ARTIFACT_PRODUCER_SNIPPETS,
        )
        if snippet not in source
    ]
    milestone_start = source.find(
        '"$PYTHON" "$ROOT/tools/qtail_seal_droid_release_milestones.py"'
    )
    milestone_end = source.find("milestone_code=$?", milestone_start)
    if milestone_start < 0 or milestone_end < 0:
        errors.append("release milestone invocation block is missing")
    else:
        milestone_block = source[milestone_start:milestone_end]
        if milestone_block.count('--data-dir "$DATA_ROOT"') != 1:
            errors.append(
                "release milestone invocation must pass --data-dir exactly once"
            )
    if source.count(
        '--orchestration-snapshot-manifest "$ORCHESTRATION_SNAPSHOT_MANIFEST"'
    ) != 2:
        errors.append(
            "environment controls and capture must each bind the ORICO snapshot"
        )
    if source.count(
        '--environment-manifest "$RESULT_ROOT/droid_environment_manifest.json"'
    ) != 1:
        errors.append(
            "formal trainer must bind exactly one environment manifest"
        )
    return errors


def ordering_errors(source: str) -> list[str]:
    cursor = 0
    for snippet in ORDERED_SNIPPETS:
        position = source.find(snippet, cursor)
        if position < 0:
            return [
                "terminal sealing order is missing or non-monotonic at "
                f"{snippet}"
            ]
        cursor = position + len(snippet)
    return []


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.pipeline.read_text(encoding="utf-8")
    positive_errors = contract_errors(source)
    training_label_mutated = source.replace(
        '"formal training launch"', '"formal training removed"', 1
    )
    training_label_errors = contract_errors(training_label_mutated)
    mirror_gate_mutated = source.replace(
        "--require-verified-mirror",
        "--verified-mirror-gate-removed",
        1,
    )
    mirror_gate_errors = contract_errors(mirror_gate_mutated)
    environment_binding_mutated = source.replace(
        '--environment-manifest "$RESULT_ROOT/droid_environment_manifest.json"',
        "--environment-manifest-removed",
        1,
    )
    environment_binding_errors = contract_errors(
        environment_binding_mutated
    )
    projection_mutated = source.replace(
        "--commit-public-projection",
        "--public-projection-removed",
        1,
    )
    projection_errors = contract_errors(projection_mutated)
    postcommit_mutated = source.replace(
        "--post-commit-read-only",
        "--postcommit-browser-qa-removed",
    )
    postcommit_errors = contract_errors(postcommit_mutated)
    postcommit_marker_mutated = source.replace(
        "DROID_POSTCOMMIT_PAGE_QA_COMPLETE",
        "POSTCOMMIT_MARKER_REMOVED",
    )
    postcommit_marker_errors = contract_errors(
        postcommit_marker_mutated
    )
    positive_order_errors = ordering_errors(source)
    sealing = ORDERED_SNIPPETS[0]
    committed = ORDERED_SNIPPETS[-1]
    ordering_mutated = source.replace(sealing, "__QTAIL_SEALING__", 1)
    ordering_mutated = ordering_mutated.replace(committed, sealing, 1)
    ordering_mutated = ordering_mutated.replace(
        "__QTAIL_SEALING__",
        committed,
        1,
    )
    negative_order_errors = ordering_errors(ordering_mutated)
    readonly_mutated = source.replace(
        "already_complete_read_only",
        "completed_state_was_deleted",
        1,
    )
    readonly_errors = contract_errors(readonly_mutated)
    owner_mutated = source.replace(
        "non-smoke final QA must be owned by qtail_orico_full_pipeline.sh",
        "unowned final QA accepted",
        1,
    )
    owner_errors = contract_errors(owner_mutated)
    checkpoint_environment_mutated = source.replace(
        'audit.get("checkpoint_environment_fingerprint")',
        'audit.get("runtime_environment_fingerprint")',
    )
    checkpoint_environment_errors = contract_errors(
        checkpoint_environment_mutated
    )
    controls = [
        {
            "name": "current_pipeline_contract_passes",
            "passed": positive_errors == [],
            "errors": positive_errors,
        },
        {
            "name": "removed_training_or_verified_mirror_gate_is_rejected",
            "passed": (
                any(
                    "formal training launch" in item
                    for item in training_label_errors
                )
                and any(
                    "--require-verified-mirror" in item
                    for item in mirror_gate_errors
                )
            ),
            "errors": [*training_label_errors, *mirror_gate_errors],
        },
        {
            "name": "removed_environment_code_binding_is_rejected",
            "passed": any(
                "--environment-manifest" in item
                for item in environment_binding_errors
            ),
            "errors": environment_binding_errors,
        },
        {
            "name": "removed_public_projection_commit_is_rejected",
            "passed": any(
                "--commit-public-projection" in item
                for item in projection_errors
            ),
            "errors": projection_errors,
        },
        {
            "name": "removed_postcommit_browser_qa_is_rejected",
            "passed": any(
                "--post-commit-read-only" in item
                for item in postcommit_errors
            ),
            "errors": postcommit_errors,
        },
        {
            "name": "removed_postcommit_marker_gate_is_rejected",
            "passed": any(
                "DROID_POSTCOMMIT_PAGE_QA_COMPLETE" in item
                for item in postcommit_marker_errors
            ),
            "errors": postcommit_marker_errors,
        },
        {
            "name": "terminal_sealing_order_passes",
            "passed": positive_order_errors == [],
            "errors": positive_order_errors,
        },
        {
            "name": "reversed_terminal_sealing_order_is_rejected",
            "passed": bool(negative_order_errors),
            "errors": negative_order_errors,
        },
        {
            "name": "removed_read_only_completed_state_is_rejected",
            "passed": any(
                "already_complete_read_only" in item
                for item in readonly_errors
            ),
            "errors": readonly_errors,
        },
        {
            "name": "removed_parent_ownership_gate_is_rejected",
            "passed": any(
                "non-smoke final QA must be owned" in item
                for item in owner_errors
            ),
            "errors": owner_errors,
        },
        {
            "name": "runtime_fingerprint_cannot_replace_checkpoint_binding",
            "passed": any(
                "checkpoint_environment_fingerprint" in item
                for item in checkpoint_environment_errors
            ),
            "errors": checkpoint_environment_errors,
        },
    ]
    passed = all(control["passed"] for control in controls)
    payload = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "control": "droid_pipeline_shell_contract_v9",
        "controls_passed": sum(control["passed"] for control in controls),
        "controls_total": len(controls),
        "controls": controls,
        "claim_boundary": (
            "This statically proves the current shell contains the locked "
            "physical-interface-bound download and formal-training invocation, "
            "plus the named formal-artifact producers and monotonic checksum, "
            "environment, training, evidence-sealing, public projection, and "
            "read-only "
            "9/9 browser-verification order. Runtime stage "
            "markers and semantic artifact verifiers provide the independent "
            "execution evidence."
        ),
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
