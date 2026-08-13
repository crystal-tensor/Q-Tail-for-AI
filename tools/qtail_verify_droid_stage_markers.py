#!/usr/bin/env python3
"""Create and verify cryptographically bound DROID stage markers."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qtail_verify_droid_download_marker import (
    MARKER_VERSION as DOWNLOAD_MARKER_VERSION,
)
from qtail_verify_droid_download_marker import build_binding


TRAINING_MARKER_VERSION = "droid_training_completion_marker_v2"
FINAL_MARKER_VERSION = "droid_final_page_qa_marker_v3"
FINAL_BOOTSTRAP_MARKER_VERSION = "droid_final_page_qa_bootstrap_v1"
PUBLIC_PROJECTION_MARKER_VERSION = "droid_public_projection_marker_v1"
POSTCOMMIT_PAGE_QA_MARKER_VERSION = (
    "droid_postcommit_page_qa_marker_v1"
)
CHECKSUM_MARKER_VERSION = "droid_checksum_verified_marker_v2"
FORMAL_PT_SOURCE_SHA256 = (
    "59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f"
)
FORMAL_EXPECTED_OBJECTS = 4_102
FORMAL_EXPECTED_TFRECORDS = 4_096
FORMAL_EXPECTED_BYTES = 3_700_745_265_151
FORMAL_EXPECTED_RECORDS = 187_891
FORMAL_HOLDOUT_RELATIVE_PATH_SHA256 = (
    "16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767"
)
COMPLETION_REQUIREMENT_IDS = {
    "existing_assets_on_orico",
    "official_source_and_manifest",
    "uniclash_transport_isolation",
    "full_mirror_checksum",
    "all_record_scan",
    "same_compute_training",
    "intermediate_artifacts",
    "runtime_health",
    "final_page_qa",
}
PIPELINE_GENERATION_GATES = (
    "pre-checksum",
    "pre-environment",
    "pre-formal-training",
)
PIPELINE_GENERATION_CHECKS = {
    "semantic_marker",
    "running_status",
    "marker_pid_matches",
    "marker_script_matches",
    "marker_job_root_matches",
    "marker_sha_matches_current_source",
    "marker_lock_owner_matches",
    "live_lock_owner_matches",
    "live_command_matches",
}
INCREMENTAL_CLOSURE_SELFTEST_CHECKS = frozenset(
    {
        "positive_current_closure",
        "require_formal_matches_exact_full_gate",
        "record_count_tamper_rejected",
        "md5_ledger_tamper_rejected",
        "md5_after_error_sample_limit_rejected",
        "missing_listed_cache_rejected",
        "post_snapshot_tfrecord_is_deferred",
    }
)
HARDENING_SELFTEST_CONTROL_NAMES = frozenset(
    {
        "checksum_binding_positive",
        "checksum_binding_tamper_rejected",
        "transition_gate_positive",
        "transition_gate_tun_rejected",
        "incremental_closure_exact_seven_positive",
        "incremental_closure_missing_formal_gate_rejected",
        "incremental_closure_extra_check_rejected",
        "incremental_closure_false_check_rejected",
        "incremental_closure_formal_success_spoof_rejected",
        "pipeline_generation_three_gate_positive",
        "pipeline_generation_false_check_rejected",
        "pipeline_generation_missing_gate_rejected",
        "pipeline_generation_source_drift_rejected",
        "recursive_manifest_positive",
        "recursive_manifest_tamper_rejected",
        "public_state_binding_positive",
        "stale_public_state_binding_is_rejected",
        "public_projection_snapshot_binding_positive",
        "public_projection_live_tamper_rejected",
        "final_path_contract_binds_only_immutable_closure",
        "lease_bound_bootstrap_positive",
        "bootstrap_artifact_tamper_rejected",
        "terminal_log_gate_positive",
        "terminal_log_predating_artifact_is_rejected",
        "empty_log_is_rejected",
        "precommit_results_remain_withheld",
        "training_only_publication_leak_is_rejected",
        "final_commit_without_valid_bootstrap_rejected",
        "final_commit_with_expired_lease_rejected",
        "final_commit_with_bootstrap_owner_mismatch_rejected",
        "real_eight_of_nine_final_marker_commit",
        "public_projection_without_postcommit_qa_is_rejected",
        "atomic_eight_to_nine_with_postcommit_browser_marker",
        "postcommit_browser_screenshot_tamper_rejected",
        "public_projection_live_and_snapshot_match",
        "sealed_projection_snapshot_tamper_rejected",
        "sealed_final_marker_tamper_rejected",
        (
            "hardening_control_identity_contract_rejects_"
            "missing_extra_or_duplicate"
        ),
    }
)
PROGRESS_PREVIEW_SELFTEST_CONTROL_NAMES = frozenset(
    {
        "preview_stage_is_qa_in_progress",
        "preview_is_not_effective_completion",
        "lease_bound_bootstrap_remains_sealing_not_complete",
        "committed_marker_reaches_complete",
        "sealed_final_without_public_projection_stays_eight_of_nine",
        "bootstrap_projection_is_not_frozen",
        "committed_projection_can_be_frozen",
        "formal_pre_page_artifact_baseline_is_64",
        "effective_qa_adds_only_nine_process_log_artifacts",
        "complete_qa_adds_five_final_artifacts_without_baseline_drift",
        "workspace_snapshot_parity_accepts_match_and_rejects_drift_or_escape",
        "passed_json_artifact_is_ready",
        "failed_corrupt_and_false_json_semantics_are_withheld",
        "failed_final_qa_artifact_family_is_withheld",
        "complete_final_qa_artifact_family_is_ready",
    }
)
ARTIFACT_MANIFEST_SELFTEST_CONTROL_NAMES = frozenset(
    {
        "missing_optional_history_is_pruned",
        "present_optional_history_is_retained_and_hashed",
        "manifest_control_files_are_excluded",
        "required_manifest_membership_drift_is_rejected",
        "missing_formal_artifact_is_rejected",
        "escaped_symlink_is_rejected",
        "dotdot_outside_addition_is_rejected",
        "required_path_set_drift_is_rejected",
    }
)
PIPELINE_SHELL_SELFTEST_CONTROL_NAMES = frozenset(
    {
        "current_pipeline_contract_passes",
        "removed_training_or_verified_mirror_gate_is_rejected",
        "removed_environment_code_binding_is_rejected",
        "removed_public_projection_commit_is_rejected",
        "removed_postcommit_browser_qa_is_rejected",
        "removed_postcommit_marker_gate_is_rejected",
        "terminal_sealing_order_passes",
        "reversed_terminal_sealing_order_is_rejected",
        "removed_read_only_completed_state_is_rejected",
        "removed_parent_ownership_gate_is_rejected",
        "runtime_fingerprint_cannot_replace_checkpoint_binding",
    }
)
UNICLASH_GATE_SELFTEST_CHECKS = frozenset(
    {
        "positive_clean_direct_guard",
        "core_off_rejected",
        "tun_on_rejected",
        "stale_guard_rejected",
        "missing_gsutil_policy_rejected",
        "system_bypass_failure_rejected",
        "idle_core_restart_pause_is_disclosed_and_accepted",
        "blocked_sample_without_event_rejected",
        "core_restart_during_transfer_rejected",
        "unknown_policy_pause_rejected",
        "wrong_route_history_rejected",
        "live_tunnel_route_rejected",
        "global_violation_rejected",
    }
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_environment_code_manifest(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        environment = read_json(path)
    except (OSError, ValueError, TypeError) as error:
        return [f"environment manifest is unreadable: {error}"], {}
    gates = environment.get("gates", {})
    if (
        environment.get("status") != "complete"
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        errors.append("environment manifest gates are not complete")

    code_rows = environment.get("code", [])
    seen_paths: set[str] = set()
    if not isinstance(code_rows, list) or not code_rows:
        errors.append("environment code inventory is empty")
        code_rows = []
    for index, item in enumerate(code_rows):
        if not isinstance(item, dict):
            errors.append(f"environment code row {index} is invalid")
            continue
        raw_path = str(item.get("path", "")).strip()
        expected = str(item.get("sha256", "")).strip().lower()
        if not raw_path or raw_path in seen_paths:
            errors.append(
                f"environment code path is missing or duplicated: {raw_path}"
            )
            continue
        seen_paths.add(raw_path)
        code_path = Path(raw_path)
        if (
            item.get("exists") is not True
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not code_path.is_file()
            or sha256(code_path) != expected
        ):
            errors.append(f"environment-bound code drifted: {raw_path}")

    snapshot = environment.get("orchestration_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    snapshot_path = Path(str(snapshot.get("manifest", "")))
    snapshot_sha256 = str(snapshot.get("manifest_sha256", "")).lower()
    if (
        snapshot.get("code_parity_passed") is not True
        or snapshot.get("manifest_errors") not in ([], None)
        or int(snapshot.get("code_mismatch_count", -1)) != 0
        or not snapshot_path.is_file()
        or len(snapshot_sha256) != 64
        or sha256(snapshot_path) != snapshot_sha256
    ):
        errors.append("ORICO orchestration snapshot binding is invalid")
    return errors, environment


def artifact_entry(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256(path),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_pipeline_generation_gate(
    path: Path,
    *,
    script_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    script = (
        script_path
        if script_path is not None
        else Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_orico_full_pipeline.sh"
    )
    try:
        payload = read_json(path)
        gates = payload.get("gates")
        if (
            payload.get("format_version")
            != "qtail_pipeline_generation_gate_v1"
        ):
            errors.append("pipeline generation gate version is invalid")
        if payload.get("status") != "passed":
            errors.append("pipeline generation gate status is not passed")
        if payload.get("latest_gate") != PIPELINE_GENERATION_GATES[-1]:
            errors.append("pipeline generation gate did not reach formal training")
        if not isinstance(gates, list) or len(gates) != len(
            PIPELINE_GENERATION_GATES
        ):
            return [
                *errors,
                "pipeline generation gate history is incomplete",
            ]

        script_hash = sha256(script)
        observed_pids: set[int] = set()
        observed_hashes: set[str] = set()
        for expected_gate, entry in zip(
            PIPELINE_GENERATION_GATES,
            gates,
            strict=True,
        ):
            if not isinstance(entry, dict):
                errors.append(
                    f"pipeline generation gate entry is invalid: {expected_gate}"
                )
                continue
            checks = entry.get("checks")
            pid = entry.get("pid")
            try:
                pid = int(pid)
                lock_owner_pid = int(entry.get("lock_owner_pid"))
            except (TypeError, ValueError):
                pid = -1
                lock_owner_pid = -2
            expected_command = f"/bin/zsh {script}"
            current_hash = str(entry.get("current_script_sha256", ""))
            marker_hash = str(entry.get("marker_script_sha256", ""))
            if entry.get("gate") != expected_gate:
                errors.append(
                    f"pipeline generation gate order differs: {expected_gate}"
                )
            if entry.get("passed") is not True:
                errors.append(
                    f"pipeline generation gate is not passed: {expected_gate}"
                )
            if (
                not isinstance(checks, dict)
                or set(checks) != PIPELINE_GENERATION_CHECKS
                or not all(value is True for value in checks.values())
            ):
                errors.append(
                    f"pipeline generation checks are incomplete: {expected_gate}"
                )
            if pid <= 0 or lock_owner_pid != pid:
                errors.append(
                    f"pipeline generation PID binding is invalid: {expected_gate}"
                )
            if (
                entry.get("command") != expected_command
                or entry.get("expected_command") != expected_command
            ):
                errors.append(
                    f"pipeline generation command binding is invalid: {expected_gate}"
                )
            if (
                len(current_hash) != 64
                or marker_hash != current_hash
                or current_hash != script_hash
            ):
                errors.append(
                    f"pipeline generation source hash is invalid: {expected_gate}"
                )
            observed_pids.add(pid)
            observed_hashes.add(current_hash)
        if len(observed_pids) != 1:
            errors.append("pipeline generation gates span multiple PIDs")
        if observed_hashes != {script_hash}:
            errors.append("pipeline generation gates span multiple source hashes")
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"pipeline generation gate is unreadable: {error}")
    return errors


def validate_data_continuity_summary(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["final pipeline data continuity audit is missing"]
    if payload.get("status") not in {"passed", "repair_events_observed"}:
        errors.append("final pipeline data continuity status is invalid")
    for field in (
        "completed_object_decrease_events",
        "verified_object_decrease_events",
        "checksum_error_samples",
        "legacy_physical_byte_decrease_events",
        "feature_pass_reset_events",
        "committed_feature_counter_decrease_events",
    ):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"final pipeline data continuity count is invalid: {field}")
    if payload.get("committed_feature_counter_decrease_events") != 0:
        errors.append("final pipeline committed feature counter decreased")
    if not str(payload.get("claim_boundary", "")).strip():
        errors.append("final pipeline data continuity claim boundary is missing")
    return errors


def training_paths(job_root: Path) -> list[Path]:
    result_root = job_root / "results" / "qtail_droid_full"
    return [
        result_root / "download_verification.json",
        result_root / "droid_release_metadata_audit.json",
        result_root / "droid_protocol_selftest.json",
        result_root / "droid_environment_contract_selftest.json",
        result_root / "droid_download_marker_selftest.json",
        result_root / "droid_mirror_verifier_selftest.json",
        result_root / "droid_downloader_single_writer_selftest.json",
        result_root / "droid_runtime_process_contract_selftest.json",
        result_root / "pipeline_generation_gate.json",
        result_root / "droid_stage_marker_hardening_selftest.json",
        result_root / "droid_progress_preview_selftest.json",
        result_root / "droid_artifact_manifest_merge_selftest.json",
        result_root / "droid_pipeline_shell_contract_selftest.json",
        result_root / "droid_training_gate_order_selftest.json",
        result_root / "uniclash_pre_checksum_gate.json",
        result_root / "uniclash_pre_checksum_gate_selftest.json",
        result_root / "download_completion_marker.json",
        job_root / "manifests" / "DROID_CHECKSUM_VERIFIED",
        result_root / "uniclash_checksum_handoff_gate.json",
        result_root / "uniclash_pre_environment_gate.json",
        result_root / "uniclash_pre_training_gate.json",
        result_root / "droid_environment_manifest.json",
        result_root / "droid_feature_cache_verification.json",
        result_root / "droid_incremental_closure_audit.json",
        result_root / "droid_incremental_closure_selftest.json",
        result_root / "droid_release_milestone_status.json",
        result_root
        / "release_milestones"
        / "droid_release_1.0.0_complete.json",
        result_root
        / "release_milestones"
        / "droid_release_1.0.1_complete.json",
        result_root / "droid_full_training_report.json",
        result_root / "droid_rare_instruction_fingerprint_coverage.json",
        result_root / "droid_intermediate_checkpoint_manifest.json",
        result_root / "droid_model_training_status.json",
        result_root / "qtail_droid_allocation_head.pt",
        result_root / "droid_training_artifact_manifest.json",
    ]


def final_paths(job_root: Path) -> list[Path]:
    result_root = job_root / "results" / "qtail_droid_full"
    repo_root = Path(__file__).resolve().parents[1]
    # Live public status files are sealed separately after the final 9/9
    # projection, avoiding a circular dependency in this precommit marker.
    return [
        job_root / "manifests" / "DROID_TRAINING_COMPLETE",
        result_root / "droid_training_artifact_manifest.json",
        result_root / "droid_artifact_manifest.json",
        repo_root / "qtail-droid-full-training.html",
        result_root / "droid_full_training_report.json",
        result_root / "droid_rare_instruction_fingerprint_coverage.json",
        result_root / "droid_model_training_status.json",
        result_root / "droid_training_curve.csv",
        result_root / "download_progress_samples_final.json",
        result_root / "pipeline_timeline_final.json",
        result_root / "pipeline_timeline_final_verification.json",
        result_root / "final_page_desktop.png",
        result_root / "final_page_mobile.png",
        result_root / "final_page_qa.json",
        result_root / "droid_process_log_manifest.json",
        result_root / "uniclash_transport_guard_final.json",
    ]


def final_bootstrap_paths(job_root: Path) -> list[Path]:
    result_root = job_root / "results" / "qtail_droid_full"
    repo_root = Path(__file__).resolve().parents[1]
    return [
        job_root / "manifests" / "DROID_TRAINING_COMPLETE",
        repo_root / "qtail-droid-full-training.html",
        result_root / "droid_full_training_report.json",
        result_root / "droid_rare_instruction_fingerprint_coverage.json",
        result_root / "droid_model_training_status.json",
        result_root / "droid_training_curve.csv",
    ]


def public_projection_paths(job_root: Path) -> list[Path]:
    result_root = job_root / "results" / "qtail_droid_full"
    return [
        job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE",
        result_root / "latest.json",
        result_root / "completion_audit.json",
        result_root / "latest_final.json",
        result_root / "completion_audit_final.json",
    ]


def postcommit_page_qa_paths(job_root: Path) -> list[Path]:
    result_root = job_root / "results" / "qtail_droid_full"
    return [
        job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE",
        job_root / "manifests" / "DROID_PUBLIC_PROJECTION_COMMITTED",
        result_root / "latest_final.json",
        result_root / "completion_audit_final.json",
        result_root / "final_page_postcommit_qa.json",
        result_root / "final_page_postcommit_desktop.png",
        result_root / "final_page_postcommit_mobile.png",
    ]


def active_final_preview(job_root: Path) -> tuple[bool, dict[str, Any]]:
    preview_path = job_root / "manifests" / "FINAL_PAGE_QA_PREVIEW"
    try:
        payload = read_json(preview_path)
        if payload.get("status") != "preview_active":
            return False, payload
        owner_pid = int(payload["owner_pid"])
        expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        if expires_at <= datetime.now(timezone.utc):
            return False, payload
        os.kill(owner_pid, 0)
        command = subprocess.run(
            ["ps", "-p", str(owner_pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        allowed_owner = (
            "qtail_orico_full_pipeline.sh" in command
            or "qtail_verify_droid_page.mjs" in command
        )
        return allowed_owner, payload
    except (KeyError, OSError, TypeError, ValueError):
        return False, {}


def commit_final_bootstrap(job_root: Path) -> dict[str, Any]:
    training = validate_training_marker(job_root)
    if not training["valid"]:
        raise ValueError("training marker is invalid")
    preview_active, preview = active_final_preview(job_root)
    if not preview_active:
        raise ValueError("final QA preview lease is not active")
    paths = final_bootstrap_paths(job_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("bootstrap marker inputs are missing: " + ", ".join(missing))
    marker = job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE"
    payload = {
        "marker_version": FINAL_BOOTSTRAP_MARKER_VERSION,
        "status": "sealing",
        "committed_at": now(),
        "owner_pid": int(preview["owner_pid"]),
        "preview_expires_at": preview["expires_at"],
        "artifacts": [artifact_entry(path) for path in paths],
    }
    atomic_write_json(marker, payload)
    return payload


def validate_final_bootstrap(job_root: Path) -> dict[str, Any]:
    marker = job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE"
    errors: list[str] = []
    try:
        payload = read_json(marker)
        if payload.get("marker_version") != FINAL_BOOTSTRAP_MARKER_VERSION:
            errors.append("final bootstrap marker version is invalid")
        if payload.get("status") != "sealing":
            errors.append("final bootstrap marker status is not sealing")
        errors.extend(
            validate_bound_artifacts(
                payload.get("artifacts"),
                final_bootstrap_paths(job_root),
            )
        )
        preview_active, preview = active_final_preview(job_root)
        if not preview_active:
            errors.append("final bootstrap preview lease is not active")
        elif int(payload.get("owner_pid", -1)) != int(preview.get("owner_pid", -2)):
            errors.append("final bootstrap owner differs from preview owner")
        if str(payload.get("preview_expires_at", "")) != str(
            preview.get("expires_at", "")
        ):
            errors.append("final bootstrap expiry differs from preview expiry")
    except (OSError, TypeError, ValueError) as error:
        payload = {}
        errors.append(f"final bootstrap marker is unreadable: {error}")
    training = validate_training_marker(job_root)
    if not training["valid"]:
        errors.append("bound training marker is invalid")
    return {"valid": not errors, "errors": errors, "marker": str(marker)}


def validate_artifact_manifest_entries(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
        entries = manifest.get("artifacts")
        if (
            manifest.get("status") != "complete"
            or not isinstance(entries, list)
            or not entries
        ):
            return ["artifact manifest contract is incomplete"]
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"artifact manifest entry is invalid: {index}")
                continue
            raw_path = str(entry.get("path", ""))
            if not raw_path or raw_path in seen:
                errors.append(
                    f"artifact manifest path is missing or duplicated: {index}"
                )
                continue
            seen.add(raw_path)
            path = Path(raw_path)
            try:
                if (
                    not path.is_file()
                    or path.stat().st_size != int(entry.get("bytes", -1))
                    or sha256(path) != entry.get("sha256")
                ):
                    errors.append(
                        f"artifact manifest entry does not match disk: {path}"
                    )
            except (OSError, TypeError, ValueError):
                errors.append(f"artifact manifest entry is unreadable: {path}")
        contract = manifest.get("formal_droid_contract", {})
        if contract:
            if contract.get("all_required_present") is not True:
                errors.append("formal artifact contract is not complete")
            required_count = int(contract.get("required_artifact_count", -1))
            if required_count <= 0 or len(entries) < required_count:
                errors.append("formal artifact count is below its contract")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"artifact manifest is unreadable: {error}")
    return errors


def validate_process_log_manifest(job_root: Path) -> list[str]:
    result_root = job_root / "results" / "qtail_droid_full"
    manifest_path = result_root / "droid_process_log_manifest.json"
    expected_names = {
        "droid_full_pipeline.log",
        "droid_feature_prewarm.log",
        "pipeline_watchdog.log",
        "progress_loop.log",
        "progress_refresh.log",
        "pipeline_generation_handoff.log",
        "manual_endpoint_generation_handoff.log",
        "qtail-web-services.log",
    }
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
        contract = manifest.get("contract", {})
        entries = manifest.get("logs", [])
        by_name = {
            Path(str(entry.get("path", ""))).name: entry
            for entry in entries
            if isinstance(entry, dict)
        }
        if (
            manifest.get("status") != "complete"
            or manifest.get("missing_required") != []
            or contract.get("snapshot_is_immutable") is not True
            or contract.get("live_logs_continue_after_snapshot") is not True
            or int(contract.get("required_log_count", -1)) != 8
            or int(contract.get("captured_required_log_count", -1)) != 8
            or not expected_names.issubset(by_name)
        ):
            errors.append("process-log manifest contract is incomplete")
        for name in expected_names:
            path = result_root / "process_logs_final" / name
            entry = by_name.get(name, {})
            try:
                if (
                    str(entry.get("path")) != str(path)
                    or path.stat().st_size <= 0
                    or path.stat().st_size != int(entry.get("bytes", -1))
                    or sha256(path) != entry.get("sha256")
                    or int(entry.get("line_count", 0)) <= 0
                    or not str(entry.get("role", "")).strip()
                ):
                    errors.append(f"process-log snapshot is invalid: {name}")
            except (OSError, TypeError, ValueError):
                errors.append(f"process-log snapshot is unreadable: {name}")

        qa_path = result_root / "final_page_qa.json"
        qa_sealing_token = (
            "QTAIL_TERMINAL qa_sealing_complete "
            f"qa_sha256={sha256(qa_path)}"
        )
        terminal_requirements = (
            (
                "QTAIL_TERMINAL checksum_complete",
                job_root / "manifests" / "DROID_CHECKSUM_VERIFIED",
            ),
            (
                "QTAIL_TERMINAL record_closure_complete records=187891",
                result_root / "droid_incremental_closure_audit.json",
            ),
            (
                "QTAIL_TERMINAL training_complete",
                job_root / "manifests" / "DROID_TRAINING_COMPLETE",
            ),
            (
                qa_sealing_token,
                qa_path,
            ),
        )
        log_text_by_path: dict[Path, str] = {}
        for name in expected_names:
            path = result_root / "process_logs_final" / name
            try:
                log_text_by_path[path] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
        for token, evidence_path in terminal_requirements:
            matching_logs = [
                path
                for path, text in log_text_by_path.items()
                if token in text
            ]
            if not matching_logs:
                errors.append(f"process logs lack terminal evidence: {token}")
                continue
            try:
                evidence_mtime = evidence_path.stat().st_mtime_ns
                if max(path.stat().st_mtime_ns for path in matching_logs) < (
                    evidence_mtime
                ):
                    errors.append(
                        "process-log terminal evidence predates artifact: "
                        f"{token}"
                    )
            except OSError:
                errors.append(
                    f"terminal evidence artifact is unreadable: {evidence_path}"
                )

        artifact_manifest = read_json(
            result_root / "droid_artifact_manifest.json"
        )
        artifact_entries = {
            str(entry.get("path", "")): entry
            for entry in artifact_manifest.get("artifacts", [])
            if isinstance(entry, dict)
        }
        for path in [
            manifest_path,
            *(
                result_root / "process_logs_final" / name
                for name in sorted(expected_names)
            ),
        ]:
            entry = artifact_entries.get(str(path), {})
            if (
                not entry
                or path.stat().st_size != int(entry.get("bytes", -1))
                or sha256(path) != entry.get("sha256")
            ):
                errors.append(
                    f"artifact manifest does not bind process log: {path}"
                )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"process-log manifest is unreadable: {error}")
    return errors


def validate_bound_artifacts(
    raw_entries: Any,
    expected_paths: list[Path],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw_entries, list):
        return ["marker artifacts must be a list"]
    entries = {
        str(item.get("path", "")): item
        for item in raw_entries
        if isinstance(item, dict)
    }
    expected = {str(path) for path in expected_paths}
    if set(entries) != expected:
        errors.append(
            "marker artifact path set differs from the required path set"
        )
    for path in expected_paths:
        key = str(path)
        entry = entries.get(key)
        if not entry:
            continue
        try:
            metadata = path.stat()
            if metadata.st_size != int(entry.get("bytes", -1)):
                errors.append(f"byte mismatch: {path}")
            expected_hash = str(entry.get("sha256", ""))
            if len(expected_hash) != 64 or sha256(path) != expected_hash:
                errors.append(f"sha256 mismatch: {path}")
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"artifact validation failed: {path}: {error}")
    return errors


def validate_checksum_marker(job_root: Path) -> list[str]:
    errors: list[str] = []
    marker_path = job_root / "manifests" / "DROID_CHECKSUM_VERIFIED"
    result_root = job_root / "results" / "qtail_droid_full"
    verification_path = result_root / "download_verification.json"
    download_marker_path = job_root / "manifests" / "DROID_DOWNLOAD_COMPLETE"
    try:
        marker = read_json(marker_path)
        if marker.get("version") != CHECKSUM_MARKER_VERSION:
            errors.append("checksum marker version is invalid")
        if marker.get("status") != "verified":
            errors.append("checksum marker status is not verified")
        expected = {
            "download_verification": verification_path,
            "download_completion_marker": download_marker_path,
        }
        for prefix, path in expected.items():
            if (
                marker.get(prefix) != str(path)
                or marker.get(f"{prefix}_bytes") != path.stat().st_size
                or marker.get(f"{prefix}_sha256") != sha256(path)
            ):
                errors.append(f"checksum marker binding is invalid: {prefix}")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"checksum marker is unreadable: {error}")
    return errors


def validate_current_download_binding(
    job_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    result_root = job_root / "results" / "qtail_droid_full"
    marker_path = job_root / "manifests" / "DROID_DOWNLOAD_COMPLETE"
    errors: list[str] = []
    binding: dict[str, Any] = {}
    try:
        marker = read_json(marker_path)
        binding, checks, file_errors = build_binding(
            data_dir=job_root / "data" / "droid",
            manifest_path=result_root / "droid_object_manifest.json",
            checksum_manifest_path=(
                result_root / "droid_object_checksum_manifest.json"
            ),
            checksum_ledger_path=(
                result_root / "droid_object_checksum_ledger.json"
            ),
            transport_status_path=(
                result_root / "parallel_download_status.json"
            ),
            expected_bytes=FORMAL_EXPECTED_BYTES,
            expected_objects=FORMAL_EXPECTED_OBJECTS,
            expected_tfrecords=FORMAL_EXPECTED_TFRECORDS,
        )
        if (
            marker.get("marker_version") != DOWNLOAD_MARKER_VERSION
            or marker.get("status") != "complete"
            or marker.get("immutable") is not True
            or marker.get("binding") != binding
            or not all(checks.values())
            or file_errors
        ):
            errors.append(
                "current DROID files, checksum ledger, transport status, "
                "and immutable download marker do not share one binding"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"current DROID mirror binding is unreadable: {error}")
    return errors, binding


def validate_transition_gate(path: Path, label: str) -> list[str]:
    errors: list[str] = []
    required_checks = {
        "guard_status_passed",
        "guard_heartbeat_fresh",
        "uniclash_core_running",
        "uniclash_tun_disabled",
        "droid_bypass_policy_enabled",
        "expected_interface_bound",
        "system_proxy_bypass_passed",
        "cumulative_history_clean",
        "live_transfers_clean_and_direct",
    }
    try:
        payload = read_json(path)
        checks = payload.get("checks", {})
        if (
            payload.get("status") != "passed"
            or not isinstance(checks, dict)
            or not required_checks.issubset(checks)
            or any(checks.get(name) is not True for name in required_checks)
            or payload.get("global_violations") != []
            or payload.get("transfer_violations") != []
            or float(payload.get("guard_age_seconds", float("inf")))
            > float(payload.get("max_guard_age_seconds", -1))
        ):
            errors.append(f"{label} UniClash transition gate is invalid")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"{label} UniClash transition gate is unreadable: {error}")
    return errors


def validate_hardening_selftest(
    path: Path,
    expected_control_names: frozenset[str] = HARDENING_SELFTEST_CONTROL_NAMES,
) -> list[str]:
    try:
        payload = read_json(path)
        controls = payload.get("controls", [])
        control_names = [
            str(control.get("name"))
            for control in controls
            if isinstance(control, dict)
        ]
        if (
            payload.get("status") != "passed"
            or not isinstance(controls, list)
            or len(controls) != len(expected_control_names)
            or len(control_names) != len(controls)
            or set(control_names) != expected_control_names
            or int(payload.get("controls_passed", -1)) != len(controls)
            or int(payload.get("controls_total", -1)) != len(controls)
            or any(
                not isinstance(control, dict)
                or control.get("passed") is not True
                for control in controls
            )
        ):
            return [f"hardening self-test is incomplete: {path.name}"]
    except (OSError, ValueError, TypeError) as error:
        return [f"hardening self-test is unreadable: {path.name}: {error}"]
    return []


def validate_incremental_closure_selftest_payload(
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    checks = payload.get("checks")
    cases = payload.get("cases")
    if (
        payload.get("format_version")
        != "qtail_droid_incremental_closure_selftest_v2"
    ):
        errors.append("incremental closure control version is invalid")
    if payload.get("status") != "passed":
        errors.append("incremental closure control status is not passed")
    if not isinstance(checks, dict):
        errors.append("incremental closure checks are not an object")
    elif set(checks) != INCREMENTAL_CLOSURE_SELFTEST_CHECKS:
        errors.append("incremental closure checks are not the exact 7")
    elif any(
        checks.get(name) is not True
        for name in INCREMENTAL_CLOSURE_SELFTEST_CHECKS
    ):
        errors.append("incremental closure checks are not all true")
    if payload.get("failed_checks") != []:
        errors.append("incremental closure controls report failed checks")
    if not isinstance(cases, list) or len(cases) != len(
        INCREMENTAL_CLOSURE_SELFTEST_CHECKS
    ):
        errors.append("incremental closure cases are not the exact 7")
        return errors
    if any(not isinstance(case, dict) for case in cases):
        errors.append("incremental closure case entry is invalid")
        return errors
    case_by_name = {
        str(case.get("name")): case
        for case in cases
        if isinstance(case, dict)
    }
    if set(case_by_name) != INCREMENTAL_CLOSURE_SELFTEST_CHECKS:
        errors.append("incremental closure case names are not the exact 7")
        return errors
    if any(
        case.get("passed") is not True for case in case_by_name.values()
    ):
        errors.append("incremental closure cases are not all passed")

    positive = case_by_name["positive_current_closure"]
    if (
        positive.get("expected_success") is not True
        or int(positive.get("returncode", -1)) != 0
    ):
        errors.append("positive incremental closure control is invalid")

    formal = case_by_name["require_formal_matches_exact_full_gate"]
    if (
        formal.get("expected_success") is not False
        or int(formal.get("returncode", 0)) == 0
        or formal.get("formal_full_mirror_gate") is not False
        or formal.get("expected_formal_full_mirror_gate") is not False
    ):
        errors.append("formal exact-full-gate rejection control is invalid")

    for name in (
        "record_count_tamper_rejected",
        "md5_ledger_tamper_rejected",
        "md5_after_error_sample_limit_rejected",
        "missing_listed_cache_rejected",
    ):
        case = case_by_name[name]
        if (
            case.get("expected_success") is not False
            or int(case.get("returncode", 0)) == 0
        ):
            errors.append(f"destructive closure control is invalid: {name}")

    deferred = case_by_name["post_snapshot_tfrecord_is_deferred"]
    if (
        deferred.get("expected_success") is not True
        or int(deferred.get("returncode", -1)) != 0
        or int(deferred.get("deferred_after_snapshot_count", -1)) != 1
        or int(
            deferred.get(
                "expected_deferred_after_snapshot_count",
                -1,
            )
        )
        != 1
        or deferred.get("formal_full_mirror_gate") is not False
        or deferred.get("expected_formal_full_mirror_gate") is not False
    ):
        errors.append("post-snapshot deferral control is invalid")
    return errors


def validate_training_semantics(job_root: Path) -> list[str]:
    result_root = job_root / "results" / "qtail_droid_full"
    errors: list[str] = []
    errors.extend(validate_checksum_marker(job_root))
    binding_errors, current_download_binding = (
        validate_current_download_binding(job_root)
    )
    errors.extend(binding_errors)
    for filename, label in (
        ("uniclash_checksum_handoff_gate.json", "checksum handoff"),
        ("uniclash_pre_environment_gate.json", "environment capture"),
        ("uniclash_pre_training_gate.json", "training launch"),
    ):
        errors.extend(validate_transition_gate(result_root / filename, label))
    errors.extend(
        validate_pipeline_generation_gate(
            result_root / "pipeline_generation_gate.json"
        )
    )
    hardening_reports = {
        "droid_stage_marker_hardening_selftest.json": (
            HARDENING_SELFTEST_CONTROL_NAMES
        ),
        "droid_progress_preview_selftest.json": (
            PROGRESS_PREVIEW_SELFTEST_CONTROL_NAMES
        ),
        "droid_artifact_manifest_merge_selftest.json": (
            ARTIFACT_MANIFEST_SELFTEST_CONTROL_NAMES
        ),
        "droid_pipeline_shell_contract_selftest.json": (
            PIPELINE_SHELL_SELFTEST_CONTROL_NAMES
        ),
    }
    for filename, expected_names in hardening_reports.items():
        errors.extend(
            validate_hardening_selftest(
                result_root / filename,
                expected_names,
            )
        )
    errors.extend(
        validate_artifact_manifest_entries(
            result_root / "droid_training_artifact_manifest.json"
        )
    )
    try:
        marker_selftest = read_json(
            result_root / "droid_download_marker_selftest.json"
        )
        controls = marker_selftest.get("controls", [])
        if (
            marker_selftest.get("status") != "passed"
            or int(marker_selftest.get("controls_passed", -1)) != 8
            or int(marker_selftest.get("controls_total", -1)) != 8
            or len(controls) != 8
            or not all(
                isinstance(control, dict)
                and control.get("passed") is True
                for control in controls
            )
        ):
            errors.append("download-marker controls are incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"download-marker controls are unreadable: {error}")

    try:
        mirror_selftest = read_json(
            result_root / "droid_mirror_verifier_selftest.json"
        )
        controls = mirror_selftest.get("controls", [])
        if (
            mirror_selftest.get("status") != "passed"
            or int(mirror_selftest.get("controls_passed", -1)) != 8
            or int(mirror_selftest.get("controls_total", -1)) != 8
            or len(controls) != 8
            or not all(
                isinstance(control, dict)
                and control.get("passed") is True
                for control in controls
            )
        ):
            errors.append("mirror-verifier controls are incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"mirror-verifier controls are unreadable: {error}")

    try:
        single_writer = read_json(
            result_root / "droid_downloader_single_writer_selftest.json"
        )
        checks = single_writer.get("checks", {})
        if (
            single_writer.get("status") != "passed"
            or int(single_writer.get("checks_passed", -1)) != 13
            or int(single_writer.get("checks_total", -1)) != 13
            or len(checks) != 13
            or not all(value is True for value in checks.values())
        ):
            errors.append("downloader single-writer controls are incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(
            f"downloader single-writer controls are unreadable: {error}"
        )

    try:
        runtime_process = read_json(
            result_root / "droid_runtime_process_contract_selftest.json"
        )
        checks = runtime_process.get("checks", {})
        if (
            runtime_process.get("status") != "passed"
            or runtime_process.get("control")
            != "droid_runtime_process_contract_v11"
            or int(runtime_process.get("checks_passed", -1)) != 16
            or int(runtime_process.get("checks_total", -1)) != 16
            or len(checks) != 16
            or not all(value is True for value in checks.values())
        ):
            errors.append("runtime process controls are incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"runtime process controls are unreadable: {error}")

    try:
        gate_order = read_json(
            result_root / "droid_training_gate_order_selftest.json"
        )
        controls = gate_order.get("controls", [])
        if (
            gate_order.get("version")
            != "qtail_droid_training_gate_order_selftest_v2"
            or gate_order.get("status") != "passed"
            or int(gate_order.get("controls_passed", -1)) != 11
            or int(gate_order.get("controls_total", -1)) != 11
            or len(controls) != 11
            or not all(
                isinstance(control, dict)
                and control.get("passed") is True
                for control in controls
            )
        ):
            errors.append("training gate-order controls are incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"training gate-order controls are unreadable: {error}")

    try:
        pre_checksum_gate = read_json(
            result_root / "uniclash_pre_checksum_gate.json"
        )
        checks = pre_checksum_gate.get("checks", {})
        if (
            pre_checksum_gate.get("status") != "passed"
            or int(pre_checksum_gate.get("checks_passed", -1)) != 10
            or int(pre_checksum_gate.get("checks_total", -1)) != 10
            or len(checks) != 10
            or not all(value is True for value in checks.values())
        ):
            errors.append("UniClash pre-checksum gate is incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"UniClash pre-checksum gate is unreadable: {error}")

    try:
        pre_checksum_selftest = read_json(
            result_root / "uniclash_pre_checksum_gate_selftest.json"
        )
        checks = pre_checksum_selftest.get("checks", {})
        if (
            pre_checksum_selftest.get("status") != "passed"
            or int(pre_checksum_selftest.get("checks_passed", -1))
            != len(UNICLASH_GATE_SELFTEST_CHECKS)
            or int(pre_checksum_selftest.get("checks_total", -1))
            != len(UNICLASH_GATE_SELFTEST_CHECKS)
            or set(checks) != UNICLASH_GATE_SELFTEST_CHECKS
            or not all(value is True for value in checks.values())
        ):
            errors.append(
                "UniClash pre-checksum gate controls are incomplete"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(
            f"UniClash pre-checksum gate controls are unreadable: {error}"
        )

    live_partial_path = (
        result_root / "droid_live_partial_marker_rejection.json"
    )
    if live_partial_path.is_file():
        try:
            live_partial = read_json(live_partial_path)
            if (
                live_partial.get("status") != "passed"
                or live_partial.get("formal_completion_evidence") is not False
                or live_partial.get("precondition", {}).get("passed") is not True
                or live_partial.get("result", {}).get("rejected") is not True
                or live_partial.get("result", {}).get("marker_created")
                is not False
            ):
                errors.append(
                    "live partial-mirror marker rejection is invalid"
                )
        except (OSError, ValueError, TypeError) as error:
            errors.append(
                "live partial-mirror marker rejection is unreadable: "
                f"{error}"
            )

    try:
        download_marker = read_json(
            result_root / "download_completion_marker.json"
        )
        binding = download_marker.get("binding", {})
        checks = binding.get("checks", {})
        if (
            download_marker.get("marker_version")
            != "droid_download_completion_marker_v1"
            or download_marker.get("status") != "complete"
            or download_marker.get("immutable") is not True
            or int(binding.get("object_count", -1)) != 4_102
            or int(binding.get("tfrecord_count", -1)) != 4_096
            or int(binding.get("official_bytes", -1))
            != 3_700_745_265_151
            or int(binding.get("local_bytes", -1))
            != 3_700_745_265_151
            or len(checks) != 13
            or not all(value is True for value in checks.values())
        ):
            errors.append("download completion marker is incomplete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"download completion marker is unreadable: {error}")

    try:
        verification = read_json(result_root / "download_verification.json")
        if (
            verification.get("status") != "complete"
            or verification.get("ready_for_full_allocation_training") is not True
            or int(verification.get("manifest_object_count", -1)) != 4_102
            or int(verification.get("complete_tfrecord_count", -1)) != 4_096
            or int(verification.get("missing_object_count", -1)) != 0
            or int(verification.get("size_mismatch_count", -1)) != 0
            or int(verification.get("checksum_error_count", -1)) != 0
        ):
            errors.append("download verification is not a complete 4,102-object mirror")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"download verification is unreadable: {error}")

    try:
        metadata = read_json(
            result_root / "droid_release_metadata_audit.json"
        )
        gates = metadata.get("gates", {})
        combined = metadata.get("combined_official_metadata", {})
        releases = {
            str(item.get("release")): item
            for item in metadata.get("releases", [])
            if isinstance(item, dict)
        }
        expected_releases = {
            "1.0.0": {
                "dataset_name": "r2d2_faceblur",
                "dataset_version": "1.4.0",
                "shards": 2_048,
                "records": 92_233,
                "split_bytes": 1_834_749_018_029,
            },
            "1.0.1": {
                "dataset_name": "droid_101",
                "dataset_version": "0.0.1",
                "shards": 2_048,
                "records": 95_658,
                "split_bytes": 1_865_993_126_270,
            },
        }
        expected_gates = {
            "official_checksum_manifest",
            "both_releases_verified",
            "combined_shards_4096",
            "combined_records_187891",
            "combined_split_bytes_match",
            "step_schemas_identical",
            "training_features_present",
        }
        metadata_valid = bool(
            metadata.get("version") == "droid_release_metadata_audit_v1"
            and metadata.get("status") == "verified"
            and isinstance(gates, dict)
            and set(gates) == expected_gates
            and all(value is True for value in gates.values())
            and int(combined.get("tfrecord_shards", -1)) == 4_096
            and int(combined.get("records", -1)) == 187_891
            and int(combined.get("split_bytes", -1))
            == 3_700_742_144_299
            and set(releases) == set(expected_releases)
        )
        for release, expected in expected_releases.items():
            item = releases.get(release, {})
            metadata_valid = bool(
                metadata_valid
                and item.get("verified") is True
                and item.get("dataset_name") == expected["dataset_name"]
                and item.get("dataset_version")
                == expected["dataset_version"]
                and int(item.get("official_tfrecord_shards", -1))
                == expected["shards"]
                and int(item.get("official_records", -1))
                == expected["records"]
                and int(item.get("official_split_bytes", -1))
                == expected["split_bytes"]
                and item.get("required_training_features_present") is True
            )
        if not metadata_valid:
            errors.append(
                "official release metadata audit is not exact for both releases"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"release metadata audit is unreadable: {error}")

    try:
        protocol = read_json(result_root / "droid_protocol_selftest.json")
        protocol_checks = protocol.get("checks", {})
        if (
            protocol.get("status") != "passed"
            or len(protocol_checks) < 30
            or not all(protocol_checks.values())
            or protocol_checks.get(
                "intermediate_checkpoint_manifest_exact_grid"
            )
            is not True
            or protocol_checks.get(
                "unexpected_intermediate_checkpoint_rejected"
            )
            is not True
        ):
            errors.append(
                "protocol self-test has fewer than 30 passing checks "
                "or lacks checkpoint-manifest controls"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"protocol self-test is unreadable: {error}")

    try:
        environment_test = read_json(
            result_root / "droid_environment_contract_selftest.json"
        )
        if (
            environment_test.get("status") != "passed"
            or environment_test.get("contract_version")
            != "qtail_droid_environment_contract_selftest_v3"
            or len(environment_test.get("checks", {})) != 9
            or not all(environment_test.get("checks", {}).values())
        ):
            errors.append(
                "environment contract self-test is not exact v3 9/9"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"environment contract self-test is unreadable: {error}")

    environment_errors, environment = validate_environment_code_manifest(
        result_root / "droid_environment_manifest.json"
    )
    errors.extend(environment_errors)

    try:
        cache = read_json(result_root / "droid_feature_cache_verification.json")
        if (
            cache.get("status") != "verified"
            or cache.get("all_official_tfrecords") is not True
            or cache.get("full_official_record_count_match") is not True
            or cache.get("all_feature_values_recomputed") is not True
            or int(cache.get("verified_cache_count", -1)) != 4_096
            or int(cache.get("recomputed_feature_count", -1)) != 4_096
            or int(cache.get("error_count", -1)) != 0
        ):
            errors.append("feature cache is not independently verified 4,096/4,096")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"feature-cache verification is unreadable: {error}")

    try:
        closure = read_json(
            result_root / "droid_incremental_closure_audit.json"
        )
        current = closure.get("current_closure", {})
        checks = closure.get("checks", {})
        if (
            closure.get("format_version")
            != "qtail_droid_incremental_closure_v2"
            or closure.get("status") != "complete"
            or closure.get("formal_full_mirror_gate") is not True
            or int(current.get("verified_objects", -1)) != 4_102
            or int(current.get("completed_tfrecords", -1)) != 4_096
            or int(current.get("listed_verified_caches", -1)) != 4_096
            or int(current.get("decoded_records", -1)) != 187_891
            or int(current.get("transport_partial_files", -1)) != 0
            or int(
                current.get("deferred_after_snapshot_tfrecords", -1)
            )
            != 0
            or int(current.get("missing_from_snapshot_tfrecords", -1))
            != 0
            or len(checks) < 13
            or not all(value is True for value in checks.values())
            or int(closure.get("error_count", -1)) != 0
            or closure.get("failed_checks") != []
        ):
            errors.append(
                "incremental MD5/record/cache closure is not formally complete"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"incremental closure is unreadable: {error}")

    try:
        closure_selftest = read_json(
            result_root / "droid_incremental_closure_selftest.json"
        )
        errors.extend(
            validate_incremental_closure_selftest_payload(closure_selftest)
        )
    except (OSError, ValueError, TypeError) as error:
        errors.append(
            f"incremental closure controls are unreadable: {error}"
        )

    try:
        milestone_status = read_json(
            result_root / "droid_release_milestone_status.json"
        )
        release_rows = {
            str(item.get("release")): item
            for item in milestone_status.get("releases", [])
            if isinstance(item, dict)
        }
        expected_release_counts = {
            "1.0.0": (2_051, 2_048, 92_233, 1_834_750_493_757),
            "1.0.1": (2_051, 2_048, 95_658, 1_865_994_656_798),
        }
        milestone_valid = bool(
            milestone_status.get("format_version")
            == "qtail_droid_release_milestone_status_v1"
            and milestone_status.get("status") == "complete"
            and int(milestone_status.get("release_count", -1)) == 2
            and int(
                milestone_status.get("completed_release_count", -1)
            )
            == 2
            and milestone_status.get("invalid_existing_milestones") == []
            and set(release_rows) == set(expected_release_counts)
        )
        for release, expected in expected_release_counts.items():
            row = release_rows.get(release, {})
            milestone_path = Path(str(row.get("milestone", "")))
            milestone = read_json(milestone_path)
            counts = milestone.get("counts", {})
            milestone_valid = bool(
                milestone_valid
                and row.get("status") == "complete"
                and len(row.get("checks", {})) == 10
                and all(row.get("checks", {}).values())
                and milestone_path.is_file()
                and sha256(milestone_path) == row.get("milestone_sha256")
                and milestone.get("format_version")
                == "qtail_droid_release_milestone_v1"
                and milestone.get("status") == "complete"
                and milestone.get("immutable") is True
                and milestone.get("release") == release
                and int(counts.get("objects", -1)) == expected[0]
                and int(counts.get("tfrecords", -1)) == expected[1]
                and int(counts.get("records", -1)) == expected[2]
                and int(counts.get("object_tfrecord_bytes", -1))
                == expected[3]
                and len(milestone.get("checks", {})) == 10
                and all(milestone.get("checks", {}).values())
            )
        if not milestone_valid:
            errors.append(
                "per-release immutable input milestones are incomplete"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"release milestones are unreadable: {error}")

    checkpoint_contract: dict[str, Any] = {}
    try:
        checkpoint_manifest = read_json(
            result_root / "droid_intermediate_checkpoint_manifest.json"
        )
        checkpoint_contract = checkpoint_manifest.get("contract", {})
        checkpoint_entries = checkpoint_manifest.get("entries", [])
        expected_labels = {
            "evaluation_source",
            "evaluation_qtail",
            "deployment_source",
            "deployment_qtail",
        }
        expected_steps = [0, 5_000, 10_000, 15_000, 20_000]
        observed_pairs = {
            (
                str(entry.get("model_stage", "")),
                int(entry.get("step", -1)),
            )
            for entry in checkpoint_entries
        }
        expected_pairs = {
            (label, step)
            for label in expected_labels
            for step in expected_steps
        }
        checkpoint_errors = []
        try:
            import torch
            from qtail_train_droid_full import (
                CHECKPOINT_CHAIN_VERSION,
                CHECKPOINT_FORMAT_VERSION,
                checkpoint_content_errors,
                state_dict_fingerprint,
                tree_fingerprint,
            )
        except ImportError as error:
            torch = None
            CHECKPOINT_CHAIN_VERSION = None
            CHECKPOINT_FORMAT_VERSION = None
            checkpoint_content_errors = None
            state_dict_fingerprint = None
            tree_fingerprint = None
            checkpoint_errors.append(f"torch unavailable: {error}")
        for entry in checkpoint_entries:
            path = Path(str(entry.get("path", "")))
            try:
                payload = (
                    torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if torch is not None
                    else {}
                )
                step = int(entry.get("step", -1))
                step_zero_fingerprint_valid = bool(
                    step != 0
                    or (
                        state_dict_fingerprint is not None
                        and isinstance(payload.get("state_dict"), dict)
                        and state_dict_fingerprint(
                            payload["state_dict"]
                        )
                        == entry.get("initialized_state_sha256")
                    )
                )
                if (
                    path.stat().st_size != int(entry.get("bytes", -1))
                    or sha256(path) != entry.get("sha256")
                    or not isinstance(payload, dict)
                    or payload.get("format_version")
                    != CHECKPOINT_FORMAT_VERSION
                    or payload.get("checkpoint_chain_version")
                    != CHECKPOINT_CHAIN_VERSION
                    or payload.get("model")
                    != entry.get("model_stage")
                    or int(payload.get("step", -1))
                    != int(entry.get("step", -2))
                    or int(payload.get("steps", -1)) != 20_000
                    or int(payload.get("seed", -1)) != 11
                    or int(
                        payload.get(
                            "optimizer_updates_completed", -1
                        )
                    )
                    != int(entry.get("step", -2))
                    or payload.get("device") != entry.get("device")
                    or payload.get("optimizer")
                    != entry.get("optimizer")
                    or payload.get("environment_fingerprint")
                    != entry.get("environment_fingerprint")
                    or payload.get("training_signature")
                    != entry.get("training_signature")
                    or payload.get("feature_sha256")
                    != entry.get("feature_sha256")
                    or payload.get("initialized_state_sha256")
                    != entry.get("initialized_state_sha256")
                    or payload.get("model_state_sha256")
                    != entry.get("model_state_sha256")
                    or payload.get("optimizer_state_sha256")
                    != entry.get("optimizer_state_sha256")
                    or payload.get("parent_checkpoint_name")
                    != entry.get("parent_checkpoint_name")
                    or payload.get("parent_checkpoint_step")
                    != entry.get("parent_checkpoint_step")
                    or payload.get("parent_checkpoint_sha256")
                    != entry.get("parent_checkpoint_sha256")
                    or not isinstance(payload.get("state_dict"), dict)
                    or not payload.get("state_dict")
                    or not isinstance(
                        payload.get("optimizer_state_dict"), dict
                    )
                    or not payload.get("optimizer_state_dict")
                    or entry.get("device") != "mps"
                    or entry.get("optimizer")
                    != "AdamW(lr=0.002, weight_decay=0.0001)"
                    or len(
                        str(entry.get("training_signature", ""))
                    )
                    != 64
                    or len(
                        str(entry.get("environment_fingerprint", ""))
                    )
                    != 64
                    or len(str(entry.get("feature_sha256", ""))) != 64
                    or len(
                        str(entry.get("initialized_state_sha256", ""))
                    )
                    != 64
                    or len(str(entry.get("model_state_sha256", ""))) != 64
                    or len(
                        str(entry.get("optimizer_state_sha256", ""))
                    )
                    != 64
                    or state_dict_fingerprint is None
                    or tree_fingerprint is None
                    or state_dict_fingerprint(payload["state_dict"])
                    != entry.get("model_state_sha256")
                    or tree_fingerprint(payload["optimizer_state_dict"])
                    != entry.get("optimizer_state_sha256")
                    or checkpoint_content_errors is None
                    or checkpoint_content_errors(payload)
                    or not step_zero_fingerprint_valid
                ):
                    checkpoint_errors.append(str(path))
            except (
                EOFError,
                OSError,
                pickle.UnpicklingError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                checkpoint_errors.append(str(path))
        feature_signatures = {
            label: {
                str(entry.get("feature_sha256", ""))
                for entry in checkpoint_entries
                if entry.get("model_stage") == label
            }
            for label in expected_labels
        }
        initialization_signatures = {
            label: {
                str(entry.get("initialized_state_sha256", ""))
                for entry in checkpoint_entries
                if entry.get("model_stage") == label
            }
            for label in expected_labels
        }
        paired_feature_signatures_equal = bool(
            len(feature_signatures["evaluation_source"]) == 1
            and feature_signatures["evaluation_source"]
            == feature_signatures["evaluation_qtail"]
            and len(feature_signatures["deployment_source"]) == 1
            and feature_signatures["deployment_source"]
            == feature_signatures["deployment_qtail"]
        )
        initialized_state_signatures_equal = bool(
            all(
                len(values) == 1
                for values in initialization_signatures.values()
            )
            and len(
                {
                    next(iter(values))
                    for values in initialization_signatures.values()
                }
            )
            == 1
        )
        parent_chains_valid = True
        for label in expected_labels:
            label_entries = sorted(
                (
                    entry
                    for entry in checkpoint_entries
                    if entry.get("model_stage") == label
                ),
                key=lambda entry: int(entry.get("step", -1)),
            )
            previous = None
            for entry in label_entries:
                if previous is None:
                    expected_parent_name = None
                    expected_parent_step = None
                    expected_parent_sha256 = None
                else:
                    expected_parent_name = Path(
                        str(previous.get("path", ""))
                    ).name
                    expected_parent_step = int(previous.get("step", -1))
                    expected_parent_sha256 = previous.get("sha256")
                parent_chains_valid = bool(
                    parent_chains_valid
                    and entry.get("parent_checkpoint_name")
                    == expected_parent_name
                    and entry.get("parent_checkpoint_step")
                    == expected_parent_step
                    and entry.get("parent_checkpoint_sha256")
                    == expected_parent_sha256
                )
                previous = entry
        if (
            checkpoint_manifest.get("status") != "complete"
            or checkpoint_manifest.get("errors") != []
            or int(
                checkpoint_contract.get(
                    "expected_checkpoint_count", -1
                )
            )
            != 20
            or checkpoint_contract.get("expected_steps")
            != expected_steps
            or set(checkpoint_contract.get("model_stages", []))
            != expected_labels
            or checkpoint_contract.get("checkpoint_format_version")
            != CHECKPOINT_FORMAT_VERSION
            or checkpoint_contract.get("checkpoint_chain_version")
            != CHECKPOINT_CHAIN_VERSION
            or checkpoint_contract.get(
                "checkpoint_content_hashes_recomputed"
            )
            is not True
            or checkpoint_contract.get(
                "parent_checkpoint_hash_chains_verified"
            )
            is not True
            or int(
                checkpoint_manifest.get(
                    "actual_checkpoint_count", -1
                )
            )
            != 20
            or len(checkpoint_entries) != 20
            or observed_pairs != expected_pairs
            or checkpoint_contract.get(
                "paired_feature_signatures_equal"
            )
            is not True
            or checkpoint_contract.get(
                "initialized_state_signatures_equal"
            )
            is not True
            or not paired_feature_signatures_equal
            or not initialized_state_signatures_equal
            or not parent_chains_valid
            or checkpoint_errors
        ):
            errors.append(
                "intermediate checkpoints are not a verified 4x5 grid"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(
            f"intermediate checkpoint manifest is unreadable: {error}"
        )

    try:
        report = read_json(result_root / "droid_full_training_report.json")
        formal = report.get("formal_protocol", {})
        holdout = report.get("holdout_evaluation", {})
        input_audit = report.get("input_audit", {})
        trajectory = report.get("trajectory_evidence", {})
        effect = report.get("effect_metrics", {})
        rare_coverage = report.get(
            "rare_instruction_fingerprint_coverage",
            {},
        )
        rare_coverage_artifact = read_json(
            result_root
            / "droid_rare_instruction_fingerprint_coverage.json"
        )
        gate = effect.get("hypothesis_gate", {})
        bootstrap = effect.get("paired_bootstrap", {})
        randomization = effect.get(
            "paired_arm_swap_randomization", {}
        )
        compute = report.get("compute_audit", {})
        checkpoint_audit = report.get(
            "intermediate_checkpoint_audit", {}
        )
        environment_code_binding = report.get(
            "environment_code_binding", {}
        )
        environment_manifest_path = (
            result_root / "droid_environment_manifest.json"
        )
        tail_gain = (
            float(effect["qtail_pred_tail_share"])
            - float(effect["source_pred_tail_share"])
        ) * 100.0
        extreme_reduction = (
            float(effect["source_extreme_underallocation_rate"])
            - float(effect["qtail_extreme_underallocation_rate"])
        ) * 100.0
        expected_supported = (
            tail_gain >= 2.0
            and float(bootstrap["ci95_low_pp"]) >= 2.0
            and extreme_reduction > 0.0
        )
        expected_not_supported = (
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
        holdout_relative_paths_raw = holdout.get(
            "holdout_relative_paths", []
        )
        holdout_relative_paths_typed = bool(
            isinstance(holdout_relative_paths_raw, list)
            and all(
                isinstance(value, str)
                for value in holdout_relative_paths_raw
            )
        )
        holdout_relative_paths = (
            holdout_relative_paths_raw
            if holdout_relative_paths_typed
            else []
        )
        holdout_relative_path_sha256 = hashlib.sha256(
            "\n".join(holdout_relative_paths).encode("utf-8")
        ).hexdigest()
        rare_status = rare_coverage.get("status")
        rare_status_valid = rare_status in {
            "complete",
            "no_eligible_fingerprints",
        }
        rare_shape_valid = (
            (
                rare_status == "complete"
                and int(
                    rare_coverage.get(
                        "rare_holdout_fingerprint_count", 0
                    )
                )
                > 0
                and [
                    int(item.get("draw_budget", -1))
                    for item in rare_coverage.get("curve", [])
                ]
                == [10, 25, 50, 100, 200, 400, 800]
            )
            or (
                rare_status == "no_eligible_fingerprints"
                and int(
                    rare_coverage.get(
                        "rare_holdout_fingerprint_count", -1
                    )
                )
                == 0
                and int(
                    rare_coverage.get(
                        "unseen_in_training_fingerprint_count", -1
                    )
                )
                == 0
                and rare_coverage.get("curve") == []
                and rare_coverage.get("time_to_coverage") == []
                and bool(rare_coverage.get("status_reason"))
            )
        )
        runtime_environment_fingerprint = str(
            compute.get("runtime_environment_fingerprint", "")
        )
        checkpoint_environment_fingerprint = str(
            compute.get("checkpoint_environment_fingerprint", "")
        )
        checkpoint_environment = compute.get(
            "checkpoint_environment_contract", {}
        )
        checkpoint_formal_binding = checkpoint_environment.get(
            "formal_environment_binding", {}
        ) if isinstance(checkpoint_environment, dict) else {}
        resume_environment_valid = all(
            item.get("environment_fingerprint")
            == checkpoint_environment_fingerprint
            and (
                not item.get("resumed")
                or item.get("checkpoint_environment_fingerprint")
                == checkpoint_environment_fingerprint
            )
            for item in compute.get("resume", {}).values()
        )
        if (
            report.get("status") != "complete"
            or report.get("training_scope")
            != "all_complete_shards_all_decodable_records"
            or formal.get("locked") is not True
            or int(formal.get("seed", -1)) != 11
            or int(formal.get("steps_per_stage", -1)) != 20_000
            or int(formal.get("holdout_shards_per_release", -1)) != 410
            or int(formal.get("bootstrap_samples", -1)) != 5_000
            or int(formal.get("randomization_samples", -1)) != 5_000
            or int(formal.get("checkpoint_every_steps", -1)) != 5_000
            or float(formal.get("min_record_parse_rate", -1.0)) != 1.0
            or float(
                formal.get("min_record_scan_complete_rate", -1.0)
            )
            != 1.0
            or formal.get("require_verified_mirror") is not True
            or formal.get("pt_source_sha256") != FORMAL_PT_SOURCE_SHA256
            or formal.get("holdout_relative_path_sha256")
            != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
            or formal.get("holdout_membership_path_scope")
            != "official_release_relative_path"
            or holdout.get("version")
            != "release_stratified_official_relative_path_hash_v2"
            or holdout.get("membership_path_scope")
            != "official_release_relative_path"
            or holdout.get("holdout_membership_locked") is not True
            or int(holdout.get("training_shards", -1)) != 3_276
            or int(holdout.get("holdout_shards", -1)) != 820
            or not holdout_relative_paths_typed
            or len(holdout_relative_paths) != 820
            or holdout_relative_paths != sorted(holdout_relative_paths)
            or len(set(holdout_relative_paths)) != 820
            or holdout_relative_path_sha256
            != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
            or holdout.get("holdout_relative_path_sha256")
            != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
            or input_audit.get("verified") is not True
            or int(input_audit.get("formal_expected_object_count", -1))
            != FORMAL_EXPECTED_OBJECTS
            or int(input_audit.get("formal_expected_tfrecord_shards", -1))
            != FORMAL_EXPECTED_TFRECORDS
            or int(input_audit.get("formal_expected_total_bytes", -1))
            != FORMAL_EXPECTED_BYTES
            or input_audit.get("current_binding") != current_download_binding
            or environment_code_binding.get("passed") is not True
            or environment_code_binding.get("required") is not True
            or environment_code_binding.get("manifest")
            != str(environment_manifest_path)
            or environment_code_binding.get("manifest_sha256")
            != sha256(environment_manifest_path)
            or int(
                environment_code_binding.get("checked_code_entries", -1)
            )
            != len(environment.get("code", []))
            or int(environment_code_binding.get("mismatch_count", -1)) != 0
            or environment_code_binding.get("errors") != []
            or environment_code_binding.get(
                "snapshot_code_parity_passed"
            )
            is not True
            or environment_code_binding.get("snapshot_manifest_sha256")
            != environment.get("orchestration_snapshot", {}).get(
                "manifest_sha256"
            )
            or int(trajectory.get("tfrecord_shards_parsed", -1))
            != FORMAL_EXPECTED_TFRECORDS
            or int(trajectory.get("records_decoded", -1))
            != FORMAL_EXPECTED_RECORDS
            or float(trajectory.get("record_parse_rate", -1.0)) != 1.0
            or float(
                trajectory.get("record_scan_complete_rate", -1.0)
            )
            != 1.0
            or len(runtime_environment_fingerprint) != 64
            or len(checkpoint_environment_fingerprint) != 64
            or checkpoint_contract.get("environment_fingerprint")
            != checkpoint_environment_fingerprint
            or checkpoint_environment.get("version")
            != "qtail_checkpoint_environment_v2"
            or checkpoint_environment.get("formal_run") is not True
            or checkpoint_environment.get("runtime_environment")
            != compute.get("runtime_environment")
            or checkpoint_formal_binding.get("required") is not True
            or checkpoint_formal_binding.get("passed") is not True
            or checkpoint_formal_binding.get(
                "environment_manifest_sha256"
            )
            != environment_code_binding.get("manifest_sha256")
            or checkpoint_formal_binding.get(
                "checked_code_aggregate_sha256"
            )
            != environment_code_binding.get(
                "checked_code_aggregate_sha256"
            )
            or checkpoint_formal_binding.get(
                "orico_snapshot_manifest_sha256"
            )
            != environment_code_binding.get("snapshot_manifest_sha256")
            or checkpoint_formal_binding.get(
                "snapshot_code_parity_passed"
            )
            is not True
            or compute.get("same_environment_fingerprint") is not True
            or not resume_environment_valid
            or bootstrap.get("p_gain_le_zero_is_p_value") is not False
            or randomization.get("version")
            != "paired_shard_arm_swap_diagnostic_v2"
            or int(randomization.get("samples", -1)) != 5_000
            or randomization.get("unit")
            != "non_independent_heldout_shard_weight"
            or randomization.get("finite_sample_correction")
            != "(k+1)/(B+1)"
            or randomization.get(
                "exchangeability_justified_by_experiment_design"
            )
            is not False
            or randomization.get("inference_role")
            != "dependency_sensitive_descriptive_diagnostic_only"
            or randomization.get(
                "conditional_p_value_is_valid_p_value"
            )
            is not False
            or not 0.0
            < float(
                randomization.get(
                    "diagnostic_exceedance_fraction", -1.0
                )
            )
            <= 1.0
            or gate.get("name")
            != "heldout_tail_allocation_outcome_v4"
            or gate.get("requires_ci95_low_at_least_minimum")
            is not True
            or gate.get(
                "requires_positive_extreme_underallocation_reduction"
            )
            is not True
            or gate.get("completion_role")
            != "outcome_only_not_experiment_execution_gate"
            or gate.get(
                "randomization_diagnostic_is_valid_p_value"
            )
            is not False
            or checkpoint_audit.get("status") != "complete"
            or checkpoint_audit.get(
                "paired_feature_signatures_equal"
            )
            is not True
            or checkpoint_audit.get(
                "initialized_state_signatures_equal"
            )
            is not True
            or int(
                checkpoint_audit.get(
                    "actual_checkpoint_count", -1
                )
            )
            != 20
            or checkpoint_audit.get(
                "all_checkpoint_hashes_recorded"
            )
            is not True
            or rare_coverage.get("version")
            != "heldout_instruction_fingerprint_coverage_v1"
            or not rare_status_valid
            or not rare_shape_valid
            or rare_coverage.get("metric_role")
            != "auxiliary_descriptive_metric_not_a_completion_gate"
            or rare_coverage.get("rarity_fit_scope")
            != "training_shards_only"
            or rare_coverage.get("evaluation_scope")
            != "holdout_shards_only"
            or int(rare_coverage.get("training_shards", -1)) != 3_276
            or int(rare_coverage.get("holdout_shards", -1)) != 820
            or rare_coverage_artifact != rare_coverage
            or any(
                not 0.0
                <= float(item.get("source_expected_coverage", -1.0))
                <= 1.0
                or not 0.0
                <= float(item.get("qtail_expected_coverage", -1.0))
                <= 1.0
                or not math.isclose(
                    float(item.get("gain_pp", math.nan)),
                    (
                        float(item.get("qtail_expected_coverage"))
                        - float(item.get("source_expected_coverage"))
                    )
                    * 100.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for item in rare_coverage.get("curve", [])
            )
            or gate.get("outcome") != expected_outcome
            or gate.get("supported") is not expected_supported
            or gate.get("passed") is not expected_supported
            or not math.isclose(
                float(
                    randomization.get(
                        "conditional_p_value", math.nan
                    )
                ),
                float(
                    randomization.get(
                        "diagnostic_exceedance_fraction", math.nan
                    )
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(effect.get("predicted_tail_share_gain_pp", math.nan)),
                tail_gain,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(
                    effect.get(
                        "extreme_underallocation_reduction_pp",
                        math.nan,
                    )
                ),
                extreme_reduction,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append(
                "formal training execution or recomputed outcome contract failed"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"training report is unreadable: {error}")
    return errors


def commit_training_marker(job_root: Path) -> dict[str, Any]:
    semantic_errors = validate_training_semantics(job_root)
    if semantic_errors:
        raise ValueError("; ".join(semantic_errors))
    paths = training_paths(job_root)
    marker = job_root / "manifests" / "DROID_TRAINING_COMPLETE"
    payload = {
        "version": TRAINING_MARKER_VERSION,
        "status": "committed",
        "committed_at": now(),
        "artifacts": [artifact_entry(path) for path in paths],
    }
    atomic_write_json(marker, payload)
    return payload


def completion_requirement_map(
    audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    requirements = audit.get("requirements", [])
    if not isinstance(requirements, list) or len(requirements) != 9:
        return {}, ["completion audit must contain exactly nine requirements"]
    mapped = {
        str(item.get("id")): item
        for item in requirements
        if isinstance(item, dict) and item.get("id")
    }
    if len(mapped) != 9 or set(mapped) != COMPLETION_REQUIREMENT_IDS:
        errors.append(
            "completion audit requirement IDs differ from the fixed contract"
        )
    return mapped, errors


def validate_final_precommit_state(job_root: Path) -> list[str]:
    result_root = job_root / "results" / "qtail_droid_full"
    errors: list[str] = []
    try:
        latest = read_json(result_root / "latest.json")
        audit = read_json(result_root / "completion_audit.json")
        mapped, mapping_errors = completion_requirement_map(audit)
        errors.extend(mapping_errors)
        final_requirement = mapped.get("final_page_qa", {})
        final_evidence = final_requirement.get("evidence", {})
        non_final = [
            item
            for requirement_id, item in mapped.items()
            if requirement_id != "final_page_qa"
        ]
        if (
            latest.get("status") != "in_progress"
            or latest.get("stage") != "final_page_qa"
            or latest.get("completion_audit") != audit
            or audit.get("status") != "in_progress"
            or int(audit.get("passed_requirements", -1)) != 8
            or int(audit.get("total_requirements", -1)) != 9
            or audit.get("experiment_execution_valid") is not True
            or audit.get("formal_results_publishable") is not False
            or audit.get("outcome_is_completion_gate") is not False
            or any(item.get("passed") is not True for item in non_final)
            or final_requirement.get("passed") is not False
            or final_evidence.get("committed") is not False
            or final_evidence.get("qa_state") != "sealing"
        ):
            errors.append(
                "precommit public projection is not an honest 8/9 sealing state"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"precommit public state is unreadable: {error}")
    return errors


def validate_final_public_state(job_root: Path) -> list[str]:
    result_root = job_root / "results" / "qtail_droid_full"
    errors: list[str] = []
    try:
        latest = read_json(result_root / "latest.json")
        audit = read_json(result_root / "completion_audit.json")
        if (
            latest.get("status") != "complete"
            or latest.get("stage") != "complete"
            or latest.get("completion_audit") != audit
        ):
            errors.append("latest public state is not formally complete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"latest public state is unreadable: {error}")
    try:
        audit = read_json(result_root / "completion_audit.json")
        mapped, mapping_errors = completion_requirement_map(audit)
        errors.extend(mapping_errors)
        final_evidence = mapped.get("final_page_qa", {}).get("evidence", {})
        if (
            audit.get("status") != "complete"
            or int(audit.get("passed_requirements", -1)) != 9
            or int(audit.get("total_requirements", -1)) != 9
            or audit.get("experiment_execution_valid") is not True
            or audit.get("formal_results_publishable") is not True
            or audit.get("outcome_is_completion_gate") is not False
            or any(item.get("passed") is not True for item in mapped.values())
            or final_evidence.get("committed") is not True
            or final_evidence.get("preview_active") is not False
            or final_evidence.get("qa_state") != "committed"
        ):
            errors.append("completion audit is not a formal 9/9 closure")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"completion audit is unreadable: {error}")
    try:
        page = (
            Path(__file__).resolve().parents[1]
            / "qtail-droid-full-training.html"
        ).read_text(encoding="utf-8")
        for token in (
            'const STATUS_URL = "results/qtail_droid_full/latest.json"',
            "completion_audit",
            "droid_full_training_report.json",
        ):
            if token not in page:
                errors.append(f"page HTML lacks required data binding: {token}")
    except OSError as error:
        errors.append(f"page HTML is unreadable: {error}")
    return errors


def commit_public_projection_marker(job_root: Path) -> dict[str, Any]:
    result_root = job_root / "results" / "qtail_droid_full"
    progress_lock = (result_root / ".progress_refresh.lock").open("a+")
    fcntl.flock(progress_lock.fileno(), fcntl.LOCK_EX)
    try:
        final = validate_final_marker(job_root, require_public_state=False)
        if not final["valid"]:
            raise ValueError(
                "final marker is invalid: " + "; ".join(final["errors"])
            )
        precommit_errors = validate_final_precommit_state(job_root)
        if precommit_errors:
            raise ValueError(
                "public state is not the honest 8/9 precommit projection: "
                + "; ".join(precommit_errors)
            )

        latest = result_root / "latest.json"
        audit = result_root / "completion_audit.json"
        latest_final = result_root / "latest_final.json"
        audit_final = result_root / "completion_audit_final.json"
        candidate_audit = copy.deepcopy(read_json(audit))
        mapped, mapping_errors = completion_requirement_map(candidate_audit)
        if mapping_errors:
            raise ValueError("; ".join(mapping_errors))
        final_requirement = mapped["final_page_qa"]
        final_requirement["passed"] = True
        final_requirement["evidence"] = {
            **final_requirement.get("evidence", {}),
            "committed": True,
            "preview_active": False,
            "qa_state": "committed",
        }
        candidate_audit["generated_at"] = now()
        candidate_audit["status"] = "complete"
        candidate_audit["passed_requirements"] = 9
        candidate_audit["total_requirements"] = 9
        candidate_audit["formal_results_publishable"] = True

        candidate_latest = copy.deepcopy(read_json(latest))
        candidate_latest["generated_at"] = candidate_audit["generated_at"]
        candidate_latest["status"] = "complete"
        candidate_latest["stage"] = "complete"
        candidate_latest["completion_audit"] = candidate_audit
        marker_state = candidate_latest.setdefault("markers", {})
        marker_state["final_page_qa_complete"] = True
        marker_state["final_page_qa_effective"] = True
        marker_state["final_page_qa_bootstrap_active"] = False
        marker_state["final_page_qa_preview_active"] = False
        marker_state["droid_public_projection_committed"] = True
        marker_state["public_projection_validation"] = {
            "valid": True,
            "errors": [],
        }

        atomic_write_json(audit_final, candidate_audit)
        atomic_write_json(latest_final, candidate_latest)
        final_marker = job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE"
        marker = job_root / "manifests" / "DROID_PUBLIC_PROJECTION_COMMITTED"
        snapshot_for_live = {
            str(latest): latest_final,
            str(audit): audit_final,
        }
        entries = []
        for path in public_projection_paths(job_root):
            source = snapshot_for_live.get(str(path), path)
            entry = artifact_entry(source)
            entry["path"] = str(path)
            entries.append(entry)
        payload = {
            "marker_version": PUBLIC_PROJECTION_MARKER_VERSION,
            "status": "committed",
            "committed_at": now(),
            "final_marker_sha256": sha256(final_marker),
            "artifacts": entries,
        }
        atomic_write_json(marker, payload)
        # Publish the audit first and latest.json last. Readers cannot observe
        # a complete latest projection before every bound dependency exists.
        atomic_copy(audit_final, audit)
        atomic_copy(latest_final, latest)
        committed = validate_public_projection_marker(job_root)
        if not committed["valid"]:
            raise ValueError(
                "public projection did not verify after commit: "
                + "; ".join(committed["errors"])
            )
        return payload
    finally:
        fcntl.flock(progress_lock.fileno(), fcntl.LOCK_UN)
        progress_lock.close()


def validate_public_projection_marker(job_root: Path) -> dict[str, Any]:
    result_root = job_root / "results" / "qtail_droid_full"
    marker = job_root / "manifests" / "DROID_PUBLIC_PROJECTION_COMMITTED"
    errors: list[str] = []
    try:
        payload = read_json(marker)
        if payload.get("marker_version") != PUBLIC_PROJECTION_MARKER_VERSION:
            errors.append("public projection marker version is invalid")
        if payload.get("status") != "committed":
            errors.append("public projection marker status is not committed")
        errors.extend(
            validate_bound_artifacts(
                payload.get("artifacts"),
                public_projection_paths(job_root),
            )
        )
        latest = result_root / "latest.json"
        audit = result_root / "completion_audit.json"
        latest_final = result_root / "latest_final.json"
        audit_final = result_root / "completion_audit_final.json"
        if latest.read_bytes() != latest_final.read_bytes():
            errors.append("live latest.json differs from sealed final snapshot")
        if audit.read_bytes() != audit_final.read_bytes():
            errors.append(
                "live completion_audit.json differs from sealed final snapshot"
            )
        errors.extend(validate_final_public_state(job_root))
    except (OSError, ValueError, TypeError) as error:
        payload = {}
        errors.append(f"public projection marker is unreadable: {error}")
    final = validate_final_marker(job_root, require_public_state=True)
    if not final["valid"]:
        errors.append("bound final marker or public state is invalid")
        errors.extend(final["errors"])
    return {
        "valid": not errors,
        "errors": errors,
        "marker": str(marker),
    }


def validate_complete_final_state(job_root: Path) -> dict[str, Any]:
    final = validate_final_marker(job_root, require_public_state=True)
    projection = validate_public_projection_marker(job_root)
    postcommit = validate_postcommit_page_qa_marker(job_root)
    errors = [
        *(f"final marker: {error}" for error in final["errors"]),
        *(
            f"public projection: {error}"
            for error in projection["errors"]
        ),
        *(
            f"postcommit page QA: {error}"
            for error in postcommit["errors"]
        ),
    ]
    return {
        "valid": not errors,
        "errors": errors,
        "marker": final["marker"],
        "public_projection_marker": projection["marker"],
        "postcommit_page_qa_marker": postcommit["marker"],
    }


def validate_postcommit_page_qa_semantics(job_root: Path) -> list[str]:
    result_root = job_root / "results" / "qtail_droid_full"
    qa_path = result_root / "final_page_postcommit_qa.json"
    errors: list[str] = []
    try:
        qa = read_json(qa_path)
        views = qa.get("final_views")
        if (
            qa.get("version") != "qtail_droid_postcommit_page_qa_v1"
            or qa.get("status") != "complete"
            or qa.get("scope")
            != "final_public_projection_read_only_browser_qa"
            or qa.get("read_only") is not True
            or qa.get("expected_completion") != "9 / 9"
            or qa.get("expected_status") != "全部完成"
            or not isinstance(views, list)
            or len(views) != 2
        ):
            errors.append("postcommit page QA contract is incomplete")
        else:
            observed_viewports = {
                (
                    int(item.get("viewport", {}).get("width", -1)),
                    int(item.get("viewport", {}).get("height", -1)),
                )
                for item in views
                if isinstance(item, dict)
            }
            if observed_viewports != {(1440, 1000), (390, 844)}:
                errors.append(
                    "postcommit page QA viewports are not desktop/mobile"
                )
            for item in views:
                if not isinstance(item, dict):
                    errors.append("postcommit page QA view is invalid")
                    continue
                if (
                    item.get("completion") != "9 / 9"
                    or item.get("status") != "全部完成"
                    or item.get("console_errors") != []
                    or item.get("page_errors") != []
                    or item.get("failed_responses") != []
                ):
                    errors.append(
                        "postcommit browser view did not render clean 9/9"
                    )
        probes = qa.get("url_probes")
        if (
            not isinstance(probes, list)
            or len(probes) < 3
            or any(
                not isinstance(item, dict)
                or item.get("ok") is not True
                or int(item.get("status", -1)) != 200
                for item in probes
            )
        ):
            errors.append("postcommit artifact URL probes are incomplete")
        public_latest = read_json(result_root / "latest_final.json")
        public_audit = read_json(result_root / "completion_audit_final.json")
        if (
            public_latest.get("status") != "complete"
            or public_latest.get("stage") != "complete"
            or public_audit.get("status") != "complete"
            or int(public_audit.get("passed_requirements", -1)) != 9
            or int(public_audit.get("total_requirements", -1)) != 9
        ):
            errors.append("postcommit QA is not bound to a complete 9/9 state")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"postcommit page QA is unreadable: {error}")
    return errors


def commit_postcommit_page_qa_marker(
    job_root: Path,
) -> dict[str, Any]:
    marker = (
        job_root
        / "manifests"
        / "DROID_POSTCOMMIT_PAGE_QA_COMPLETE"
    )
    lock_path = (
        job_root
        / "manifests"
        / ".DROID_POSTCOMMIT_PAGE_QA_COMPLETE.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as marker_lock:
        fcntl.flock(marker_lock.fileno(), fcntl.LOCK_EX)
        try:
            if marker.is_file():
                existing = validate_postcommit_page_qa_marker(job_root)
                if existing["valid"]:
                    return read_json(marker)
                raise ValueError(
                    "existing postcommit page QA marker is invalid: "
                    + "; ".join(existing["errors"])
                )

            projection = validate_public_projection_marker(job_root)
            errors = [
                f"public projection: {error}"
                for error in projection["errors"]
            ]
            errors.extend(validate_postcommit_page_qa_semantics(job_root))
            paths = postcommit_page_qa_paths(job_root)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                errors.append(
                    "postcommit page QA inputs are missing: "
                    + ", ".join(missing)
                )
            if errors:
                raise ValueError("; ".join(errors))
            payload = {
                "marker_version": POSTCOMMIT_PAGE_QA_MARKER_VERSION,
                "status": "committed",
                "committed_at": now(),
                "artifacts": [artifact_entry(path) for path in paths],
            }
            atomic_write_json(marker, payload)
            return payload
        finally:
            fcntl.flock(marker_lock.fileno(), fcntl.LOCK_UN)


def validate_postcommit_page_qa_marker(
    job_root: Path,
) -> dict[str, Any]:
    marker = (
        job_root
        / "manifests"
        / "DROID_POSTCOMMIT_PAGE_QA_COMPLETE"
    )
    errors: list[str] = []
    try:
        payload = read_json(marker)
        if (
            payload.get("marker_version")
            != POSTCOMMIT_PAGE_QA_MARKER_VERSION
        ):
            errors.append("postcommit page QA marker version is invalid")
        if payload.get("status") != "committed":
            errors.append("postcommit page QA marker is not committed")
        errors.extend(
            validate_bound_artifacts(
                payload.get("artifacts"),
                postcommit_page_qa_paths(job_root),
            )
        )
    except (OSError, ValueError, TypeError) as error:
        payload = {}
        errors.append(
            f"postcommit page QA marker is unreadable: {error}"
        )
    errors.extend(validate_postcommit_page_qa_semantics(job_root))
    projection = validate_public_projection_marker(job_root)
    if not projection["valid"]:
        errors.append("bound public projection is invalid")
        errors.extend(projection["errors"])
    return {
        "valid": not errors,
        "errors": errors,
        "marker": str(marker),
    }


def commit_final_marker(job_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    bootstrap = validate_final_bootstrap(job_root)
    if not bootstrap["valid"]:
        errors.append(
            "lease-bound final bootstrap is invalid: "
            + "; ".join(bootstrap["errors"])
        )
    training = validate_training_marker(job_root)
    if not training["valid"]:
        errors.append("training marker is invalid")
    result_root = job_root / "results" / "qtail_droid_full"
    try:
        qa = read_json(result_root / "final_page_qa.json")
        if qa.get("status") != "complete":
            errors.append("final page QA artifact is not complete")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"final page QA artifact is unreadable: {error}")
    try:
        timeline = read_json(
            result_root / "pipeline_timeline_final_verification.json"
        )
        if (
            timeline.get("status") != "passed"
            or timeline.get("scope") != "final_precommit"
            or int(
                timeline.get("final_completion", {}).get(
                    "passed_requirements", -1
                )
            )
            != 8
        ):
            errors.append(
                "final precommit timeline verification is incomplete"
            )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"final timeline verification is unreadable: {error}")
    errors.extend(
        validate_artifact_manifest_entries(
            result_root / "droid_artifact_manifest.json"
        )
    )
    errors.extend(validate_process_log_manifest(job_root))
    errors.extend(validate_final_precommit_state(job_root))
    paths = final_paths(job_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        errors.append("final marker inputs are missing: " + ", ".join(missing))
    if errors:
        raise ValueError("; ".join(errors))
    marker = job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE"
    payload = {
        "marker_version": FINAL_MARKER_VERSION,
        "status": "committed",
        "committed_at": now(),
        "artifacts": [artifact_entry(path) for path in paths],
    }
    atomic_write_json(marker, payload)
    return payload


def validate_training_marker(job_root: Path) -> dict[str, Any]:
    marker = job_root / "manifests" / "DROID_TRAINING_COMPLETE"
    errors: list[str] = []
    try:
        payload = read_json(marker)
        if payload.get("version") != TRAINING_MARKER_VERSION:
            errors.append("training marker version is invalid")
        if payload.get("status") != "committed":
            errors.append("training marker status is not committed")
        errors.extend(
            validate_bound_artifacts(payload.get("artifacts"), training_paths(job_root))
        )
    except (OSError, ValueError, TypeError) as error:
        payload = {}
        errors.append(f"training marker is unreadable: {error}")
    errors.extend(validate_training_semantics(job_root))
    return {"valid": not errors, "errors": errors, "marker": str(marker)}


def validate_final_marker(
    job_root: Path,
    *,
    require_public_state: bool = True,
) -> dict[str, Any]:
    marker = job_root / "manifests" / "FINAL_PAGE_QA_COMPLETE"
    errors: list[str] = []
    try:
        payload = read_json(marker)
        if payload.get("marker_version") != FINAL_MARKER_VERSION:
            errors.append("final marker version is invalid")
        if payload.get("status") != "committed":
            errors.append("final marker status is not committed")
        errors.extend(
            validate_bound_artifacts(payload.get("artifacts"), final_paths(job_root))
        )
        errors.extend(
            validate_artifact_manifest_entries(
                job_root
                / "results"
                / "qtail_droid_full"
                / "droid_artifact_manifest.json"
            )
        )
        qa = read_json(
            job_root / "results" / "qtail_droid_full" / "final_page_qa.json"
        )
        if qa.get("status") != "complete":
            errors.append("final page QA artifact is not complete")
        timeline_verification = read_json(
            job_root
            / "results"
            / "qtail_droid_full"
            / "pipeline_timeline_final_verification.json"
        )
        if (
            timeline_verification.get("status") != "passed"
            or timeline_verification.get("scope")
            != "final_precommit"
            or int(timeline_verification.get("final_completed_objects", -1))
            != 4_102
            or int(
                timeline_verification.get("final_completion", {}).get(
                    "passed_requirements", -1
                )
            )
            != 8
        ):
            errors.append(
                "final precommit pipeline timeline verification is incomplete"
            )
        errors.extend(
            validate_data_continuity_summary(
                timeline_verification.get("data_continuity")
            )
        )
        errors.extend(validate_process_log_manifest(job_root))
        if require_public_state:
            errors.extend(validate_final_public_state(job_root))
    except (OSError, ValueError, TypeError) as error:
        payload = {}
        errors.append(f"final marker is unreadable: {error}")
    training = validate_training_marker(job_root)
    if not training["valid"]:
        errors.append("bound training marker is invalid")
    return {"valid": not errors, "errors": errors, "marker": str(marker)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-root",
        type=Path,
        default=Path("/Volumes/ORICO/qtail_full_training"),
    )
    parser.add_argument(
        "--stage",
        choices=("training", "final"),
        required=True,
    )
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--commit-bootstrap", action="store_true")
    parser.add_argument("--commit-public-projection", action="store_true")
    parser.add_argument("--commit-postcommit-qa", action="store_true")
    parser.add_argument("--validate-projection", action="store_true")
    parser.add_argument("--print-paths", action="store_true")
    args = parser.parse_args()

    commit_modes = sum(
        (
            args.commit,
            args.commit_bootstrap,
            args.commit_public_projection,
            args.commit_postcommit_qa,
            args.validate_projection,
        )
    )
    if commit_modes > 1:
        parser.error(
            "commit and validation modes are mutually exclusive"
        )

    if args.print_paths:
        paths = (
            training_paths(args.job_root)
            if args.stage == "training"
            else final_paths(args.job_root)
        )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "paths": [str(path) for path in paths],
                },
                ensure_ascii=False,
            )
        )
        return

    if args.commit_bootstrap:
        if args.stage != "final":
            parser.error("--commit-bootstrap requires --stage final")
        commit_final_bootstrap(args.job_root)
        result = validate_final_bootstrap(args.job_root)
        print(json.dumps(result, ensure_ascii=False))
        if not result["valid"]:
            raise SystemExit(1)
        return

    if args.commit_public_projection:
        if args.stage != "final":
            parser.error("--commit-public-projection requires --stage final")
        commit_public_projection_marker(args.job_root)
        result = validate_public_projection_marker(args.job_root)
        print(json.dumps(result, ensure_ascii=False))
        if not result["valid"]:
            raise SystemExit(1)
        return

    if args.commit_postcommit_qa:
        if args.stage != "final":
            parser.error(
                "--commit-postcommit-qa requires --stage final"
            )
        commit_postcommit_page_qa_marker(args.job_root)
        result = validate_postcommit_page_qa_marker(args.job_root)
        print(json.dumps(result, ensure_ascii=False))
        if not result["valid"]:
            raise SystemExit(1)
        return

    if args.validate_projection:
        if args.stage != "final":
            parser.error("--validate-projection requires --stage final")
        result = validate_public_projection_marker(args.job_root)
        print(json.dumps(result, ensure_ascii=False))
        if not result["valid"]:
            raise SystemExit(1)
        return

    if args.commit:
        if args.stage == "training":
            commit_training_marker(args.job_root)
        else:
            commit_final_marker(args.job_root)
    if args.stage == "training":
        result = validate_training_marker(args.job_root)
    elif args.commit:
        result = validate_final_marker(
            args.job_root,
            require_public_state=False,
        )
    else:
        result = validate_complete_final_state(args.job_root)
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
