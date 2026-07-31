#!/usr/bin/env python3
"""Positive/negative controls for DROID marker hardening helpers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import qtail_verify_droid_stage_markers as marker_module
from qtail_verify_droid_stage_markers import (
    CHECKSUM_MARKER_VERSION,
    artifact_entry,
    atomic_write_json,
    final_paths,
    validate_artifact_manifest_entries,
    validate_bound_artifacts,
    validate_checksum_marker,
    validate_process_log_manifest,
    validate_transition_gate,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gate_payload() -> dict:
    checks = {
        "guard_status_passed": True,
        "guard_heartbeat_fresh": True,
        "uniclash_core_running": True,
        "uniclash_tun_disabled": True,
        "droid_bypass_policy_enabled": True,
        "curl_and_gsutil_guarded": True,
        "expected_interface_bound": True,
        "system_proxy_bypass_passed": True,
        "cumulative_history_clean": True,
        "live_transfers_clean_and_direct": True,
    }
    return {
        "generated_at": now(),
        "status": "passed",
        "guard_age_seconds": 0.1,
        "max_guard_age_seconds": 10.0,
        "checks": checks,
        "global_violations": [],
        "transfer_violations": [],
    }


def completion_projection(committed: bool) -> tuple[dict, dict]:
    requirements = []
    for requirement_id in sorted(marker_module.COMPLETION_REQUIREMENT_IDS):
        is_final = requirement_id == "final_page_qa"
        passed = committed if is_final else True
        requirement = {
            "id": requirement_id,
            "label": requirement_id,
            "passed": passed,
            "evidence": {},
        }
        if is_final:
            requirement["evidence"] = {
                "committed": committed,
                "preview_active": False if committed else True,
                "qa_state": "committed" if committed else "sealing",
            }
        requirements.append(requirement)
    audit = {
        "status": "complete" if committed else "in_progress",
        "passed_requirements": 9 if committed else 8,
        "total_requirements": 9,
        "experiment_execution_valid": True,
        "formal_results_publishable": True,
        "outcome_is_completion_gate": False,
        "requirements": requirements,
    }
    latest = {
        "status": "complete" if committed else "in_progress",
        "stage": "complete" if committed else "final_page_qa",
        "completion_audit": audit,
    }
    return latest, audit


def run_public_projection_integration_controls() -> list[dict]:
    controls: list[dict] = []
    with tempfile.TemporaryDirectory(
        prefix="qtail-public-projection-selftest-"
    ) as raw:
        job_root = Path(raw)
        result_root = job_root / "results" / "qtail_droid_full"
        marker_root = job_root / "manifests"
        result_root.mkdir(parents=True)
        marker_root.mkdir()

        for path in final_paths(job_root):
            if path.is_relative_to(job_root):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"selftest artifact: {path.name}\n",
                    encoding="utf-8",
                )
        atomic_write_json(
            result_root / "final_page_qa.json",
            {"status": "complete"},
        )
        atomic_write_json(
            result_root / "pipeline_timeline_final_verification.json",
            {
                "status": "passed",
                "scope": "final_precommit",
                "final_completed_objects": 4_102,
                "final_completion": {"passed_requirements": 8},
                "data_continuity": {},
            },
        )
        latest, audit = completion_projection(False)
        atomic_write_json(result_root / "latest.json", latest)
        atomic_write_json(result_root / "completion_audit.json", audit)

        original_training_validator = marker_module.validate_training_marker
        original_manifest_validator = (
            marker_module.validate_artifact_manifest_entries
        )
        original_process_validator = marker_module.validate_process_log_manifest
        original_continuity_validator = (
            marker_module.validate_data_continuity_summary
        )
        original_preview_validator = marker_module.active_final_preview
        preview_state = {
            "active": False,
            "owner_pid": os.getpid(),
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        marker_module.validate_training_marker = lambda _root: {
            "valid": True,
            "errors": [],
        }
        marker_module.validate_artifact_manifest_entries = lambda _path: []
        marker_module.validate_process_log_manifest = lambda _root: []
        marker_module.validate_data_continuity_summary = lambda _payload: []
        marker_module.active_final_preview = lambda _root: (
            bool(preview_state["active"]),
            {
                "owner_pid": preview_state["owner_pid"],
                "expires_at": preview_state["expires_at"],
            },
        )
        try:
            missing_bootstrap_rejected = False
            try:
                marker_module.commit_final_marker(job_root)
            except ValueError:
                missing_bootstrap_rejected = True
            controls.append(
                {
                    "name": "final_commit_without_valid_bootstrap_rejected",
                    "passed": missing_bootstrap_rejected,
                }
            )

            preview_state["active"] = True
            marker_module.commit_final_bootstrap(job_root)
            preview_state["active"] = False
            expired_lease_rejected = False
            try:
                marker_module.commit_final_marker(job_root)
            except ValueError:
                expired_lease_rejected = True
            controls.append(
                {
                    "name": "final_commit_with_expired_lease_rejected",
                    "passed": expired_lease_rejected,
                }
            )

            preview_state["active"] = True
            marker_module.commit_final_bootstrap(job_root)
            bootstrap_path = marker_root / "FINAL_PAGE_QA_COMPLETE"
            bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            bootstrap["owner_pid"] = int(bootstrap["owner_pid"]) + 1
            atomic_write_json(bootstrap_path, bootstrap)
            owner_mismatch_rejected = False
            try:
                marker_module.commit_final_marker(job_root)
            except ValueError:
                owner_mismatch_rejected = True
            controls.append(
                {
                    "name": "final_commit_with_bootstrap_owner_mismatch_rejected",
                    "passed": owner_mismatch_rejected,
                }
            )

            marker_module.commit_final_bootstrap(job_root)
            marker_module.commit_final_marker(job_root)
            precommit = marker_module.validate_final_marker(
                job_root,
                require_public_state=False,
            )
            controls.append(
                {
                    "name": "real_eight_of_nine_final_marker_commit",
                    "passed": precommit["valid"],
                }
            )

            marker_module.commit_public_projection_marker(job_root)
            controls.append(
                {
                    "name": (
                        "public_projection_without_postcommit_qa_is_rejected"
                    ),
                    "passed": not marker_module.validate_complete_final_state(
                        job_root
                    )["valid"],
                }
            )
            atomic_write_json(
                result_root / "final_page_postcommit_qa.json",
                {
                    "version": "qtail_droid_postcommit_page_qa_v1",
                    "status": "complete",
                    "scope": (
                        "final_public_projection_read_only_browser_qa"
                    ),
                    "read_only": True,
                    "expected_completion": "9 / 9",
                    "expected_status": "全部完成",
                    "final_views": [
                        {
                            "viewport": {"width": 1440, "height": 1000},
                            "completion": "9 / 9",
                            "status": "全部完成",
                            "console_errors": [],
                            "page_errors": [],
                            "failed_responses": [],
                        },
                        {
                            "viewport": {"width": 390, "height": 844},
                            "completion": "9 / 9",
                            "status": "全部完成",
                            "console_errors": [],
                            "page_errors": [],
                            "failed_responses": [],
                        },
                    ],
                    "url_probes": [
                        {"status": 200, "ok": True},
                        {"status": 200, "ok": True},
                        {"status": 200, "ok": True},
                    ],
                },
            )
            (
                result_root / "final_page_postcommit_desktop.png"
            ).write_bytes(b"desktop")
            (
                result_root / "final_page_postcommit_mobile.png"
            ).write_bytes(b"mobile")
            postcommit_marker = (
                marker_root / "DROID_POSTCOMMIT_PAGE_QA_COMPLETE"
            )
            marker_module.commit_postcommit_page_qa_marker(job_root)
            original_postcommit_marker = postcommit_marker.read_bytes()
            controls.append(
                {
                    "name": (
                        "atomic_eight_to_nine_with_postcommit_browser_marker"
                    ),
                    "passed": marker_module.validate_complete_final_state(
                        job_root
                    )["valid"],
                }
            )
            postcommit_desktop = (
                result_root / "final_page_postcommit_desktop.png"
            )
            postcommit_desktop.write_bytes(b"tampered")
            controls.append(
                {
                    "name": "postcommit_browser_screenshot_tamper_rejected",
                    "passed": not marker_module.validate_complete_final_state(
                        job_root
                    )["valid"],
                }
            )
            postcommit_desktop.write_bytes(b"desktop")
            marker_module.commit_postcommit_page_qa_marker(job_root)
            controls.append(
                {
                    "name": "public_projection_live_and_snapshot_match",
                    "passed": (
                        postcommit_marker.read_bytes()
                        == original_postcommit_marker
                        and
                        (result_root / "latest.json").read_bytes()
                        == (result_root / "latest_final.json").read_bytes()
                        and (result_root / "completion_audit.json").read_bytes()
                        == (
                            result_root / "completion_audit_final.json"
                        ).read_bytes()
                    ),
                }
            )

            (result_root / "latest_final.json").write_text(
                '{"status":"tampered"}\n',
                encoding="utf-8",
            )
            controls.append(
                {
                    "name": "sealed_projection_snapshot_tamper_rejected",
                    "passed": not marker_module.validate_complete_final_state(
                        job_root
                    )["valid"],
                }
            )

            marker_module.atomic_copy(
                result_root / "latest.json",
                result_root / "latest_final.json",
            )
            marker_module.atomic_copy(
                result_root / "completion_audit.json",
                result_root / "completion_audit_final.json",
            )
            marker_module.commit_postcommit_page_qa_marker(job_root)
            final_marker = marker_root / "FINAL_PAGE_QA_COMPLETE"
            final_payload = json.loads(final_marker.read_text(encoding="utf-8"))
            final_payload["status"] = "tampered"
            atomic_write_json(final_marker, final_payload)
            controls.append(
                {
                    "name": "sealed_final_marker_tamper_rejected",
                    "passed": not marker_module.validate_complete_final_state(
                        job_root
                    )["valid"],
                }
            )
        finally:
            marker_module.validate_training_marker = original_training_validator
            marker_module.validate_artifact_manifest_entries = (
                original_manifest_validator
            )
            marker_module.validate_process_log_manifest = (
                original_process_validator
            )
            marker_module.validate_data_continuity_summary = (
                original_continuity_validator
            )
            marker_module.active_final_preview = original_preview_validator
    return controls


def run_controls() -> list[dict]:
    controls: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="qtail-marker-selftest-") as raw:
        job_root = Path(raw)
        result_root = job_root / "results" / "qtail_droid_full"
        marker_root = job_root / "manifests"
        result_root.mkdir(parents=True)
        marker_root.mkdir()

        verification = result_root / "download_verification.json"
        download_marker = marker_root / "DROID_DOWNLOAD_COMPLETE"
        verification.write_text('{"status":"passed"}\n', encoding="utf-8")
        download_marker.write_text('{"status":"committed"}\n', encoding="utf-8")
        checksum_marker = marker_root / "DROID_CHECKSUM_VERIFIED"
        payload = {
            "version": CHECKSUM_MARKER_VERSION,
            "status": "verified",
            "download_verification": str(verification),
            "download_verification_bytes": verification.stat().st_size,
            "download_verification_sha256": artifact_entry(verification)[
                "sha256"
            ],
            "download_completion_marker": str(download_marker),
            "download_completion_marker_bytes": download_marker.stat().st_size,
            "download_completion_marker_sha256": artifact_entry(
                download_marker
            )["sha256"],
        }
        atomic_write_json(checksum_marker, payload)
        controls.append(
            {
                "name": "checksum_binding_positive",
                "passed": validate_checksum_marker(job_root) == [],
            }
        )
        verification.write_text('{"status":"tampered"}\n', encoding="utf-8")
        controls.append(
            {
                "name": "checksum_binding_tamper_rejected",
                "passed": bool(validate_checksum_marker(job_root)),
            }
        )

        gate = result_root / "gate.json"
        atomic_write_json(gate, gate_payload())
        controls.append(
            {
                "name": "transition_gate_positive",
                "passed": validate_transition_gate(gate, "selftest") == [],
            }
        )
        failed_gate = gate_payload()
        failed_gate["checks"]["uniclash_tun_disabled"] = False
        atomic_write_json(gate, failed_gate)
        controls.append(
            {
                "name": "transition_gate_tun_rejected",
                "passed": bool(validate_transition_gate(gate, "selftest")),
            }
        )

        artifact = result_root / "artifact.bin"
        artifact.write_bytes(b"formal-evidence")
        manifest = result_root / "droid_artifact_manifest.json"
        atomic_write_json(
            manifest,
            {
                "status": "complete",
                "formal_droid_contract": {
                    "required_artifact_count": 1,
                    "all_required_present": True,
                },
                "artifacts": [artifact_entry(artifact)],
            },
        )
        controls.append(
            {
                "name": "recursive_manifest_positive",
                "passed": validate_artifact_manifest_entries(manifest) == [],
            }
        )
        artifact.write_bytes(b"changed")
        controls.append(
            {
                "name": "recursive_manifest_tamper_rejected",
                "passed": bool(validate_artifact_manifest_entries(manifest)),
            }
        )

        public_state = result_root / "latest.json"
        completion = result_root / "completion_audit.json"
        public_state.write_text('{"status":"complete"}\n', encoding="utf-8")
        precommit_latest, precommit_audit = completion_projection(False)
        atomic_write_json(public_state, precommit_latest)
        atomic_write_json(completion, precommit_audit)
        public_entries = [
            artifact_entry(public_state),
            artifact_entry(completion),
        ]
        controls.append(
            {
                "name": "public_state_binding_positive",
                "passed": validate_bound_artifacts(
                    public_entries, [public_state, completion]
                )
                == [],
            }
        )
        completion.write_text('{"status":"changed"}\n', encoding="utf-8")
        controls.append(
            {
                "name": "stale_public_state_binding_is_rejected",
                "passed": bool(
                    validate_bound_artifacts(
                        public_entries, [public_state, completion]
                    )
                ),
            }
        )
        precommit_latest, precommit_audit = completion_projection(False)
        atomic_write_json(public_state, precommit_latest)
        atomic_write_json(completion, precommit_audit)
        (marker_root / "FINAL_PAGE_QA_COMPLETE").write_text(
            '{"status":"committed"}\n',
            encoding="utf-8",
        )
        original_final_validator = marker_module.validate_final_marker
        original_public_validator = marker_module.validate_final_public_state
        marker_module.validate_final_marker = lambda _root, **_kwargs: {
            "valid": True,
            "errors": [],
            "marker": str(marker_root / "FINAL_PAGE_QA_COMPLETE"),
        }
        marker_module.validate_final_public_state = lambda _root: []
        try:
            marker_module.commit_public_projection_marker(job_root)
            controls.append(
                {
                    "name": "public_projection_snapshot_binding_positive",
                    "passed": marker_module.validate_public_projection_marker(
                        job_root
                    )["valid"],
                }
            )
            public_state.write_text(
                '{"status":"post-seal-tamper"}\n',
                encoding="utf-8",
            )
            controls.append(
                {
                    "name": "public_projection_live_tamper_rejected",
                    "passed": not marker_module.validate_public_projection_marker(
                        job_root
                    )["valid"],
                }
            )
        finally:
            marker_module.validate_final_marker = original_final_validator
            marker_module.validate_final_public_state = original_public_validator
        final_names = {path.name for path in final_paths(job_root)}
        controls.append(
            {
                "name": "final_path_contract_binds_only_immutable_closure",
                "passed": {
                    "qtail-droid-full-training.html",
                    "droid_full_training_report.json",
                    "droid_training_curve.csv",
                    "download_progress_samples_final.json",
                    "pipeline_timeline_final.json",
                    "pipeline_timeline_final_verification.json",
                    "droid_process_log_manifest.json",
                }.issubset(final_names)
                and {
                    "latest.json",
                    "completion_audit.json",
                    "download_progress_samples.json",
                    "pipeline_timeline.json",
                    "pipeline_timeline_current_verification.json",
                }.isdisjoint(final_names),
            }
        )

        bootstrap_inputs = (
            marker_root / "DROID_TRAINING_COMPLETE",
            result_root / "droid_full_training_report.json",
            result_root / "droid_rare_instruction_fingerprint_coverage.json",
            result_root / "droid_model_training_status.json",
            result_root / "droid_training_curve.csv",
        )
        for path in bootstrap_inputs:
            path.write_text('{"status":"complete"}\n', encoding="utf-8")
        original_training_validator = marker_module.validate_training_marker
        original_preview_validator = marker_module.active_final_preview
        marker_module.validate_training_marker = lambda _root: {
            "valid": True,
            "errors": [],
        }
        marker_module.active_final_preview = lambda _root: (
            True,
            {
                "owner_pid": os.getpid(),
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
        )
        try:
            marker_module.commit_final_bootstrap(job_root)
            controls.append(
                {
                    "name": "lease_bound_bootstrap_positive",
                    "passed": marker_module.validate_final_bootstrap(job_root)[
                        "valid"
                    ],
                }
            )
            (result_root / "droid_full_training_report.json").write_text(
                '{"status":"tampered"}\n',
                encoding="utf-8",
            )
            controls.append(
                {
                    "name": "bootstrap_artifact_tamper_rejected",
                    "passed": not marker_module.validate_final_bootstrap(job_root)[
                        "valid"
                    ],
                }
            )
        finally:
            marker_module.validate_training_marker = original_training_validator
            marker_module.active_final_preview = original_preview_validator

        evidence_paths = (
            marker_root / "DROID_CHECKSUM_VERIFIED",
            result_root / "droid_incremental_closure_audit.json",
            marker_root / "DROID_TRAINING_COMPLETE",
            result_root / "final_page_qa.json",
        )
        for path in evidence_paths:
            path.write_text('{"status":"complete"}\n', encoding="utf-8")
        log_names = (
            "droid_full_pipeline.log",
            "droid_feature_prewarm.log",
            "pipeline_watchdog.log",
            "progress_loop.log",
            "progress_refresh.log",
            "pipeline_generation_handoff.log",
            "manual_endpoint_generation_handoff.log",
            "qtail-web-services.log",
        )
        snapshot_root = result_root / "process_logs_final"
        snapshot_root.mkdir()
        qa_sha256 = artifact_entry(
            result_root / "final_page_qa.json"
        )["sha256"]
        tokens = "\n".join(
            (
                "QTAIL_TERMINAL checksum_complete",
                "QTAIL_TERMINAL record_closure_complete records=187891",
                "QTAIL_TERMINAL training_complete",
                "QTAIL_TERMINAL qa_sealing_complete "
                f"qa_sha256={qa_sha256}",
            )
        )
        log_entries = []
        for name in log_names:
            path = snapshot_root / name
            path.write_text(
                tokens + "\n" if name == "droid_full_pipeline.log"
                else f"terminal evidence log: {name}\n",
                encoding="utf-8",
            )
            log_entries.append(
                {
                    **artifact_entry(path),
                    "role": f"selftest role for {name}",
                    "line_count": len(path.read_text().splitlines()),
                }
            )
        process_manifest = result_root / "droid_process_log_manifest.json"
        atomic_write_json(
            process_manifest,
            {
                "status": "complete",
                "contract": {
                    "snapshot_is_immutable": True,
                    "live_logs_continue_after_snapshot": True,
                    "required_log_count": 8,
                    "captured_required_log_count": 8,
                },
                "missing_required": [],
                "logs": log_entries,
            },
        )
        atomic_write_json(
            manifest,
            {
                "status": "complete",
                "formal_droid_contract": {
                    "required_artifact_count": 9,
                    "all_required_present": True,
                },
                "artifacts": [
                    artifact_entry(process_manifest),
                    *(artifact_entry(snapshot_root / name) for name in log_names),
                ],
            },
        )
        controls.append(
            {
                "name": "terminal_log_gate_positive",
                "passed": validate_process_log_manifest(job_root) == [],
            }
        )
        pipeline_snapshot = snapshot_root / "droid_full_pipeline.log"
        os.utime(pipeline_snapshot, (1, 1))
        controls.append(
            {
                "name": "terminal_log_predating_artifact_is_rejected",
                "passed": bool(validate_process_log_manifest(job_root)),
            }
        )
        (snapshot_root / "progress_refresh.log").write_text(
            "", encoding="utf-8"
        )
        controls.append(
            {
                "name": "empty_log_is_rejected",
                "passed": bool(validate_process_log_manifest(job_root)),
            }
        )
    controls.extend(run_public_projection_integration_controls())
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    controls = run_controls()
    passed = all(control["passed"] for control in controls)
    payload = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "controls_passed": sum(control["passed"] for control in controls),
        "controls_total": len(controls),
        "controls": controls,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
