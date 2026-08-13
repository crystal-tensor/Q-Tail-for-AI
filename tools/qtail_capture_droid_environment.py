#!/usr/bin/env python3
"""Capture a secret-free, reproducible environment manifest for DROID training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qtail_assert_uniclash_transport_gate import validate_guard


EXPECTED_BACKEND_COMMIT = "9a29c832b4c81bf38401111f5e4cdddaca217581"
EXPECTED_BACKEND_ORIGIN = (
    "https://github.com/droid-dataset/droid_policy_learning"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
    }


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_snapshot_manifest(path: Path | None) -> tuple[dict[str, str], list[str]]:
    if path is None or not path.is_file():
        return {}, ["snapshot_manifest_missing"]
    entries: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            digest, relative = line.split("  ./", 1)
        except ValueError:
            errors.append(f"line_{line_number}:invalid_format")
            continue
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            errors.append(f"line_{line_number}:invalid_entry")
            continue
        if relative in entries:
            errors.append(f"line_{line_number}:duplicate_path:{relative}")
            continue
        entries[relative] = digest
    if not entries:
        errors.append("snapshot_manifest_empty")
    return entries, errors


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def normalize_git_origin(value: str) -> str:
    normalized = value.strip().removesuffix("/").removesuffix(".git")
    if normalized.startswith("git@github.com:"):
        normalized = (
            "https://github.com/" + normalized.removeprefix("git@github.com:")
        )
    return normalized


def package_inventory() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if name:
            packages[name.lower()] = str(distribution.version)
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-source", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--download-verification", type=Path)
    parser.add_argument("--transport-status", type=Path)
    parser.add_argument("--uniclash-guard-status", type=Path)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--orchestration-snapshot-manifest", type=Path)
    parser.add_argument("--require-final-inputs", action="store_true")
    args = parser.parse_args()

    code_paths = [
        args.repo_root / "tools" / "qtail_capture_droid_environment.py",
        args.repo_root
        / "tools"
        / "qtail_droid_environment_contract_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_prewarm_status_contract_selftest.py",
        args.repo_root / "tools" / "qtail_parallel_gcs_download.py",
        args.repo_root
        / "tools"
        / "qtail_verify_droid_download_marker.py",
        args.repo_root
        / "tools"
        / "qtail_droid_download_marker_selftest.py",
        args.repo_root / "tools" / "qtail_verify_droid_mirror.py",
        args.repo_root
        / "tools"
        / "qtail_droid_mirror_verifier_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_downloader_single_writer_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_runtime_process_contract_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_assert_uniclash_transport_gate.py",
        args.repo_root
        / "tools"
        / "qtail_uniclash_transport_gate_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_capture_droid_partial_marker_rejection.py",
        args.repo_root
        / "tools"
        / "qtail_record_droid_transport_tuning.py",
        args.repo_root
        / "tools"
        / "qtail_adjudicate_uniclash_transport.py",
        args.repo_root
        / "tools"
        / "qtail_audit_droid_release_metadata.py",
        args.repo_root / "tools" / "qtail_uniclash_transport_guard.py",
        args.repo_root / "tools" / "qtail_train_openx_demo.py",
        args.repo_root / "tools" / "qtail_train_droid_full.py",
        args.repo_root
        / "tools"
        / "qtail_droid_training_gate_order_selftest.py",
        args.repo_root / "tools" / "qtail_droid_full_progress.py",
        args.repo_root
        / "tools"
        / "qtail_droid_timeline_monotonic_selftest.py",
        args.repo_root / "tools" / "qtail_verify_droid_timeline.py",
        args.repo_root / "tools" / "qtail_droid_protocol_selftest.py",
        args.repo_root / "tools" / "qtail_verify_droid_feature_cache.py",
        args.repo_root
        / "tools"
        / "qtail_audit_droid_incremental_closure.py",
        args.repo_root
        / "tools"
        / "qtail_droid_incremental_closure_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_merge_droid_artifact_manifest.py",
        args.repo_root
        / "tools"
        / "qtail_artifact_manifest_merge_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_progress_preview_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_stage_marker_hardening_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_pipeline_shell_contract_selftest.py",
        args.repo_root
        / "tools"
        / "qtail_seal_droid_release_milestones.py",
        args.repo_root / "tools" / "qtail_verify_droid_page.mjs",
        args.repo_root / "tools" / "qtail_verify_droid_stage_markers.py",
        args.repo_root
        / "tools"
        / "qtail_publish_orchestration_snapshot.py",
        args.repo_root / "tools" / "qtail_summarize_droid_forecast.py",
        args.repo_root / "scripts" / "qtail_droid_feature_prewarm_loop.sh",
        args.repo_root / "scripts" / "qtail_droid_pipeline_watchdog.sh",
        args.repo_root / "scripts" / "qtail_droid_progress_loop.sh",
        args.repo_root / "scripts" / "qtail_droid_terminal_launcher.command",
        args.repo_root / "scripts" / "qtail_orico_full_pipeline.sh",
        args.repo_root
        / "scripts"
        / "qtail_reload_pipeline_after_download.sh",
        args.repo_root / "launchd" / "com.qtail.droid-full-pipeline.plist",
        args.repo_root
        / "launchd"
        / "com.qtail.uniclash-transport-guard.plist",
        args.repo_root / "qtail-droid-full-training.html",
        args.repo_root / "docs" / "experiments" / "qtail_droid_full_protocol.md",
        args.repo_root
        / "docs"
        / "experiments"
        / "qtail_droid_red_team_audit.md",
    ]
    input_paths = [
        args.pt_source,
        args.object_manifest,
        args.checksum_manifest,
    ]
    if args.download_verification:
        input_paths.append(args.download_verification)
    if args.transport_status:
        input_paths.append(args.transport_status)
    if args.uniclash_guard_status:
        input_paths.append(args.uniclash_guard_status)
    if args.orchestration_snapshot_manifest:
        input_paths.append(args.orchestration_snapshot_manifest)
    code = [file_entry(path) for path in code_paths]
    inputs = [file_entry(path) for path in input_paths]
    missing_code = [item["path"] for item in code if not item["exists"]]
    missing_inputs = [item["path"] for item in inputs if not item["exists"]]
    if args.require_final_inputs and not args.download_verification:
        missing_inputs.append("download_verification_argument")
    if args.require_final_inputs and not args.uniclash_guard_status:
        missing_inputs.append("uniclash_guard_status_argument")
    if args.require_final_inputs and not args.orchestration_snapshot_manifest:
        missing_inputs.append("orchestration_snapshot_manifest_argument")

    snapshot_entries, snapshot_manifest_errors = read_snapshot_manifest(
        args.orchestration_snapshot_manifest
    )
    repo_root_resolved = args.repo_root.resolve()
    snapshot_code_mismatches: list[dict[str, Any]] = []
    for item in code:
        try:
            relative = str(
                Path(str(item["path"])).resolve().relative_to(
                    repo_root_resolved
                )
            )
        except ValueError:
            snapshot_code_mismatches.append(
                {
                    "path": item["path"],
                    "reason": "outside_repo_root",
                }
            )
            continue
        snapshot_sha256 = snapshot_entries.get(relative)
        if snapshot_sha256 != item.get("sha256"):
            snapshot_code_mismatches.append(
                {
                    "path": item["path"],
                    "relative_path": relative,
                    "workspace_sha256": item.get("sha256"),
                    "snapshot_sha256": snapshot_sha256,
                    "reason": (
                        "missing_from_snapshot"
                        if snapshot_sha256 is None
                        else "sha256_mismatch"
                    ),
                }
            )
    snapshot_code_parity_passed = bool(
        args.orchestration_snapshot_manifest
        and not snapshot_manifest_errors
        and code
        and not snapshot_code_mismatches
    )

    backend_commit = command_output(
        ["git", "-C", str(args.backend_root), "rev-parse", "HEAD"]
    )
    backend_fsck = command_output(
        ["git", "-C", str(args.backend_root), "fsck", "--no-progress"]
    )
    backend_origin = command_output(
        ["git", "-C", str(args.backend_root), "remote", "get-url", "origin"]
    )
    backend_status = command_output(
        [
            "git",
            "-C",
            str(args.backend_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    )
    backend_status_entries = [
        line for line in backend_status["stdout"].splitlines() if line.strip()
    ]
    mount = command_output(["/sbin/mount"])
    mount_line = next(
        (
            line
            for line in mount["stdout"].splitlines()
            if " on /Volumes/ORICO (" in line
        ),
        None,
    )
    disk_usage = command_output(["/bin/df", "-k", str(args.job_root)])
    sw_vers = command_output(["/usr/bin/sw_vers"])
    memory = command_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    cpu_count = command_output(["/usr/sbin/sysctl", "-n", "hw.ncpu"])
    packages = package_inventory()
    object_payload = read_json(args.object_manifest)
    checksum_payload = read_json(args.checksum_manifest)
    verification_payload = read_json(args.download_verification)
    transport_payload = read_json(args.transport_status)
    uniclash_guard_payload = read_json(args.uniclash_guard_status)
    object_rows = object_payload.get("objects", [])
    checksum_rows = checksum_payload.get("objects", [])
    object_by_path = {
        str(item.get("relative_path")): int(item.get("bytes", -1))
        for item in object_rows
        if item.get("relative_path")
    }
    checksum_by_path = {
        str(item.get("relative_path")): item
        for item in checksum_rows
        if item.get("relative_path")
    }
    expected_object_count = len(object_by_path)
    expected_total_bytes = sum(object_by_path.values())
    object_manifest_contract_passed = bool(
        object_payload.get("status") in {"verified", "complete"}
        and object_payload.get("source") == "gs://gresearch/robotics/droid"
        and len(object_rows) == expected_object_count
        and int(object_payload.get("object_count", -1))
        == expected_object_count
        and int(object_payload.get("total_bytes", -1))
        == expected_total_bytes
        and expected_object_count > 0
        and expected_total_bytes > 0
    )
    checksum_manifest_contract_passed = bool(
        checksum_payload.get("status") in {"verified", "complete"}
        and checksum_payload.get("source") == "gs://gresearch/robotics/droid"
        and len(checksum_rows) == len(checksum_by_path)
        and int(checksum_payload.get("object_count", -1))
        == expected_object_count
        and int(checksum_payload.get("total_bytes", -1))
        == expected_total_bytes
        and set(checksum_by_path) == set(object_by_path)
        and all(
            int(item.get("bytes", -1))
            == object_by_path[relative]
            and bool(item.get("md5_base64"))
            for relative, item in checksum_by_path.items()
        )
    )
    download_verification_semantic_passed = bool(
        verification_payload.get("status") == "complete"
        and verification_payload.get(
            "ready_for_full_allocation_training"
        )
        is True
        and verification_payload.get("official_source")
        == "gs://gresearch/robotics/droid"
        and int(
            verification_payload.get("manifest_object_count", -1)
        )
        == expected_object_count
        and int(
            verification_payload.get("manifest_duplicate_path_count", -1)
        )
        == 0
        and int(verification_payload.get("manifest_total_bytes", -1))
        == expected_total_bytes
        and int(verification_payload.get("remote_bytes", -1))
        == expected_total_bytes
        and int(verification_payload.get("local_official_bytes", -1))
        == expected_total_bytes
        and float(verification_payload.get("local_to_remote_ratio", 0.0))
        == 1.0
        and int(verification_payload.get("local_file_count", -1))
        == expected_object_count
        and int(verification_payload.get("complete_tfrecord_count", 0)) > 0
        and int(verification_payload.get("missing_object_count", -1)) == 0
        and int(verification_payload.get("size_mismatch_count", -1)) == 0
        and int(verification_payload.get("extra_file_count", -1)) == 0
        and int(verification_payload.get("partial_file_count", -1)) == 0
        and int(verification_payload.get("checksum_rsync_returncode", -1))
        == 0
    )
    route_guard = transport_payload.get("route_guard", {})
    route_observations = route_guard.get("observations", [])
    active_routes = sorted(
        {
            str(item.get("proxy", ""))
            for item in transport_payload.get("active", [])
            if item.get("proxy")
        }
    )
    forbidden_prefixes = (
        "utun",
        "tun",
        "tap",
        "ppp",
        "ipsec",
        "gif",
        "stf",
        "lo",
    )
    direct_transport_guard_passed = bool(
        args.transport_status
        and route_guard.get("enabled") is True
        and route_guard.get("status") == "passed"
        and route_observations
        and all(
            not str(item.get("interface", "")).lower().startswith(
                forbidden_prefixes
            )
            for item in route_observations
        )
        and all(route == "direct" for route in active_routes)
    )
    uniclash = uniclash_guard_payload.get("uniclash", {})
    bypass = uniclash_guard_payload.get("system_proxy_bypass", {})
    cumulative = uniclash_guard_payload.get("cumulative", {})
    expected_interface = str(
        uniclash_guard_payload.get("policy", {}).get(
            "expected_interface", "en1"
        )
    )
    uniclash_gate = validate_guard(
        uniclash_guard_payload,
        expected_interface=expected_interface,
        max_age_seconds=180.0,
    )
    uniclash_isolation_guard_passed = bool(
        args.uniclash_guard_status
        and uniclash_gate.get("status") == "passed"
    )

    gates = {
        "critical_code_present": not missing_code,
        "orchestration_snapshot_code_parity_passed": (
            snapshot_code_parity_passed
        ),
        "immutable_inputs_present": not missing_inputs,
        "backend_commit_resolved": (
            backend_commit["returncode"] == 0
            and len(backend_commit["stdout"]) == 40
        ),
        "backend_commit_pinned": (
            backend_commit["returncode"] == 0
            and backend_commit["stdout"] == EXPECTED_BACKEND_COMMIT
        ),
        "backend_origin_official": (
            backend_origin["returncode"] == 0
            and normalize_git_origin(backend_origin["stdout"])
            == EXPECTED_BACKEND_ORIGIN
        ),
        "backend_worktree_clean": (
            backend_status["returncode"] == 0
            and not backend_status_entries
        ),
        "backend_git_fsck_passed": backend_fsck["returncode"] == 0,
        "download_verification_present": bool(
            args.download_verification
            and args.download_verification.is_file()
        ),
        "object_manifest_contract_passed": (
            object_manifest_contract_passed
        ),
        "checksum_manifest_contract_passed": (
            checksum_manifest_contract_passed
        ),
        "download_verification_semantic_passed": (
            download_verification_semantic_passed
        ),
        "orico_is_mounted": mount_line is not None,
        "torch_available": bool(torch.__version__),
        "feature_dimension_is_ten": True,
        "direct_transport_guard_passed": direct_transport_guard_passed,
        "uniclash_isolation_guard_passed": (
            uniclash_isolation_guard_passed
        ),
    }
    status = "complete" if all(gates.values()) else "preflight"
    if args.require_final_inputs and status != "complete":
        status = "failed"

    payload = {
        "generated_at": now(),
        "status": status,
        "secret_policy": (
            "Only an explicit deterministic environment-variable allowlist is "
            "captured. Proxy URLs, credentials, tokens, and user data are excluded."
        ),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mac_ver": platform.mac_ver(),
            "sw_vers": sw_vers,
            "logical_cpu_count": (
                int(cpu_count["stdout"])
                if cpu_count["returncode"] == 0
                else os.cpu_count()
            ),
            "memory_bytes": (
                int(memory["stdout"])
                if memory["returncode"] == 0
                else None
            ),
        },
        "accelerator": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "selected_device_contract": (
                "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        },
        "storage": {
            "job_root": str(args.job_root),
            "orico_mount_line": mount_line,
            "disk_usage": disk_usage,
        },
        "transport_audit": {
            "path": (
                str(args.transport_status)
                if args.transport_status
                else None
            ),
            "status": transport_payload.get("status"),
            "route_guard": route_guard,
            "active_routes": active_routes,
            "failure_keys": sorted(
                transport_payload.get("failures", {}).keys()
            ),
        },
        "uniclash_isolation_audit": {
            "path": (
                str(args.uniclash_guard_status)
                if args.uniclash_guard_status
                else None
            ),
            "status": uniclash_guard_payload.get("status"),
            "policy": uniclash_guard_payload.get("policy", {}),
            "uniclash": uniclash,
            "system_proxy_bypass": bypass,
            "cumulative": cumulative,
            "transition_gate": uniclash_gate,
            "gate_passed": uniclash_isolation_guard_passed,
        },
        "download_verification_audit": {
            "path": (
                str(args.download_verification)
                if args.download_verification
                else None
            ),
            "status": verification_payload.get("status"),
            "ready_for_full_allocation_training": (
                verification_payload.get(
                    "ready_for_full_allocation_training"
                )
            ),
            "manifest_object_count": verification_payload.get(
                "manifest_object_count"
            ),
            "local_official_bytes": verification_payload.get(
                "local_official_bytes"
            ),
            "semantic_gate_passed": (
                download_verification_semantic_passed
            ),
        },
        "backend": {
            "path": str(args.backend_root),
            "commit": backend_commit["stdout"] or None,
            "expected_commit": EXPECTED_BACKEND_COMMIT,
            "origin": backend_origin["stdout"] or None,
            "expected_origin": EXPECTED_BACKEND_ORIGIN,
            "worktree_clean": (
                backend_status["returncode"] == 0
                and not backend_status_entries
            ),
            "worktree_status_entry_count": len(backend_status_entries),
            "git_status_returncode": backend_status["returncode"],
            "git_fsck_returncode": backend_fsck["returncode"],
        },
        "code": code,
        "orchestration_snapshot": {
            "manifest": (
                str(args.orchestration_snapshot_manifest)
                if args.orchestration_snapshot_manifest
                else None
            ),
            "manifest_sha256": sha256(
                args.orchestration_snapshot_manifest
            ) if args.orchestration_snapshot_manifest else None,
            "entry_count": len(snapshot_entries),
            "manifest_errors": snapshot_manifest_errors,
            "code_mismatch_count": len(snapshot_code_mismatches),
            "code_mismatch_sample": snapshot_code_mismatches[:20],
            "code_parity_passed": snapshot_code_parity_passed,
            "claim_boundary": (
                "Every critical workspace code hash must equal the hash in "
                "the atomically published ORICO orchestration snapshot."
            ),
        },
        "immutable_inputs": inputs,
        "training_contract": {
            "architecture": "AllocationHead(10→32→16→1)",
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "seed": 11,
            "holdout_fraction": 0.20,
            "holdout_method": (
                "release_stratified_official_relative_path_hash_v2"
            ),
            "evaluation_steps_per_arm": 20_000,
            "deployment_steps_per_arm": 20_000,
            "total_steps_per_arm": 40_000,
            "records_per_shard": 0,
            "bootstrap_samples": 5_000,
            "download_transport": (
                "direct curl --noproxy '*' plus direct gsutil, guarded "
                "against UniClash and tunnel routes"
            ),
            "forbidden_download_interfaces": (
                "utun|tun|tap|ppp|ipsec|gif|stf|lo"
            ),
        },
        "deterministic_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "packages": packages,
        "package_count": len(packages),
        "gates": gates,
        "missing_code": missing_code,
        "missing_inputs": missing_inputs,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "status": status, "gates": gates}))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
