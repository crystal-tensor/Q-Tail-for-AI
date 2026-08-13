#!/usr/bin/env python3
"""Black-box positive and negative controls for the DROID environment gate."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "qtail_droid_environment_contract_selftest_v3"
EXPECTED_BACKEND_COMMIT = "9a29c832b4c81bf38401111f5e4cdddaca217581"
EXPECTED_BACKEND_ORIGIN = (
    "https://github.com/droid-dataset/droid_policy_learning"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def run_checked(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_capture(
    *,
    capture_script: Path,
    repo_root: Path,
    job_root: Path,
    pt_source: Path,
    object_manifest: Path,
    checksum_manifest: Path,
    verification: Path,
    transport_status: Path,
    uniclash_guard_status: Path,
    backend_root: Path,
    orchestration_snapshot_manifest: Path,
    out: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(capture_script),
        "--repo-root",
        str(repo_root),
        "--job-root",
        str(job_root),
        "--out",
        str(out),
        "--pt-source",
        str(pt_source),
        "--object-manifest",
        str(object_manifest),
        "--checksum-manifest",
        str(checksum_manifest),
        "--download-verification",
        str(verification),
        "--transport-status",
        str(transport_status),
        "--uniclash-guard-status",
        str(uniclash_guard_status),
        "--backend-root",
        str(backend_root),
        "--orchestration-snapshot-manifest",
        str(orchestration_snapshot_manifest),
        "--require-final-inputs",
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    environment = read_json(out) if out.is_file() else {}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "environment_status": environment.get("status"),
        "gates": environment.get("gates", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-source", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--transport-status", type=Path, required=True)
    parser.add_argument(
        "--uniclash-guard-status", type=Path, required=True
    )
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument(
        "--orchestration-snapshot-manifest", type=Path, required=True
    )
    args = parser.parse_args()

    capture_script = (
        args.repo_root / "tools" / "qtail_capture_droid_environment.py"
    )
    object_payload = read_json(args.object_manifest)
    checksum_payload = read_json(args.checksum_manifest)
    checksum_by_path = {
        str(item["relative_path"]): item
        for item in checksum_payload.get("objects", [])
    }
    fixture_object = next(
        (
            item
            for item in object_payload.get("objects", [])
            if "tfrecord" in str(item.get("relative_path", "")).lower()
            and str(item.get("relative_path")) in checksum_by_path
        ),
        None,
    )
    if not fixture_object:
        raise SystemExit("No checksummed TFRecord object available for self-test")
    relative = str(fixture_object["relative_path"])
    expected_bytes = int(fixture_object["bytes"])
    fixture_checksum = checksum_by_path[relative]

    temporary_root = args.job_root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="droid-environment-contract-",
        dir=temporary_root,
    ) as temporary_name:
        root = Path(temporary_name)
        backend_fixture = root / "backend_fixture"
        clone_result = run_checked(
            [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                str(args.backend_root),
                str(backend_fixture),
            ]
        )
        checkout_result = run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "checkout",
                "--detach",
                EXPECTED_BACKEND_COMMIT,
            ]
        )
        run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "remote",
                "set-url",
                "origin",
                EXPECTED_BACKEND_ORIGIN,
            ]
        )
        run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "config",
                "user.name",
                "Q-Tail Environment Self-Test",
            ]
        )
        run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "config",
                "user.email",
                "qtail-selftest@invalid.example",
            ]
        )
        mini_object_manifest = {
            "status": "verified",
            "source": "gs://gresearch/robotics/droid",
            "object_count": 1,
            "total_bytes": expected_bytes,
            "objects": [fixture_object],
        }
        mini_checksum_manifest = {
            "status": "verified",
            "source": "gs://gresearch/robotics/droid",
            "object_count": 1,
            "total_bytes": expected_bytes,
            "objects": [fixture_checksum],
        }
        positive_verification = {
            "status": "complete",
            "official_source": "gs://gresearch/robotics/droid",
            "ready_for_full_allocation_training": True,
            "manifest_object_count": 1,
            "manifest_duplicate_path_count": 0,
            "manifest_total_bytes": expected_bytes,
            "remote_bytes": expected_bytes,
            "local_official_bytes": expected_bytes,
            "local_to_remote_ratio": 1.0,
            "local_file_count": 1,
            "complete_tfrecord_count": 1,
            "missing_object_count": 0,
            "size_mismatch_count": 0,
            "extra_file_count": 0,
            "partial_file_count": 0,
            "checksum_rsync_returncode": 0,
        }
        negative_byte_verification = copy.deepcopy(positive_verification)
        negative_byte_verification["local_official_bytes"] = (
            expected_bytes + 1
        )
        negative_checksum_manifest = copy.deepcopy(mini_checksum_manifest)
        negative_checksum_manifest["objects"][0]["md5_base64"] = ""
        negative_uniclash_guard = read_json(args.uniclash_guard_status)
        negative_uniclash_guard.setdefault("cumulative", {})[
            "blocked_samples"
        ] = 1
        negative_uniclash_guard["cumulative"]["violation_events"] = [
            {
                "at": now(),
                "global_violations": [],
                "blocked_processes": [999_999],
                "transfer_violations": [
                    {
                        "pid": 999_999,
                        "kind": "curl",
                        "violations": [
                            "socket reached UniClash loopback/7993"
                        ],
                    }
                ],
            }
        ]

        object_path = root / "object_manifest.json"
        checksum_path = root / "checksum_manifest.json"
        checksum_negative_path = root / "checksum_manifest_missing_md5.json"
        verification_path = root / "verification.json"
        verification_negative_path = root / "verification_byte_mismatch.json"
        uniclash_negative_path = root / "uniclash_guard_violation.json"
        write_json(object_path, mini_object_manifest)
        write_json(checksum_path, mini_checksum_manifest)
        write_json(checksum_negative_path, negative_checksum_manifest)
        write_json(verification_path, positive_verification)
        write_json(
            verification_negative_path,
            negative_byte_verification,
        )
        write_json(uniclash_negative_path, negative_uniclash_guard)
        snapshot_negative_path = root / "orchestration_snapshot_drift.SHA256SUMS"
        snapshot_lines = args.orchestration_snapshot_manifest.read_text(
            encoding="utf-8"
        ).splitlines()
        trainer_suffix = "./tools/qtail_train_droid_full.py"
        trainer_line_indexes = [
            index
            for index, line in enumerate(snapshot_lines)
            if line.endswith(trainer_suffix)
        ]
        if len(trainer_line_indexes) != 1:
            raise SystemExit(
                "ORICO snapshot must contain exactly one trainer entry"
            )
        trainer_index = trainer_line_indexes[0]
        snapshot_lines[trainer_index] = (
            "0" * 64 + "  " + trainer_suffix
        )
        snapshot_negative_path.write_text(
            "\n".join(snapshot_lines) + "\n",
            encoding="utf-8",
        )

        positive = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_positive.json",
        )
        negative_snapshot = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=snapshot_negative_path,
            out=root / "environment_negative_snapshot.json",
        )
        negative_byte = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_negative_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_byte.json",
        )
        negative_checksum = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_negative_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_checksum.json",
        )
        negative_uniclash = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=uniclash_negative_path,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_uniclash.json",
        )
        classifier_selftest_path = root / "classifier_v6_selftest.json"
        classifier_selftest_result = subprocess.run(
            [
                sys.executable,
                str(
                    args.repo_root
                    / "tools"
                    / "qtail_uniclash_transport_guard.py"
                ),
                "--classifier-selftest-out",
                str(classifier_selftest_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        classifier_selftest = (
            read_json(classifier_selftest_path)
            if classifier_selftest_path.is_file()
            else {}
        )
        dirty_probe = backend_fixture / ".qtail_environment_dirty_probe"
        dirty_probe.write_text("intentional self-test dirt\n", encoding="utf-8")
        negative_dirty_backend = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_dirty_backend.json",
        )
        dirty_probe.unlink()
        run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "remote",
                "set-url",
                "origin",
                "https://example.invalid/not-official",
            ]
        )
        negative_backend_origin = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_backend_origin.json",
        )
        run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "remote",
                "set-url",
                "origin",
                EXPECTED_BACKEND_ORIGIN,
            ]
        )
        drift_commit_result = run_checked(
            [
                "git",
                "-C",
                str(backend_fixture),
                "commit",
                "--allow-empty",
                "-m",
                "intentional environment self-test commit drift",
            ]
        )
        negative_backend_commit = run_capture(
            capture_script=capture_script,
            repo_root=args.repo_root,
            job_root=args.job_root,
            pt_source=args.pt_source,
            object_manifest=object_path,
            checksum_manifest=checksum_path,
            verification=verification_path,
            transport_status=args.transport_status,
            uniclash_guard_status=args.uniclash_guard_status,
            backend_root=backend_fixture,
            orchestration_snapshot_manifest=(
                args.orchestration_snapshot_manifest
            ),
            out=root / "environment_negative_backend_commit.json",
        )

    checks = {
        "positive_control_completes": (
            positive["returncode"] == 0
            and positive["environment_status"] == "complete"
            and all(positive["gates"].values())
        ),
        "one_byte_mirror_mismatch_fails": (
            negative_byte["returncode"] != 0
            and negative_byte["environment_status"] == "failed"
            and negative_byte["gates"].get(
                "download_verification_semantic_passed"
            )
            is False
        ),
        "orchestration_snapshot_code_drift_fails": (
            negative_snapshot["returncode"] != 0
            and negative_snapshot["environment_status"] == "failed"
            and negative_snapshot["gates"].get(
                "orchestration_snapshot_code_parity_passed"
            )
            is False
        ),
        "missing_official_md5_fails": (
            negative_checksum["returncode"] != 0
            and negative_checksum["environment_status"] == "failed"
            and negative_checksum["gates"].get(
                "checksum_manifest_contract_passed"
            )
            is False
        ),
        "uniclash_violation_fails": (
            negative_uniclash["returncode"] != 0
            and negative_uniclash["environment_status"] == "failed"
            and negative_uniclash["gates"].get(
                "uniclash_isolation_guard_passed"
            )
            is False
        ),
        "transport_classifier_v6_selftest_passes": (
            classifier_selftest_result.returncode == 0
            and classifier_selftest.get("status") == "passed"
            and classifier_selftest.get("classifier_version")
            == "droid_transport_downloader_descendants_v6_interface_bound_live"
            and all(classifier_selftest.get("checks", {}).values())
        ),
        "backend_commit_drift_fails": (
            negative_backend_commit["returncode"] != 0
            and negative_backend_commit["environment_status"] == "failed"
            and negative_backend_commit["gates"].get(
                "backend_commit_pinned"
            )
            is False
            and negative_backend_commit["gates"].get(
                "backend_worktree_clean"
            )
            is True
        ),
        "backend_origin_drift_fails": (
            negative_backend_origin["returncode"] != 0
            and negative_backend_origin["environment_status"] == "failed"
            and negative_backend_origin["gates"].get(
                "backend_origin_official"
            )
            is False
        ),
        "backend_worktree_dirty_fails": (
            negative_dirty_backend["returncode"] != 0
            and negative_dirty_backend["environment_status"] == "failed"
            and negative_dirty_backend["gates"].get(
                "backend_worktree_clean"
            )
            is False
        ),
    }
    passed = all(checks.values())
    payload = {
        "generated_at": now(),
        "contract_version": CONTRACT_VERSION,
        "status": "passed" if passed else "failed",
        "backend_fixture": {
            "source": str(args.backend_root),
            "expected_commit": EXPECTED_BACKEND_COMMIT,
            "expected_origin": EXPECTED_BACKEND_ORIGIN,
            "clone_returncode": clone_result["returncode"],
            "checkout_returncode": checkout_result["returncode"],
            "drift_commit_returncode": drift_commit_result["returncode"],
        },
        "fixture": {
            "relative_path": relative,
            "expected_bytes": expected_bytes,
            "official_md5_base64": fixture_checksum.get("md5_base64"),
        },
        "checks": checks,
        "positive_control": positive,
        "negative_byte_control": negative_byte,
        "negative_checksum_control": negative_checksum,
        "negative_uniclash_control": negative_uniclash,
        "negative_backend_commit_control": negative_backend_commit,
        "negative_backend_origin_control": negative_backend_origin,
        "negative_backend_worktree_control": negative_dirty_backend,
        "transport_classifier_v6_selftest": classifier_selftest,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "status": payload["status"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
