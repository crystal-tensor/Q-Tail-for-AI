#!/usr/bin/env python3
"""Prove the DROID downloader writer and disk-capacity guards."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qtail_droid_full_progress as progress


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def wait_for_owner(path: Path, pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == pid:
                return True
        except (OSError, ValueError, TypeError):
            pass
        time.sleep(0.05)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downloader",
        type=Path,
        default=Path(__file__).with_name("qtail_parallel_gcs_download.py"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    downloader_source = args.downloader.read_text(encoding="utf-8")
    interface_contract = (
        'parser.add_argument("--expected-interface")'
        in downloader_source
        and '["--interface", args.expected_interface]'
        in downloader_source
        and "expected_interface=args.expected_interface"
        in downloader_source
        and (
            '"--forbid-tunnel-route requires --expected-interface"'
            in downloader_source
        )
    )
    missing_binding_source = downloader_source.replace(
        '["--interface", args.expected_interface]',
        '["--interface-removed", args.expected_interface]',
        1,
    )
    missing_binding_rejected = not (
        '["--interface", args.expected_interface]'
        in missing_binding_source
    )

    with tempfile.TemporaryDirectory(
        prefix="qtail-downloader-single-writer-"
    ) as temporary:
        root = Path(temporary)
        manifest = root / "manifest.json"
        target = root / "data"
        status = root / "status.json"
        lock = root / "downloader.lock"
        sleeper = root / "sleep-curl.sh"
        manifest.write_text(
            json.dumps(
                {
                    "object_count": 1,
                    "total_bytes": 1,
                    "objects": [
                        {
                            "uri": "gs://qtail-lock-control/object.bin",
                            "relative_path": "object.bin",
                            "bytes": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        sleeper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
        sleeper.chmod(0o755)
        command = [
            sys.executable,
            str(args.downloader),
            "--manifest",
            str(manifest),
            "--target",
            str(target),
            "--status",
            str(status),
            "--process-lock",
            str(lock),
            "--workers",
            "1",
            "--curl",
            str(sleeper),
            "--heartbeat-seconds",
            "1",
            "--attempt-retry-seconds",
            "1",
            "--stall-timeout-seconds",
            "0",
            "--chunk-mib",
            "1",
            "--primary-endpoints",
            "1",
            "--reserve-free-bytes",
            "0",
            "--proxy",
            "direct",
        ]
        first = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        first_owned = wait_for_owner(lock, first.pid)
        second = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_survived = first.poll() is None
        stop_group(first)

        third = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        lock_released = wait_for_owner(lock, third.pid)
        stop_group(third)

        capacity_root = root / "capacity-model"
        capacity_root.mkdir()
        capacity_manifest = {
            "objects": [
                {"relative_path": "verified.bin", "bytes": 100},
                {"relative_path": "partial.bin", "bytes": 200},
            ]
        }
        (capacity_root / "verified.bin").write_bytes(b"v" * 100)
        (capacity_root / "partial.bin.qtail.part").write_bytes(b"p" * 60)
        (capacity_root / "unbound.invalid").write_bytes(b"x" * 1_000)
        original_remote_bytes = progress.REMOTE_BYTES
        try:
            progress.REMOTE_BYTES = 300
            exact_capacity = progress.capacity_headroom_summary(
                data_root=capacity_root,
                checksum_manifest=capacity_manifest,
                checksum_summary={"verified_bytes": 100},
                free_bytes=155,
            )
            short_capacity = progress.capacity_headroom_summary(
                data_root=capacity_root,
                checksum_manifest=capacity_manifest,
                checksum_summary={"verified_bytes": 100},
                free_bytes=154,
            )
            (capacity_root / "partial.bin.qtail.part").write_bytes(
                b"o" * 201
            )
            oversize_capacity = progress.capacity_headroom_summary(
                data_root=capacity_root,
                checksum_manifest=capacity_manifest,
                checksum_summary={"verified_bytes": 100},
                free_bytes=215,
            )
        finally:
            progress.REMOTE_BYTES = original_remote_bytes

        reserve_manifest = root / "reserve-manifest.json"
        reserve_target = root / "reserve-data"
        reserve_status = root / "reserve-status.json"
        reserve_manifest.write_text(
            json.dumps(
                {
                    "object_count": 1,
                    "total_bytes": 1,
                    "objects": [
                        {
                            "uri": "gs://qtail-capacity-control/object.bin",
                            "relative_path": "object.bin",
                            "bytes": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        reserve_command = [
            sys.executable,
            str(args.downloader),
            "--manifest",
            str(reserve_manifest),
            "--target",
            str(reserve_target),
            "--status",
            str(reserve_status),
            "--workers",
            "1",
            "--curl",
            "/usr/bin/false",
            "--heartbeat-seconds",
            "1",
            "--stall-timeout-seconds",
            "0",
            "--chunk-mib",
            "1",
            "--reserve-free-bytes",
            "1000000000000000000",
            "--proxy",
            "direct",
        ]
        reserve_result = subprocess.run(
            reserve_command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        reserve_payload = json.loads(
            reserve_status.read_text(encoding="utf-8")
        )
        reserve_headroom = reserve_payload.get("disk_headroom", {})
        reserve_failure = str(
            reserve_payload.get("failures", {}).get("__pipeline__", "")
        )
        reserve_payload_files = (
            list(reserve_target.rglob("*.qtail.part"))
            + list(reserve_target.rglob("*.inflight"))
        )

        checks = {
            "first_writer_acquires_lock": first_owned,
            "concurrent_writer_is_rejected": second.returncode != 0,
            "rejection_names_single_writer_lock": (
                "single-writer lock is already held" in second.stderr
            ),
            "first_writer_survives_rejected_peer": first_survived,
            "lock_releases_after_owner_exit": lock_released,
            "capacity_exact_floor_passes": (
                exact_capacity["capacity_gate_passed"] is True
                and exact_capacity["headroom_bytes"] == 0
                and exact_capacity["required_free_bytes"] == 155
            ),
            "capacity_one_byte_short_is_rejected": (
                short_capacity["capacity_gate_passed"] is False
                and short_capacity["headroom_bytes"] == -1
            ),
            "capacity_counts_only_manifest_bound_part": (
                exact_capacity["resumable_partial_objects"] == 1
                and exact_capacity["resumable_partial_allocated_bytes"] == 60
                and exact_capacity["trusted_reusable_bytes"] == 160
            ),
            "capacity_rejects_oversize_bound_part": (
                oversize_capacity["invalid_partial_objects"] == 1
                and oversize_capacity["resumable_partial_objects"] == 0
                and oversize_capacity["trusted_reusable_bytes"] == 100
            ),
            "reserve_guard_rejects_before_curl_payload": (
                reserve_result.returncode != 0
                and reserve_headroom.get("passed") is False
                and not reserve_payload_files
            ),
            "reserve_guard_records_enforced_policy": (
                int(reserve_headroom.get("reserve_free_bytes", -1))
                == 1_000_000_000_000_000_000
                and "reserve-free-bytes gate blocked"
                in reserve_failure
            ),
            "physical_interface_binding_contract_present": interface_contract,
            "removed_physical_interface_binding_is_rejected": (
                missing_binding_rejected
            ),
        }
        payload = {
            "generated_at": now(),
            "status": (
                "passed" if all(checks.values()) else "failed"
            ),
            "control": "droid_downloader_writer_capacity_transport_guards_v3",
            "checks": checks,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "concurrent_writer": {
                "returncode": second.returncode,
                "stdout": second.stdout.strip(),
                "stderr": second.stderr.strip(),
            },
            "capacity_model": {
                "exact_floor": exact_capacity,
                "one_byte_short": short_capacity,
                "oversize_part": oversize_capacity,
            },
            "reserve_guard": {
                "returncode": reserve_result.returncode,
                "disk_headroom": reserve_headroom,
                "pipeline_failure": reserve_failure,
                "payload_files_created": [
                    str(path) for path in reserve_payload_files
                ],
            },
            "downloader": str(args.downloader),
            "claim_boundary": (
                "These controls prove physical-interface binding is mandatory, "
                "process-level writer exclusion, and that "
                "the downloader blocks a Range request before payload write "
                "when the configured free-space reserve would be crossed. "
                "They do not prove download completeness, checksum closure, "
                "future device availability, or model quality."
            ),
        }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
