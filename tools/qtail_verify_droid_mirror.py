#!/usr/bin/env python3
"""Verify a local DROID mirror against the exact official object manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_SOURCE = "gs://gresearch/robotics/droid"
FORMAL_EXPECTED_OBJECTS = 4_102
FORMAL_EXPECTED_TFRECORDS = 4_096


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument(
        "--expected-objects",
        type=int,
        default=FORMAL_EXPECTED_OBJECTS,
    )
    parser.add_argument(
        "--expected-tfrecords",
        type=int,
        default=FORMAL_EXPECTED_TFRECORDS,
    )
    parser.add_argument("--checksum-returncode", type=int, required=True)
    parser.add_argument(
        "--rehash-local",
        action="store_true",
        help="Recompute every local object's MD5 from bytes.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checksum_manifest = json.loads(
        args.checksum_manifest.read_text(encoding="utf-8")
    )
    checksum_ledger = json.loads(
        args.checksum_ledger.read_text(encoding="utf-8")
    )
    objects = manifest.get("objects", [])
    expected = {
        str(item["relative_path"]): int(item["bytes"])
        for item in objects
    }
    duplicate_manifest_paths = len(objects) - len(expected)
    checksum_objects = checksum_manifest.get("objects", [])
    checksums = {
        str(item["relative_path"]): item
        for item in checksum_objects
        if isinstance(item, dict) and "relative_path" in item
    }
    duplicate_checksum_paths = len(checksum_objects) - len(checksums)
    ledger_objects = checksum_ledger.get("objects", {})
    if not isinstance(ledger_objects, dict):
        ledger_objects = {}
    checksum_errors: list[dict[str, Any]] = []

    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    for relative, expected_bytes in expected.items():
        path = args.data_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            size_mismatches.append(
                {
                    "relative_path": relative,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                }
            )
            continue
        checksum = checksums.get(relative)
        ledger = ledger_objects.get(relative)
        if not isinstance(checksum, dict) or not isinstance(ledger, dict):
            checksum_errors.append(
                {
                    "relative_path": relative,
                    "error": "missing_checksum_or_ledger_entry",
                }
            )
            continue
        stat = path.stat()
        expected_md5 = str(checksum.get("md5_base64", ""))
        if (
            int(checksum.get("bytes", -1)) != expected_bytes
            or not expected_md5
            or int(ledger.get("bytes", -1)) != expected_bytes
            or str(ledger.get("official_md5_base64", "")) != expected_md5
            or str(ledger.get("local_md5_base64", "")) != expected_md5
            or ledger.get("generation") != checksum.get("generation")
            or int(ledger.get("mtime_ns", -1)) != stat.st_mtime_ns
            or int(ledger.get("ctime_ns", -1)) != stat.st_ctime_ns
        ):
            checksum_errors.append(
                {
                    "relative_path": relative,
                    "error": "checksum_ledger_binding_mismatch",
                }
            )
            continue
        if args.rehash_local:
            actual_md5 = file_md5_base64(path)
            if actual_md5 != expected_md5:
                checksum_errors.append(
                    {
                        "relative_path": relative,
                        "error": "local_byte_md5_mismatch",
                        "expected_md5_base64": expected_md5,
                        "actual_md5_base64": actual_md5,
                    }
                )

    actual_files: dict[str, int] = {}
    partials: list[str] = []
    ignored_metadata: list[str] = []
    tfrecords = 0
    for directory, _, names in os.walk(args.data_dir):
        for name in names:
            path = Path(directory) / name
            relative = str(path.relative_to(args.data_dir))
            if name == ".DS_Store" or name.startswith("._"):
                ignored_metadata.append(relative)
                continue
            actual_files[relative] = path.stat().st_size
            lowered = name.lower()
            if (
                ".gstmp" in lowered
                or ".qtail.part" in lowered
                or ".invalid-" in lowered
                or lowered.endswith((".part", ".tmp", ".inflight"))
            ):
                partials.append(relative)
            elif "tfrecord" in lowered:
                tfrecords += 1

    extras = sorted(set(actual_files) - set(expected))
    local_official_bytes = sum(
        actual_files.get(relative, 0)
        for relative in expected
    )
    ratio = (
        local_official_bytes / args.expected_bytes
        if args.expected_bytes
        else 0.0
    )
    ready = (
        args.checksum_returncode == 0
        and manifest.get("source") == OFFICIAL_SOURCE
        and checksum_manifest.get("source") == OFFICIAL_SOURCE
        and manifest.get("status") in {"verified", "complete"}
        and checksum_manifest.get("status") in {"verified", "complete"}
        and len(expected) == args.expected_objects
        and int(manifest.get("object_count", -1))
        == args.expected_objects
        and len(checksums) == args.expected_objects
        and int(checksum_manifest.get("object_count", -1))
        == args.expected_objects
        and int(manifest.get("total_bytes", -1)) == args.expected_bytes
        and int(checksum_manifest.get("total_bytes", -1))
        == args.expected_bytes
        and checksum_ledger.get("format_version") == 1
        and duplicate_manifest_paths == 0
        and duplicate_checksum_paths == 0
        and set(checksums) == set(expected)
        and set(ledger_objects) == set(expected)
        and not missing
        and not size_mismatches
        and not checksum_errors
        and not extras
        and not partials
        and len(actual_files) == len(expected)
        and local_official_bytes == args.expected_bytes
        and tfrecords == args.expected_tfrecords
    )
    payload = {
        "generated_at": now(),
        "status": "complete" if ready else "failed",
        "official_source": OFFICIAL_SOURCE,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "checksum_manifest": str(args.checksum_manifest),
        "checksum_manifest_sha256": file_sha256(args.checksum_manifest),
        "checksum_ledger": str(args.checksum_ledger),
        "checksum_ledger_sha256": file_sha256(args.checksum_ledger),
        "manifest_object_count": len(expected),
        "expected_object_count": args.expected_objects,
        "manifest_duplicate_path_count": duplicate_manifest_paths,
        "checksum_manifest_object_count": len(checksums),
        "checksum_manifest_duplicate_path_count": duplicate_checksum_paths,
        "checksum_ledger_object_count": len(ledger_objects),
        "manifest_total_bytes": int(manifest.get("total_bytes", -1)),
        "remote_bytes": args.expected_bytes,
        "local_official_bytes": local_official_bytes,
        "local_to_remote_ratio": ratio,
        "local_file_count": len(actual_files),
        "complete_tfrecord_count": tfrecords,
        "expected_tfrecord_count": args.expected_tfrecords,
        "missing_object_count": len(missing),
        "missing_object_sample": missing[:20],
        "size_mismatch_count": len(size_mismatches),
        "size_mismatch_sample": size_mismatches[:20],
        "checksum_error_count": len(checksum_errors),
        "checksum_error_sample": checksum_errors[:20],
        "extra_file_count": len(extras),
        "extra_file_sample": extras[:20],
        "partial_file_count": len(partials),
        "partial_file_sample": partials[:20],
        "ignored_filesystem_metadata_count": len(ignored_metadata),
        "ignored_filesystem_metadata_sample": ignored_metadata[:20],
        "checksum_rsync_returncode": args.checksum_returncode,
        "local_md5_rehash_requested": args.rehash_local,
        "local_md5_rehash_complete": (
            args.rehash_local
            and not missing
            and not size_mismatches
            and not checksum_errors
        ),
        "ready_for_full_allocation_training": ready,
    }
    atomic_write_json(args.out, payload)
    if not ready:
        raise SystemExit(
            "DROID full-data verification failed: "
            f"missing={len(missing)} mismatched={len(size_mismatches)} "
            f"checksum_errors={len(checksum_errors)} "
            f"extras={len(extras)} partials={len(partials)} "
            f"bytes={local_official_bytes}/{args.expected_bytes} "
            f"checksum_returncode={args.checksum_returncode}"
        )


if __name__ == "__main__":
    main()
