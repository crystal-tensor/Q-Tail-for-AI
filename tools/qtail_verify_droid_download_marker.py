#!/usr/bin/env python3
"""Create or verify an immutable DROID download-completion marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MARKER_VERSION = "droid_download_completion_marker_v1"


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
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_binding(
    *,
    data_dir: Path,
    manifest_path: Path,
    checksum_manifest_path: Path,
    checksum_ledger_path: Path,
    transport_status_path: Path,
    expected_bytes: int,
    expected_objects: int,
    expected_tfrecords: int,
) -> tuple[dict[str, Any], dict[str, bool], list[str]]:
    manifest = read_json(manifest_path)
    checksum_manifest = read_json(checksum_manifest_path)
    ledger = read_json(checksum_ledger_path)
    transport = read_json(transport_status_path)

    manifest_items = manifest.get("objects", [])
    checksum_items = checksum_manifest.get("objects", [])
    if not isinstance(manifest_items, list):
        manifest_items = []
    if not isinstance(checksum_items, list):
        checksum_items = []
    expected = {
        str(item.get("relative_path")): int(item.get("bytes", -1))
        for item in manifest_items
        if isinstance(item, dict) and item.get("relative_path")
    }
    checksums = {
        str(item.get("relative_path")): item
        for item in checksum_items
        if isinstance(item, dict) and item.get("relative_path")
    }
    ledger_objects = ledger.get("objects", {})
    if not isinstance(ledger_objects, dict):
        ledger_objects = {}

    identity = hashlib.sha256()
    file_errors: list[str] = []
    tfrecords = 0
    local_bytes = 0
    for relative in sorted(expected):
        path = data_dir / relative
        expected_size = expected[relative]
        checksum = checksums.get(relative, {})
        ledger_entry = ledger_objects.get(relative, {})
        try:
            stat = path.stat()
        except OSError as error:
            file_errors.append(f"{relative}: unreadable: {error}")
            continue
        if not path.is_file():
            file_errors.append(f"{relative}: not a regular file")
            continue
        local_bytes += stat.st_size
        if "tfrecord" in path.name.lower():
            tfrecords += 1
        expected_md5 = str(checksum.get("md5_base64", ""))
        if stat.st_size != expected_size:
            file_errors.append(
                f"{relative}: bytes={stat.st_size}/{expected_size}"
            )
        if (
            not isinstance(ledger_entry, dict)
            or int(ledger_entry.get("bytes", -1)) != expected_size
            or int(ledger_entry.get("mtime_ns", -1)) != stat.st_mtime_ns
            or int(ledger_entry.get("ctime_ns", -1)) != stat.st_ctime_ns
            or str(ledger_entry.get("official_md5_base64", ""))
            != expected_md5
            or str(ledger_entry.get("local_md5_base64", ""))
            != expected_md5
            or ledger_entry.get("generation") != checksum.get("generation")
        ):
            file_errors.append(f"{relative}: ledger identity mismatch")
        identity.update(relative.encode("utf-8"))
        identity.update(b"\0")
        identity.update(str(stat.st_size).encode("ascii"))
        identity.update(b"\0")
        identity.update(str(stat.st_mtime_ns).encode("ascii"))
        identity.update(b"\0")
        identity.update(str(stat.st_ctime_ns).encode("ascii"))
        identity.update(b"\0")
        identity.update(expected_md5.encode("ascii"))
        identity.update(b"\n")

    route_guard = transport.get("route_guard", {})
    failures = transport.get("failures", {})
    active = transport.get("active", [])
    checks = {
        "manifest_verified": manifest.get("status")
        in {"verified", "complete"},
        "manifest_exact_object_count": (
            len(manifest_items)
            == len(expected)
            == expected_objects
            == int(manifest.get("object_count", -1))
        ),
        "manifest_exact_total_bytes": (
            sum(expected.values())
            == expected_bytes
            == int(manifest.get("total_bytes", -1))
        ),
        "checksum_manifest_verified": (
            checksum_manifest.get("status") in {"verified", "complete"}
        ),
        "checksum_manifest_exact_identity": (
            len(checksum_items)
            == len(checksums)
            == expected_objects
            and set(checksums) == set(expected)
            and int(checksum_manifest.get("object_count", -1))
            == expected_objects
            and int(checksum_manifest.get("total_bytes", -1))
            == expected_bytes
            and all(
                int(checksums[path].get("bytes", -1)) == expected[path]
                and bool(checksums[path].get("md5_base64"))
                for path in expected
            )
        ),
        "checksum_ledger_exact_identity": (
            ledger.get("format_version") == 1
            and set(ledger_objects) == set(expected)
        ),
        "local_files_bound_to_ledger": not file_errors,
        "local_exact_bytes": local_bytes == expected_bytes,
        "local_exact_tfrecord_count": tfrecords == expected_tfrecords,
        "transport_declares_complete": transport.get("status") == "complete",
        "transport_exact_counts": (
            int(transport.get("object_count", -1)) == expected_objects
            and int(transport.get("expected_bytes", -1)) == expected_bytes
            and int(transport.get("completed_objects", -1))
            == expected_objects
            and int(transport.get("completed_bytes", -1)) == expected_bytes
            and int(transport.get("checksum_expected_objects", -1))
            == expected_objects
            and int(transport.get("checksum_verified_objects", -1))
            == expected_objects
        ),
        "transport_quiescent": failures == {} and active == [],
        "transport_direct_route_guard": (
            isinstance(route_guard, dict)
            and route_guard.get("enabled") is True
            and route_guard.get("status") == "passed"
        ),
    }
    binding = {
        "official_source": "gs://gresearch/robotics/droid",
        "data_dir": str(data_dir),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "checksum_manifest": {
            "path": str(checksum_manifest_path),
            "bytes": checksum_manifest_path.stat().st_size,
            "sha256": sha256(checksum_manifest_path),
        },
        "checksum_ledger": {
            "path": str(checksum_ledger_path),
            "bytes": checksum_ledger_path.stat().st_size,
            "sha256": sha256(checksum_ledger_path),
        },
        "transport_status": {
            "path": str(transport_status_path),
            "bytes": transport_status_path.stat().st_size,
            "sha256": sha256(transport_status_path),
        },
        "object_count": len(expected),
        "tfrecord_count": tfrecords,
        "official_bytes": expected_bytes,
        "local_bytes": local_bytes,
        "local_identity_sha256": identity.hexdigest(),
        "file_error_count": len(file_errors),
        "file_error_sample": file_errors[:50],
        "checks": checks,
    }
    return binding, checks, file_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--transport-status", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-objects", type=int, default=4_102)
    parser.add_argument("--expected-tfrecords", type=int, default=4_096)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    binding, checks, file_errors = build_binding(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        checksum_manifest_path=args.checksum_manifest,
        checksum_ledger_path=args.checksum_ledger,
        transport_status_path=args.transport_status,
        expected_bytes=args.expected_bytes,
        expected_objects=args.expected_objects,
        expected_tfrecords=args.expected_tfrecords,
    )
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit(
            "DROID download marker gate failed: "
            f"checks={failures} file_errors={len(file_errors)}"
        )

    existing: dict[str, Any] | None = None
    if args.marker.exists() and args.marker.stat().st_size:
        existing = read_json(args.marker)
    if args.write:
        if existing is not None:
            if (
                existing.get("marker_version") != MARKER_VERSION
                or existing.get("status") != "complete"
                or existing.get("immutable") is not True
                or existing.get("binding") != binding
            ):
                raise SystemExit(
                    "existing non-empty DROID download marker is not the "
                    "same immutable completion binding"
                )
        else:
            atomic_write_json(
                args.marker,
                {
                    "marker_version": MARKER_VERSION,
                    "generated_at": now(),
                    "status": "complete",
                    "immutable": True,
                    "binding": binding,
                },
            )
    else:
        if (
            existing is None
            or existing.get("marker_version") != MARKER_VERSION
            or existing.get("status") != "complete"
            or existing.get("immutable") is not True
            or existing.get("binding") != binding
        ):
            raise SystemExit(
                "DROID download completion marker does not match current "
                "manifest, ledger, files, and transport status"
            )

    print(
        json.dumps(
            {
                "status": "verified",
                "mode": "write" if args.write else "verify",
                "marker": str(args.marker),
                "object_count": binding["object_count"],
                "tfrecord_count": binding["tfrecord_count"],
                "official_bytes": binding["official_bytes"],
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
