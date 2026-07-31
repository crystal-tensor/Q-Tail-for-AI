#!/usr/bin/env python3
"""Audit the exact closure of the currently completed DROID mirror subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_OBJECTS = 4_102
OFFICIAL_TFRECORDS = 4_096
OFFICIAL_RECORDS = 187_891
LEDGER_ERROR_KINDS = {
    "ledger_not_official",
    "ledger_file_missing",
    "ledger_size",
    "local_size",
    "md5",
    "stale_mtime",
    "stale_ctime",
}
CACHE_ERROR_KINDS = {
    "cache_missing",
    "cache_sha256",
    "cache_identity",
    "duplicate_cache",
    "cache_not_md5_verified",
    "cache_source_identity",
    "cache_source_path",
    "cache_record_scan",
    "cache_record_count",
    "missing_cache",
    "unverified_cache",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_tfrecord(relative_path: str) -> bool:
    return "tfrecord" in Path(relative_path).name.lower()


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--record-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help=(
            "Fail unless the exact 4,102-object / 4,096-TFRecord / "
            "187,891-record formal closure is satisfied."
        ),
    )
    args = parser.parse_args()

    checksum_manifest = read_json(args.checksum_manifest)
    checksum_ledger = read_json(args.checksum_ledger)
    cache_manifest = read_json(args.cache_manifest)
    record_audit = read_json(args.record_audit)

    official = {
        str(item["relative_path"]): item
        for item in checksum_manifest.get("objects", [])
    }
    ledger = checksum_ledger.get("objects", {})
    errors: list[dict[str, Any]] = []
    error_counts: dict[str, int] = {}

    def add_error(kind: str, path: str, detail: str) -> None:
        error_counts[kind] = error_counts.get(kind, 0) + 1
        if len(errors) < 200:
            errors.append({"kind": kind, "path": path, "detail": detail})

    verified_bytes = 0
    completed_tfrecords: set[str] = set()
    release_summary: dict[str, dict[str, int]] = {}
    for relative, entry in ledger.items():
        relative = str(relative)
        expected = official.get(relative)
        local = args.data_dir / relative
        if expected is None:
            add_error("ledger_not_official", relative, "not in official manifest")
            continue
        if not local.is_file():
            add_error("ledger_file_missing", relative, "local file is missing")
            continue
        stat = local.stat()
        expected_bytes = int(expected.get("bytes", -1))
        if int(entry.get("bytes", -2)) != expected_bytes:
            add_error("ledger_size", relative, "ledger and official sizes differ")
        if stat.st_size != expected_bytes:
            add_error("local_size", relative, "local and official sizes differ")
        official_md5 = str(expected.get("md5_base64", ""))
        if (
            not official_md5
            or entry.get("official_md5_base64") != official_md5
            or entry.get("local_md5_base64") != official_md5
        ):
            add_error("md5", relative, "official/local MD5 closure failed")
        if int(entry.get("mtime_ns", -1)) != stat.st_mtime_ns:
            add_error("stale_mtime", relative, "file changed after MD5 ledger entry")
        if int(entry.get("ctime_ns", -1)) != stat.st_ctime_ns:
            add_error("stale_ctime", relative, "file changed after MD5 ledger entry")
        verified_bytes += stat.st_size
        if is_tfrecord(relative):
            completed_tfrecords.add(relative)
            release = Path(relative).parts[0]
            summary = release_summary.setdefault(
                release,
                {"objects": 0, "bytes": 0, "records": 0},
            )
            summary["objects"] += 1
            summary["bytes"] += stat.st_size

    cache_by_shard: dict[str, dict[str, Any]] = {}
    decoded_records = 0
    represented_bytes = 0
    for artifact in cache_manifest.get("artifacts", []):
        cache_path = Path(str(artifact.get("path", "")))
        if not cache_path.is_file():
            add_error("cache_missing", str(cache_path), "listed cache is missing")
            continue
        expected_sha = str(artifact.get("sha256", ""))
        if not expected_sha or sha256(cache_path) != expected_sha:
            add_error("cache_sha256", str(cache_path), "cache SHA-256 mismatch")
            continue
        cache = read_json(cache_path)
        identity = cache.get("identity", {})
        row = cache.get("row", {})
        relative = str(identity.get("relative_path", ""))
        if not relative:
            add_error("cache_identity", str(cache_path), "relative path missing")
            continue
        if relative in cache_by_shard:
            add_error("duplicate_cache", relative, "multiple listed caches")
            continue
        cache_by_shard[relative] = cache
        local = args.data_dir / relative
        if relative not in completed_tfrecords:
            add_error(
                "cache_not_md5_verified",
                relative,
                "cache source is absent from the verified-object ledger",
            )
            continue
        if (
            int(identity.get("bytes", -1)) != local.stat().st_size
            or int(identity.get("mtime_ns", -1)) != local.stat().st_mtime_ns
            or int(identity.get("ctime_ns", -1)) != local.stat().st_ctime_ns
        ):
            add_error("cache_source_identity", relative, "source identity is stale")
        if Path(str(row.get("path", ""))) != local:
            add_error("cache_source_path", relative, "cache points to another file")
        if (
            int(row.get("record_parse_ok", 0)) != 1
            or int(row.get("record_scan_complete", 0)) != 1
            or str(row.get("record_parse_error", ""))
        ):
            add_error("cache_record_scan", relative, "full record scan did not pass")
        records = int(row.get("records_decoded", -1))
        if records < 0:
            add_error("cache_record_count", relative, "invalid decoded record count")
            continue
        decoded_records += records
        represented_bytes += local.stat().st_size
        release = Path(relative).parts[0]
        release_summary.setdefault(
            release,
            {"objects": 0, "bytes": 0, "records": 0},
        )["records"] += records

    completed_without_cache = sorted(completed_tfrecords - set(cache_by_shard))
    cache_without_completed = sorted(set(cache_by_shard) - completed_tfrecords)
    source_snapshot_at = parse_timestamp(
        cache_manifest.get("source_snapshot_at")
    )
    deferred_after_snapshot = []
    missing_from_snapshot = []
    for relative in completed_without_cache:
        verified_at = parse_timestamp(
            ledger.get(relative, {}).get("verified_at")
        )
        if (
            source_snapshot_at is not None
            and verified_at is not None
            and verified_at >= source_snapshot_at
        ):
            deferred_after_snapshot.append(relative)
        else:
            missing_from_snapshot.append(relative)
    for relative in missing_from_snapshot[:100]:
        add_error("missing_cache", relative, "completed TFRecord has no listed cache")
    for relative in cache_without_completed[:100]:
        add_error("unverified_cache", relative, "listed cache has no completed source")

    partial_paths = sorted(
        str(path.relative_to(args.data_dir))
        for path in args.data_dir.rglob("*")
        if path.is_file()
        and (
            ".qtail.part" in path.name
            or path.name.endswith(".gstmp")
            or path.name.endswith(".tmp")
        )
    )
    partials_excluded = not (
        set(partial_paths) & (completed_tfrecords | set(cache_by_shard))
    )
    record_checks = {
        "record_audit_passed": record_audit.get("status") == "verified",
        "record_audit_has_no_errors": not record_audit.get("errors", []),
        "record_audit_cache_count_matches": int(
            record_audit.get("verified_cache_count", -1)
        )
        == len(cache_by_shard),
        "record_audit_decoded_records_match": int(
            record_audit.get("verified_decoded_records", -1)
        )
        == decoded_records,
        "record_audit_represented_bytes_match": int(
            record_audit.get("represented_bytes", -1)
        )
        == represented_bytes,
        "record_audit_official_shard_lengths_match": int(
            record_audit.get("record_count_mismatch_count", -1)
        )
        == 0
        and int(record_audit.get("metadata_error_count", -1)) == 0,
    }
    closure_checks = {
        "official_checksum_manifest_complete": (
            checksum_manifest.get("status") == "verified"
            and int(checksum_manifest.get("object_count", -1))
            == OFFICIAL_OBJECTS
            and len(official) == OFFICIAL_OBJECTS
        ),
        "verified_ledger_is_nonempty": bool(ledger),
        "ledger_entries_match_official_md5_and_live_files": not any(
            error_counts.get(kind, 0) for kind in LEDGER_ERROR_KINDS
        ),
        "every_completed_tfrecord_has_one_listed_cache": (
            not missing_from_snapshot and not cache_without_completed
        ),
        "listed_cache_integrity_passes": not any(
            error_counts.get(kind, 0) for kind in CACHE_ERROR_KINDS
        ),
        "cache_manifest_contract_excludes_unreferenced_files": (
            cache_manifest.get("selection_contract")
            == (
                "Only artifacts listed below are training inputs; "
                "unreferenced cache files are excluded."
            )
        ),
        "transport_partials_are_excluded": partials_excluded,
        **record_checks,
    }
    failed_checks = sorted(
        name for name, passed in closure_checks.items() if not passed
    )
    formal_ready = bool(
        not failed_checks
        and len(ledger) == OFFICIAL_OBJECTS
        and len(completed_tfrecords) == OFFICIAL_TFRECORDS
        and decoded_records == OFFICIAL_RECORDS
        and not partial_paths
        and not deferred_after_snapshot
        and set(release_summary) == {"1.0.0", "1.0.1"}
    )
    payload = {
        "format_version": "qtail_droid_incremental_closure_v2",
        "generated_at": now(),
        "status": (
            "failed"
            if failed_checks
            else "complete"
            if formal_ready
            else "passed_incremental"
        ),
        "claim_boundary": (
            "This proves byte identity, official MD5 closure, full-record "
            "decoding, and cache provenance only for the currently completed "
            "objects. It is not full-mirror or model-quality evidence until "
            "4,102 objects, 4,096 TFRecords, and 187,891 records all close. "
            "TFRecords verified after the cache source snapshot are explicitly "
            "deferred to the next prewarm pass, never counted as cached."
        ),
        "official_target": {
            "objects": OFFICIAL_OBJECTS,
            "tfrecords": OFFICIAL_TFRECORDS,
            "records": OFFICIAL_RECORDS,
        },
        "current_closure": {
            "verified_objects": len(ledger),
            "verified_bytes": verified_bytes,
            "completed_tfrecords": len(completed_tfrecords),
            "listed_verified_caches": len(cache_by_shard),
            "decoded_records": decoded_records,
            "represented_bytes": represented_bytes,
            "transport_partial_files": len(partial_paths),
            "unreferenced_cache_files": int(
                cache_manifest.get("unreferenced_cache_count", 0)
            ),
            "source_snapshot_at": cache_manifest.get("source_snapshot_at"),
            "deferred_after_snapshot_tfrecords": len(
                deferred_after_snapshot
            ),
            "missing_from_snapshot_tfrecords": len(missing_from_snapshot),
        },
        "release_closure": [
            {"release": release, **values}
            for release, values in sorted(release_summary.items())
        ],
        "formal_full_mirror_gate": formal_ready,
        "checks": closure_checks,
        "failed_checks": failed_checks,
        "error_count": sum(error_counts.values()),
        "error_counts": dict(sorted(error_counts.items())),
        "errors": errors,
        "errors_truncated": sum(error_counts.values()) > len(errors),
        "completed_without_cache_count": len(completed_without_cache),
        "deferred_after_snapshot_count": len(deferred_after_snapshot),
        "deferred_after_snapshot_sample": deferred_after_snapshot[:25],
        "missing_from_snapshot_count": len(missing_from_snapshot),
        "cache_without_completed_count": len(cache_without_completed),
        "partial_file_sample": partial_paths[:25],
        "input_artifacts": {
            "checksum_manifest": {
                "path": str(args.checksum_manifest),
                "sha256": sha256(args.checksum_manifest),
            },
            "checksum_ledger": {
                "path": str(args.checksum_ledger),
                "sha256": sha256(args.checksum_ledger),
            },
            "cache_manifest": {
                "path": str(args.cache_manifest),
                "sha256": sha256(args.cache_manifest),
            },
            "record_audit": {
                "path": str(args.record_audit),
                "sha256": sha256(args.record_audit),
            },
        },
    }
    atomic_write_json(args.out, payload)
    if failed_checks:
        raise SystemExit(
            "incremental closure failed: " + ", ".join(failed_checks)
        )
    if args.require_formal and not formal_ready:
        raise SystemExit(
            "formal closure required but full-mirror gate is false: "
            f"objects={len(ledger)}/{OFFICIAL_OBJECTS} "
            f"tfrecords={len(completed_tfrecords)}/{OFFICIAL_TFRECORDS} "
            f"records={decoded_records}/{OFFICIAL_RECORDS}"
        )


if __name__ == "__main__":
    main()
