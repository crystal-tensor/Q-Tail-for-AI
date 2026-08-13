#!/usr/bin/env python3
"""Verify selected Open X objects against the downloader's MD5 ledger."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checksum = json.loads(args.checksum_manifest.read_text(encoding="utf-8"))
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ledger = {"objects": {}}
    expected = {
        str(item["relative_path"]): int(item["bytes"])
        for item in manifest.get("objects", [])
    }
    official = {
        str(item["relative_path"]): item
        for item in checksum.get("objects", [])
    }
    ledger_objects = ledger.get("objects", {})
    if not isinstance(ledger_objects, dict):
        ledger_objects = {}

    complete = 0
    complete_bytes = 0
    md5_verified = 0
    missing: list[str] = []
    size_mismatch: list[dict[str, Any]] = []
    ledger_mismatch: list[str] = []
    partials: list[str] = []
    for relative, expected_bytes in expected.items():
        path = args.data_dir / relative
        part = path.with_name(path.name + ".qtail.part")
        inflight = part.with_name(part.name + ".inflight")
        if part.exists():
            partials.append(str(part))
        if inflight.exists():
            partials.append(str(inflight))
        try:
            stat = path.stat()
        except OSError:
            missing.append(relative)
            continue
        if stat.st_size != expected_bytes:
            size_mismatch.append(
                {
                    "relative_path": relative,
                    "expected": expected_bytes,
                    "actual": stat.st_size,
                }
            )
            continue
        complete += 1
        complete_bytes += expected_bytes
        entry = ledger_objects.get(relative, {})
        checksum_entry = official.get(relative, {})
        ledger_valid = (
            int(entry.get("bytes", -1)) == stat.st_size
            and int(entry.get("mtime_ns", -1)) == stat.st_mtime_ns
            and int(entry.get("ctime_ns", -1)) == stat.st_ctime_ns
            and entry.get("official_md5_base64") == checksum_entry.get("md5_base64")
            and entry.get("local_md5_base64") == checksum_entry.get("md5_base64")
            and entry.get("generation") == checksum_entry.get("generation")
        )
        if ledger_valid:
            md5_verified += 1
        else:
            ledger_mismatch.append(relative)

    expected_count = len(expected)
    expected_bytes = sum(expected.values())
    checks = {
        "manifest_verified": manifest.get("status") == "verified",
        "checksum_manifest_verified": checksum.get("status") == "verified",
        "manifest_checksum_paths_match": set(expected) == set(official),
        "all_objects_complete": complete == expected_count,
        "all_bytes_complete": complete_bytes == expected_bytes,
        "all_md5_ledger_entries_current": md5_verified == expected_count,
        "no_selected_partials": not partials,
        "no_missing_objects": not missing,
        "no_size_mismatches": not size_mismatch,
    }
    passed = all(checks.values())
    report = {
        "format_version": "qtail_openx_expansion_verification_v1",
        "generated_at": now(),
        "status": "verified" if passed else "incomplete",
        "checks": checks,
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "expected_objects": expected_count,
        "complete_objects": complete,
        "md5_verified_objects": md5_verified,
        "expected_bytes": expected_bytes,
        "complete_bytes": complete_bytes,
        "progress_percent": complete_bytes / max(expected_bytes, 1) * 100.0,
        "missing": missing[:100],
        "size_mismatch": size_mismatch[:100],
        "ledger_mismatch": ledger_mismatch[:100],
        "partials": partials[:100],
        "claim_boundary": (
            "MD5 content was computed by the downloader and is accepted here only "
            "when the official hash, generation, size, mtime and ctime still match."
        ),
    }
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_complete and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
