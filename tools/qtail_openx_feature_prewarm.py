#!/usr/bin/env python3
"""Precompute Open X record features for checksum-verified expansion shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qtail_train_openx_demo as trainer


FORMAT_VERSION = "qtail_openx_feature_cache_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def cache_path(cache_dir: Path, relative_path: str) -> Path:
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    return cache_dir / key[:2] / f"{key}.json"


def cache_matches(
    payload: dict[str, Any],
    *,
    relative_path: str,
    source: Path,
    ledger_entry: dict[str, Any],
    records_per_shard: int,
) -> bool:
    try:
        stat = source.stat()
    except OSError:
        return False
    return bool(
        payload.get("format_version") == FORMAT_VERSION
        and payload.get("feature_extractor_version") == trainer.FEATURE_EXTRACTOR_VERSION
        and payload.get("relative_path") == relative_path
        and int(payload.get("records_per_shard", -1)) == records_per_shard
        and int(payload.get("bytes", -1)) == stat.st_size
        and int(payload.get("mtime_ns", -1)) == stat.st_mtime_ns
        and payload.get("official_md5_base64") == ledger_entry.get("official_md5_base64")
        and ledger_entry.get("official_md5_base64") == ledger_entry.get("local_md5_base64")
        and isinstance(payload.get("row"), dict)
    )


def build_row(source: Path, data_dir: Path, records_per_shard: int) -> dict[str, Any]:
    size = source.stat().st_size
    shard_idx, shard_total = trainer.shard_coordinates(source.name)
    return {
        "dataset": trainer.dataset_name(source, data_dir),
        "path": str(source),
        "bytes": size,
        "log_bytes": math.log1p(size),
        "shard_idx": shard_idx,
        "shard_total": shard_total,
        **trainer.aggregate_records(source, records_per_shard),
    }


def publish_status(
    path: Path,
    *,
    expected_objects: int,
    verified_objects: int,
    cached_objects: int,
    parsed_objects: int,
    failed_objects: int,
    records_decoded: int,
    active_relative_path: str,
    complete: bool,
    full_manifest_complete: bool = False,
) -> None:
    atomic_json(
        path,
        {
            "format_version": "qtail_openx_feature_prewarm_status_v1",
            "generated_at": now(),
            "status": "caught_up" if complete else "prewarming",
            "feature_extractor_version": trainer.FEATURE_EXTRACTOR_VERSION,
            "records_per_shard": 4,
            "expected_objects": expected_objects,
            "verified_objects": verified_objects,
            "cached_objects": cached_objects,
            "parsed_objects": parsed_objects,
            "failed_objects": failed_objects,
            "records_decoded": records_decoded,
            "active_relative_path": active_relative_path,
            "verified_cache_percent": (
                100.0 * cached_objects / verified_objects if verified_objects else 100.0
            ),
            "full_manifest_complete": full_manifest_complete,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--process-lock", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.records_per_shard <= 0:
        raise SystemExit("--records-per-shard must be positive for bounded prewarm")

    lock_handle = None
    if args.process_lock:
        args.process_lock.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = args.process_lock.open("a+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    manifest = load_json(args.manifest, {})
    ledger = load_json(args.ledger, {})
    expected_objects = sum(
        1
        for item in manifest.get("objects", [])
        if "tfrecord" in str(item.get("relative_path", "")).lower()
    )
    raw_ledger_objects = ledger.get("objects", {})
    if not isinstance(raw_ledger_objects, dict):
        raise SystemExit("checksum ledger objects must be a mapping")
    ledger_objects = {
        relative_path: entry
        for relative_path, entry in raw_ledger_objects.items()
        if "tfrecord" in relative_path.lower()
    }

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cached_objects = 0
    parsed_objects = 0
    failed_objects = 0
    records_decoded = 0
    new_objects = 0

    for relative_path, entry in sorted(ledger_objects.items()):
        if not isinstance(entry, dict):
            continue
        source = args.data_dir / relative_path
        target = cache_path(args.cache_dir, relative_path)
        payload = load_json(target, {})
        if cache_matches(
            payload,
            relative_path=relative_path,
            source=source,
            ledger_entry=entry,
            records_per_shard=args.records_per_shard,
        ):
            row = payload["row"]
            cached_objects += 1
            parsed_objects += int(bool(row.get("record_parse_ok")))
            failed_objects += int(not bool(row.get("record_parse_ok")))
            records_decoded += int(row.get("records_decoded", 0))
            continue

        if args.max_new and new_objects >= args.max_new:
            continue
        try:
            stat = source.stat()
        except OSError:
            continue
        if (
            stat.st_size != int(entry.get("bytes", -1))
            or entry.get("official_md5_base64") != entry.get("local_md5_base64")
        ):
            continue

        publish_status(
            args.status,
            expected_objects=expected_objects,
            verified_objects=len(ledger_objects),
            cached_objects=cached_objects,
            parsed_objects=parsed_objects,
            failed_objects=failed_objects,
            records_decoded=records_decoded,
            active_relative_path=relative_path,
            complete=False,
            full_manifest_complete=False,
        )
        row = build_row(source, args.data_dir, args.records_per_shard)
        payload = {
            "format_version": FORMAT_VERSION,
            "generated_at": now(),
            "feature_extractor_version": trainer.FEATURE_EXTRACTOR_VERSION,
            "records_per_shard": args.records_per_shard,
            "relative_path": relative_path,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "official_md5_base64": entry.get("official_md5_base64"),
            "generation": entry.get("generation"),
            "row": row,
        }
        atomic_json(target, payload)
        cached_objects += 1
        parsed_objects += int(bool(row.get("record_parse_ok")))
        failed_objects += int(not bool(row.get("record_parse_ok")))
        records_decoded += int(row.get("records_decoded", 0))
        new_objects += 1

    full_manifest_complete = bool(
        cached_objects == expected_objects
        and len(ledger_objects) == expected_objects
        and failed_objects == 0
    )
    publish_status(
        args.status,
        expected_objects=expected_objects,
        verified_objects=len(ledger_objects),
        cached_objects=cached_objects,
        parsed_objects=parsed_objects,
        failed_objects=failed_objects,
        records_decoded=records_decoded,
        active_relative_path="",
        complete=cached_objects == len(ledger_objects),
        full_manifest_complete=full_manifest_complete,
    )
    if args.require_complete and not full_manifest_complete:
        raise SystemExit(
            "feature prewarm incomplete: "
            f"cached={cached_objects}/{expected_objects} "
            f"verified={len(ledger_objects)}/{expected_objects} "
            f"failed={failed_objects}"
        )
    print(
        json.dumps(
            {
                "status": "caught_up" if cached_objects == len(ledger_objects) else "prewarming",
                "verified_objects": len(ledger_objects),
                "cached_objects": cached_objects,
                "new_objects": new_objects,
                "failed_objects": failed_objects,
            }
        )
    )


if __name__ == "__main__":
    main()
