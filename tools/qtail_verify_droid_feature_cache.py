#!/usr/bin/env python3
"""Verify DROID feature caches against official shards and source identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_sha256(path: Path, window_bytes: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(16, "big", signed=False))
    with path.open("rb") as handle:
        digest.update(handle.read(window_bytes))
        if size > window_bytes:
            handle.seek(max(0, size - window_bytes))
            digest.update(handle.read(window_bytes))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def feature_rows_match(
    cached: dict[str, Any],
    recomputed: dict[str, Any],
) -> bool:
    if set(cached) != set(recomputed):
        return False
    for key in cached:
        left = cached[key]
        right = recomputed[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not (
                math.isfinite(float(left))
                and math.isfinite(float(right))
                and math.isclose(
                    float(left),
                    float(right),
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
            ):
                return False
        elif left != right:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--require-all-official-tfrecords", action="store_true")
    parser.add_argument(
        "--recompute-feature-values",
        action="store_true",
        help=(
            "Stream every source record again and require the cached feature "
            "row to equal an independent recomputation."
        ),
    )
    args = parser.parse_args()
    training = None
    if args.recompute_feature_values:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qtail_train_droid_full as training_module

        training = training_module

    object_manifest = json.loads(args.object_manifest.read_text(encoding="utf-8"))
    cache_manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    official = {
        str(item["relative_path"]): int(item["bytes"])
        for item in object_manifest.get("objects", [])
        if "tfrecord" in Path(str(item["relative_path"])).name.lower()
    }
    artifacts = cache_manifest.get("artifacts", [])
    errors: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_shards: set[str] = set()
    represented_bytes = 0
    official_record_expectations: dict[str, int] = {}
    decoded_records_by_shard: dict[str, int] = {}
    release_metadata: dict[str, dict[str, Any]] = {}
    metadata_error_count = 0
    record_count_mismatch_count = 0
    recomputed_feature_count = 0

    def add_error(kind: str, path: str, detail: str) -> None:
        errors.append({"kind": kind, "path": path, "detail": detail})

    def load_release_metadata(release: str) -> None:
        nonlocal metadata_error_count
        if release in release_metadata:
            return
        info_path = args.data_dir / release / "dataset_info.json"
        release_official = sorted(
            relative
            for relative in official
            if Path(relative).parts and Path(relative).parts[0] == release
        )
        metadata: dict[str, Any] = {
            "release": release,
            "dataset_info": str(info_path),
            "dataset_name": None,
            "official_tfrecord_count": len(release_official),
            "metadata_mapped_tfrecord_count": 0,
            "official_expected_records": 0,
        }
        release_metadata[release] = metadata
        if not info_path.is_file():
            metadata_error_count += 1
            add_error(
                "missing_dataset_info",
                str(info_path),
                "official release metadata is missing",
            )
            return
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            metadata_error_count += 1
            add_error("dataset_info_json", str(info_path), str(error))
            return
        dataset_name = str(info.get("name", ""))
        splits = info.get("splits", [])
        metadata["dataset_name"] = dataset_name
        if not dataset_name or not isinstance(splits, list) or not splits:
            metadata_error_count += 1
            add_error(
                "dataset_info_schema",
                str(info_path),
                "dataset name or split metadata is missing",
            )
            return

        mapped: dict[str, int] = {}
        for split in splits:
            split_name = str(split.get("name", ""))
            shard_lengths = split.get("shardLengths", [])
            if not split_name or not isinstance(shard_lengths, list) or not shard_lengths:
                metadata_error_count += 1
                add_error(
                    "dataset_info_split",
                    str(info_path),
                    f"invalid split metadata for {split_name or '<missing>'}",
                )
                continue
            try:
                expected_lengths = [int(value) for value in shard_lengths]
            except (TypeError, ValueError):
                metadata_error_count += 1
                add_error(
                    "dataset_info_shard_lengths",
                    str(info_path),
                    f"non-integer shardLengths for split {split_name}",
                )
                continue
            pattern = re.compile(
                rf"^{re.escape(dataset_name)}-{re.escape(split_name)}"
                rf"\.tfrecord-(\d+)-of-(\d+)$"
            )
            for relative in release_official:
                match = pattern.match(Path(relative).name)
                if not match:
                    continue
                index = int(match.group(1))
                declared_total = int(match.group(2))
                if declared_total != len(expected_lengths) or index >= declared_total:
                    metadata_error_count += 1
                    add_error(
                        "dataset_info_shard_index",
                        relative,
                        f"index={index} total={declared_total} "
                        f"metadata_total={len(expected_lengths)}",
                    )
                    continue
                mapped[relative] = expected_lengths[index]

        unmapped = sorted(set(release_official) - set(mapped))
        if unmapped:
            metadata_error_count += len(unmapped)
            for relative in unmapped[:100]:
                add_error(
                    "dataset_info_shard_mapping",
                    relative,
                    "official TFRecord is not represented by dataset_info.json",
                )
        official_record_expectations.update(mapped)
        metadata["metadata_mapped_tfrecord_count"] = len(mapped)
        metadata["official_expected_records"] = sum(mapped.values())

    for entry in artifacts:
        cache_path = Path(str(entry.get("path", "")))
        cache_key = str(cache_path)
        if cache_key in seen_paths:
            add_error("duplicate_cache", cache_key, "cache path appears more than once")
            continue
        seen_paths.add(cache_key)
        if not cache_path.is_file():
            add_error("missing_cache", cache_key, "cache file does not exist")
            continue
        if cache_path.stat().st_size != int(entry.get("bytes", -1)):
            add_error("cache_size", cache_key, "cache artifact byte count differs")
            continue
        if file_sha256(cache_path) != entry.get("sha256"):
            add_error("cache_sha256", cache_key, "cache artifact SHA-256 differs")
            continue
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            add_error("cache_json", cache_key, str(error))
            continue

        identity = cache.get("identity", {})
        row = cache.get("row", {})
        relative = str(identity.get("relative_path", ""))
        if not relative or relative in seen_shards:
            add_error("duplicate_shard", relative or cache_key, "missing or duplicate shard identity")
            continue
        seen_shards.add(relative)
        if any(token in relative.lower() for token in (".qtail.part", ".inflight", ".gstmp")):
            add_error("partial_source", relative, "partial transport file entered the cache")
            continue
        expected_bytes = official.get(relative)
        if expected_bytes is None:
            add_error("unknown_shard", relative, "shard is absent from the official manifest")
            continue
        source = args.data_dir / relative
        if not source.is_file():
            add_error("missing_source", relative, "source shard does not exist")
            continue
        source_stat = source.stat()
        represented_bytes += source_stat.st_size
        if source_stat.st_size != expected_bytes:
            add_error("official_size", relative, "source size differs from official manifest")
        if int(identity.get("bytes", -1)) != source_stat.st_size:
            add_error("identity_size", relative, "cache identity size differs")
        if int(identity.get("mtime_ns", -1)) != source_stat.st_mtime_ns:
            add_error("identity_mtime", relative, "cache identity mtime differs")
        if int(identity.get("ctime_ns", -1)) != source_stat.st_ctime_ns:
            add_error("identity_ctime", relative, "cache identity ctime differs")
        if int(identity.get("records_per_shard", -1)) != 0:
            add_error("record_scope", relative, "cache was not built in all-record mode")
        if boundary_sha256(source) != identity.get("boundary_sha256"):
            add_error("boundary_sha256", relative, "source boundary fingerprint differs")
        if int(row.get("bytes", -1)) != source_stat.st_size:
            add_error("row_size", relative, "feature row byte count differs")
        if int(row.get("record_parse_ok", 0)) != 1:
            add_error("record_parse", relative, str(row.get("record_parse_error", "")))
        if int(row.get("record_scan_complete", 0)) != 1:
            add_error("record_scan", relative, "full record scan is not complete")
        if training is not None:
            try:
                recomputed = training.build_row(
                    source,
                    args.data_dir,
                    0,
                    identity,
                )
                if not feature_rows_match(row, recomputed):
                    add_error(
                        "feature_recompute_mismatch",
                        relative,
                        "cached feature values differ from an independent "
                        "all-record recomputation",
                    )
                else:
                    recomputed_feature_count += 1
            except Exception as error:
                add_error(
                    "feature_recompute_error",
                    relative,
                    f"{type(error).__name__}: {error}",
                )
        release = Path(relative).parts[0]
        load_release_metadata(release)
        decoded_records = int(row.get("records_decoded", -1))
        decoded_records_by_shard[relative] = decoded_records
        expected_records = official_record_expectations.get(relative)
        if expected_records is None:
            record_count_mismatch_count += 1
            add_error(
                "official_record_count_missing",
                relative,
                "no official shardLengths expectation is available",
            )
        elif decoded_records != expected_records:
            record_count_mismatch_count += 1
            add_error(
                "official_record_count_mismatch",
                relative,
                f"decoded={decoded_records} official={expected_records}",
            )

    expected_cache_count = int(cache_manifest.get("expected_shard_count", -1))
    if int(cache_manifest.get("cache_count", -1)) != len(artifacts):
        add_error(
            "cache_count",
            str(args.cache_manifest),
            "declared cache count differs from artifact entries",
        )
    if expected_cache_count != len(artifacts):
        add_error(
            "expected_cache_count",
            str(args.cache_manifest),
            "selected cache count differs from expected shard count",
        )
    if cache_manifest.get("all_expected_caches_present") is not True:
        add_error(
            "cache_presence",
            str(args.cache_manifest),
            "cache manifest does not declare all selected caches present",
        )
    cache_dir = Path(str(cache_manifest.get("cache_dir", "")))
    directory_cache_files = (
        sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    )
    selected_cache_paths = {Path(path).resolve() for path in seen_paths}
    unreferenced_cache_files = [
        path
        for path in directory_cache_files
        if path.resolve() not in selected_cache_paths
    ]
    unreferenced_names = "\n".join(
        path.name for path in unreferenced_cache_files
    )
    unreferenced_name_sha256 = hashlib.sha256(
        unreferenced_names.encode("utf-8")
    ).hexdigest()
    unreferenced_cache_bytes = sum(
        path.stat().st_size for path in unreferenced_cache_files
    )
    if int(cache_manifest.get("cache_directory_count", -1)) != len(
        directory_cache_files
    ):
        add_error(
            "cache_directory_count",
            str(cache_dir),
            "cache directory file count differs from manifest",
        )
    if int(cache_manifest.get("unreferenced_cache_count", -1)) != len(
        unreferenced_cache_files
    ):
        add_error(
            "unreferenced_cache_count",
            str(cache_dir),
            "unreferenced cache count differs from manifest",
        )
    if int(cache_manifest.get("unreferenced_cache_bytes", -1)) != (
        unreferenced_cache_bytes
    ):
        add_error(
            "unreferenced_cache_bytes",
            str(cache_dir),
            "unreferenced cache bytes differ from manifest",
        )
    if cache_manifest.get("unreferenced_cache_name_sha256") != (
        unreferenced_name_sha256
    ):
        add_error(
            "unreferenced_cache_name_sha256",
            str(cache_dir),
            "unreferenced cache path digest differs from manifest",
        )
    if cache_manifest.get("selection_contract") != (
        "Only artifacts listed below are training inputs; "
        "unreferenced cache files are excluded."
    ):
        add_error(
            "cache_selection_contract",
            str(args.cache_manifest),
            "cache selection contract is missing or unexpected",
        )
    all_official = (
        len(seen_shards) == len(official)
        and seen_shards == set(official)
        and expected_cache_count == len(official)
    )
    if args.require_all_official_tfrecords and not all_official:
        add_error(
            "full_scope",
            str(args.cache_manifest),
            f"cache coverage {len(seen_shards)} / {len(official)} official TFRecords",
        )
    if represented_bytes != int(cache_manifest.get("represented_bytes", -1)):
        add_error(
            "represented_bytes",
            str(args.cache_manifest),
            "recomputed represented bytes differ from cache manifest",
        )

    release_record_audit = []
    for release, metadata in sorted(release_metadata.items()):
        release_seen = sorted(
            relative
            for relative in seen_shards
            if Path(relative).parts and Path(relative).parts[0] == release
        )
        expected_seen_records = sum(
            official_record_expectations.get(relative, 0)
            for relative in release_seen
        )
        decoded_seen_records = sum(
            decoded_records_by_shard.get(relative, 0)
            for relative in release_seen
        )
        release_record_audit.append(
            {
                **metadata,
                "verified_cache_count": len(release_seen),
                "verified_expected_records": expected_seen_records,
                "verified_decoded_records": decoded_seen_records,
                "verified_record_count_match": (
                    all(
                        relative in official_record_expectations
                        and decoded_records_by_shard.get(relative)
                        == official_record_expectations[relative]
                        for relative in release_seen
                    )
                ),
            }
        )
    official_record_counts_verified = (
        metadata_error_count == 0
        and record_count_mismatch_count == 0
        and all(relative in official_record_expectations for relative in seen_shards)
    )
    full_official_record_count_match = (
        all_official
        and official_record_counts_verified
        and len(official_record_expectations) == len(official)
        and sum(decoded_records_by_shard.values())
        == sum(official_record_expectations.values())
    )

    payload = {
        "generated_at": now(),
        "status": "verified" if not errors else "failed",
        "identity_algorithm": "sha256(size_u128_be || first_1MiB || last_1MiB)",
        "data_dir": str(args.data_dir),
        "object_manifest": str(args.object_manifest),
        "object_manifest_sha256": file_sha256(args.object_manifest),
        "cache_manifest": str(args.cache_manifest),
        "cache_manifest_sha256": file_sha256(args.cache_manifest),
        "official_tfrecord_count": len(official),
        "verified_cache_count": len(seen_shards),
        "cache_directory_count": len(directory_cache_files),
        "unreferenced_cache_count": len(unreferenced_cache_files),
        "unreferenced_cache_bytes": unreferenced_cache_bytes,
        "unreferenced_cache_name_sha256": unreferenced_name_sha256,
        "unreferenced_cache_excluded_from_training": True,
        "all_official_tfrecords": all_official,
        "official_record_counts_verified": official_record_counts_verified,
        "full_official_record_count_match": full_official_record_count_match,
        "feature_values_recomputed": args.recompute_feature_values,
        "recomputed_feature_count": recomputed_feature_count,
        "all_feature_values_recomputed": (
            args.recompute_feature_values
            and recomputed_feature_count == len(official)
        ),
        "official_expected_records": sum(official_record_expectations.values()),
        "verified_decoded_records": sum(decoded_records_by_shard.values()),
        "metadata_error_count": metadata_error_count,
        "record_count_mismatch_count": record_count_mismatch_count,
        "release_record_audit": release_record_audit,
        "represented_bytes": represented_bytes,
        "error_count": len(errors),
        "errors": errors[:100],
        "errors_truncated": len(errors) > 100,
    }
    atomic_write_json(args.out, payload)

    if not errors and args.artifact_manifest:
        manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
        entries = {
            str(entry["path"]): entry
            for entry in manifest.get("artifacts", [])
        }
        entries[str(args.out)] = artifact_entry(args.out)
        atomic_write_json(
            args.artifact_manifest,
            {
                **manifest,
                "generated_at": now(),
                "status": "complete",
                "artifacts": [
                    entries[path]
                    for path in sorted(entries)
                ],
            },
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
