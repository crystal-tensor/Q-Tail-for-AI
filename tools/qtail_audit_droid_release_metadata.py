#!/usr/bin/env python3
"""Verify official DROID release metadata and shared training schema."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_RELEASES = {
    "1.0.0": {
        "dataset_name": "r2d2_faceblur",
        "dataset_version": "1.4.0",
        "shards": 2_048,
        "records": 92_233,
        "split_bytes": 1_834_749_018_029,
    },
    "1.0.1": {
        "dataset_name": "droid_101",
        "dataset_version": "0.0.1",
        "shards": 2_048,
        "records": 95_658,
        "split_bytes": 1_865_993_126_270,
    },
}
EXPECTED_STEP_FEATURES = {
    "action",
    "action_dict",
    "discount",
    "is_first",
    "is_last",
    "is_terminal",
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "observation",
    "reward",
}
REQUIRED_TRAINING_FEATURES = {
    "action",
    "is_last",
    "is_terminal",
    "language_instruction",
    "observation",
    "reward",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


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
        temporary.unlink(missing_ok=True)


def file_audit(
    path: Path,
    relative_path: str,
    checksum_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = checksum_by_path.get(relative_path, {})
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_md5 = md5_base64(path) if exists else None
    expected_bytes = int(expected.get("bytes", -1))
    expected_md5 = str(expected.get("md5_base64", ""))
    return {
        "path": str(path),
        "relative_path": relative_path,
        "exists": exists,
        "bytes": actual_bytes,
        "sha256": sha256(path) if exists else None,
        "md5_base64": actual_md5,
        "official": {
            "bytes": expected_bytes,
            "md5_base64": expected_md5,
            "crc32c_base64": expected.get("crc32c_base64"),
            "generation": expected.get("generation"),
        },
        "verified": (
            exists
            and expected_bytes >= 0
            and len(expected_md5) > 0
            and actual_bytes == expected_bytes
            and actual_md5 == expected_md5
        ),
    }


def step_feature_keys(features: dict[str, Any]) -> set[str]:
    node: Any = features
    for key in (
        "featuresDict",
        "features",
        "steps",
        "sequence",
        "feature",
        "featuresDict",
        "features",
    ):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"features.json is missing schema node: {key}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError("step feature schema is not an object")
    return {str(key) for key in node}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checksum_manifest = json.loads(
        args.checksum_manifest.read_text(encoding="utf-8")
    )
    checksum_by_path = {
        str(item.get("relative_path")): item
        for item in checksum_manifest.get("objects", [])
        if isinstance(item, dict) and item.get("relative_path")
    }
    releases = []
    schemas: dict[str, list[str]] = {}
    for release, expected in EXPECTED_RELEASES.items():
        info_relative = f"{release}/dataset_info.json"
        features_relative = f"{release}/features.json"
        info_path = args.data_dir / info_relative
        features_path = args.data_dir / features_relative
        info_file = file_audit(
            info_path,
            info_relative,
            checksum_by_path,
        )
        features_file = file_audit(
            features_path,
            features_relative,
            checksum_by_path,
        )
        parse_error = None
        dataset_name = None
        dataset_version = None
        split_names: list[str] = []
        shard_count = 0
        record_count = 0
        split_bytes = 0
        keys: set[str] = set()
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = json.loads(features_path.read_text(encoding="utf-8"))
            dataset_name = str(info.get("name", ""))
            dataset_version = str(info.get("version", ""))
            splits = info.get("splits", [])
            if not isinstance(splits, list):
                raise ValueError("dataset_info splits is not a list")
            split_names = [str(split.get("name", "")) for split in splits]
            shard_count = sum(
                len(split.get("shardLengths", []))
                for split in splits
            )
            record_count = sum(
                sum(
                    int(value)
                    for value in split.get("shardLengths", [])
                )
                for split in splits
            )
            split_bytes = sum(
                int(split.get("numBytes", 0))
                for split in splits
            )
            keys = step_feature_keys(features)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            parse_error = str(error)
        schemas[release] = sorted(keys)
        releases.append(
            {
                "release": release,
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "split_names": split_names,
                "official_tfrecord_shards": shard_count,
                "official_records": record_count,
                "official_split_bytes": split_bytes,
                "step_feature_keys": sorted(keys),
                "required_training_features_present": (
                    REQUIRED_TRAINING_FEATURES.issubset(keys)
                ),
                "dataset_info_file": info_file,
                "features_file": features_file,
                "parse_error": parse_error,
                "verified": (
                    parse_error is None
                    and info_file["verified"]
                    and features_file["verified"]
                    and dataset_name == expected["dataset_name"]
                    and dataset_version == expected["dataset_version"]
                    and split_names == ["train"]
                    and shard_count == expected["shards"]
                    and record_count == expected["records"]
                    and split_bytes == expected["split_bytes"]
                    and keys == EXPECTED_STEP_FEATURES
                    and REQUIRED_TRAINING_FEATURES.issubset(keys)
                ),
            }
        )

    combined = {
        "tfrecord_shards": sum(
            int(item["official_tfrecord_shards"])
            for item in releases
        ),
        "records": sum(int(item["official_records"]) for item in releases),
        "split_bytes": sum(
            int(item["official_split_bytes"])
            for item in releases
        ),
    }
    canonical_schema = "\n".join(
        sorted(EXPECTED_STEP_FEATURES)
    ).encode("utf-8")
    gates = {
        "official_checksum_manifest": (
            checksum_manifest.get("status") == "verified"
            and checksum_manifest.get("source")
            == "gs://gresearch/robotics/droid"
            and int(checksum_manifest.get("object_count", -1)) == 4_102
        ),
        "both_releases_verified": (
            len(releases) == 2
            and all(item["verified"] for item in releases)
        ),
        "combined_shards_4096": combined["tfrecord_shards"] == 4_096,
        "combined_records_187891": combined["records"] == 187_891,
        "combined_split_bytes_match": (
            combined["split_bytes"] == 3_700_742_144_299
        ),
        "step_schemas_identical": (
            schemas["1.0.0"] == schemas["1.0.1"]
            and set(schemas["1.0.0"]) == EXPECTED_STEP_FEATURES
        ),
        "training_features_present": all(
            item["required_training_features_present"]
            for item in releases
        ),
    }
    payload = {
        "version": "droid_release_metadata_audit_v1",
        "status": "verified" if all(gates.values()) else "failed",
        "claim_boundary": (
            "This verifies official metadata and schema compatibility only. "
            "It does not prove that 4,096 TFRecord shards or 187,891 records "
            "have been downloaded and decoded."
        ),
        "source": "gs://gresearch/robotics/droid",
        "checksum_manifest": {
            "path": str(args.checksum_manifest),
            "sha256": sha256(args.checksum_manifest),
        },
        "expected_releases": EXPECTED_RELEASES,
        "required_training_features": sorted(REQUIRED_TRAINING_FEATURES),
        "canonical_step_schema_sha256": hashlib.sha256(
            canonical_schema
        ).hexdigest(),
        "releases": releases,
        "combined_official_metadata": combined,
        "gates": gates,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(
        {
            "out": str(args.out),
            "status": payload["status"],
            "gates": gates,
        },
        ensure_ascii=False,
    ))
    if payload["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
