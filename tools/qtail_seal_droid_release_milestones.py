#!/usr/bin/env python3
"""Seal immutable per-release DROID mirror/record milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASES = {
    "1.0.0": {
        "dataset_name": "r2d2_faceblur",
        "objects": 2_051,
        "tfrecords": 2_048,
        "records": 92_233,
        "object_tfrecord_bytes": 1_834_750_493_757,
        "dataset_info_split_bytes": 1_834_749_018_029,
    },
    "1.0.1": {
        "dataset_name": "droid_101",
        "objects": 2_051,
        "tfrecords": 2_048,
        "records": 95_658,
        "object_tfrecord_bytes": 1_865_994_656_798,
        "dataset_info_split_bytes": 1_865_993_126_270,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
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


def belongs_to_release(relative_path: str, release: str) -> bool:
    return (
        relative_path.startswith(f"{release}/")
        or relative_path == f"{release}_$folder$"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--milestone-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.checksum_manifest)
    ledger_payload = read_json(args.checksum_ledger)
    closure = read_json(args.closure)
    official = {
        str(item["relative_path"]): item
        for item in manifest.get("objects", [])
    }
    ledger = ledger_payload.get("objects", {})
    release_closure = {
        str(item.get("release")): item
        for item in closure.get("release_closure", [])
    }
    partial_paths = [
        str(path.relative_to(args.data_dir))
        for path in args.data_dir.rglob("*")
        if path.is_file()
        and (
            ".qtail.part" in path.name
            or path.name.endswith(".gstmp")
            or path.name.endswith(".tmp")
        )
    ]
    statuses: list[dict[str, Any]] = []
    invalid_existing: list[str] = []

    for release, expected in RELEASES.items():
        official_release = {
            relative: item
            for relative, item in official.items()
            if belongs_to_release(relative, release)
        }
        official_tfrecords = {
            relative: item
            for relative, item in official_release.items()
            if is_tfrecord(relative)
        }
        ledger_release = {
            relative: entry
            for relative, entry in ledger.items()
            if belongs_to_release(relative, release)
        }
        closure_release = release_closure.get(release, {})
        release_partials = sorted(
            relative
            for relative in partial_paths
            if belongs_to_release(relative, release)
        )
        ledger_identity_matches = bool(ledger_release) and all(
            relative in official_release
            and int(entry.get("bytes", -1))
            == int(official_release[relative].get("bytes", -2))
            and entry.get("official_md5_base64")
            == official_release[relative].get("md5_base64")
            and entry.get("local_md5_base64")
            == official_release[relative].get("md5_base64")
            and entry.get("generation")
            == official_release[relative].get("generation")
            for relative, entry in ledger_release.items()
        )
        checks = {
            "global_incremental_closure_passed": (
                closure.get("status") in {"passed_incremental", "complete"}
                and closure.get("error_count") == 0
                and all(closure.get("checks", {}).values())
            ),
            "official_release_object_count_exact": (
                len(official_release) == expected["objects"]
            ),
            "official_release_tfrecord_count_exact": (
                len(official_tfrecords) == expected["tfrecords"]
            ),
            "official_release_tfrecord_bytes_exact": (
                sum(int(item["bytes"]) for item in official_tfrecords.values())
                == expected["object_tfrecord_bytes"]
            ),
            "verified_ledger_covers_release_exactly": (
                set(ledger_release) == set(official_release)
            ),
            "verified_ledger_release_identity_matches": ledger_identity_matches,
            "decoded_release_tfrecord_count_exact": (
                int(closure_release.get("objects", -1))
                == expected["tfrecords"]
            ),
            "decoded_release_tfrecord_bytes_exact": (
                int(closure_release.get("bytes", -1))
                == expected["object_tfrecord_bytes"]
            ),
            "decoded_release_record_count_exact": (
                int(closure_release.get("records", -1))
                == expected["records"]
            ),
            "release_has_no_transport_partials": not release_partials,
        }
        complete = all(checks.values())
        milestone_path = (
            args.milestone_dir / f"droid_release_{release}_complete.json"
        )
        progress_path = (
            args.milestone_dir / f"droid_release_{release}_progress.json"
        )
        if milestone_path.exists():
            existing_at_complete_path = read_json(milestone_path)
            if (
                existing_at_complete_path.get("status") == "waiting"
                and existing_at_complete_path.get("immutable") is False
                and existing_at_complete_path.get("release") == release
            ):
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                milestone_path.replace(progress_path)
        if complete:
            official_subset = [
                official_release[relative]
                for relative in sorted(official_release)
            ]
            ledger_subset = {
                relative: ledger_release[relative]
                for relative in sorted(ledger_release)
            }
            milestone = {
                "format_version": "qtail_droid_release_milestone_v1",
                "generated_at": now(),
                "status": "complete",
                "immutable": True,
                "release": release,
                "dataset_name": expected["dataset_name"],
                "claim_boundary": (
                    "This seals byte, MD5, shard, and official record-count "
                    "closure for one DROID release. It is an input milestone, "
                    "not model-quality or policy-success evidence."
                ),
                "counts": {
                    "objects": len(ledger_release),
                    "tfrecords": int(closure_release["objects"]),
                    "records": int(closure_release["records"]),
                    "object_tfrecord_bytes": int(closure_release["bytes"]),
                    "dataset_info_split_bytes": expected[
                        "dataset_info_split_bytes"
                    ],
                    "metadata_to_object_byte_delta": (
                        expected["object_tfrecord_bytes"]
                        - expected["dataset_info_split_bytes"]
                    ),
                },
                "checks": checks,
                "official_subset_sha256": canonical_sha256(official_subset),
                "verified_ledger_subset_sha256": canonical_sha256(
                    ledger_subset
                ),
                "source_artifacts": {
                    "checksum_manifest": {
                        "path": str(args.checksum_manifest),
                        "sha256": file_sha256(args.checksum_manifest),
                    },
                    "checksum_ledger_at_seal": {
                        "path": str(args.checksum_ledger),
                        "sha256": file_sha256(args.checksum_ledger),
                    },
                    "incremental_closure_at_seal": {
                        "path": str(args.closure),
                        "sha256": file_sha256(args.closure),
                    },
                },
            }
            if milestone_path.exists():
                existing = read_json(milestone_path)
                if (
                    existing.get("status") == "waiting"
                    and existing.get("immutable") is False
                    and existing.get("release") == release
                ):
                    atomic_write_json(milestone_path, milestone)
                else:
                    stable_keys = (
                        "format_version",
                        "status",
                        "immutable",
                        "release",
                        "dataset_name",
                        "counts",
                        "checks",
                        "official_subset_sha256",
                        "verified_ledger_subset_sha256",
                    )
                    if any(
                        existing.get(key) != milestone.get(key)
                        for key in stable_keys
                    ):
                        invalid_existing.append(str(milestone_path))
                    milestone = existing
            else:
                atomic_write_json(milestone_path, milestone)
            statuses.append(
                {
                    "release": release,
                    "dataset_name": expected["dataset_name"],
                    "status": "complete",
                    "target": expected,
                    "milestone": str(milestone_path),
                    "milestone_sha256": file_sha256(milestone_path),
                    "progress": (
                        str(progress_path)
                        if progress_path.exists()
                        else None
                    ),
                    "checks": checks,
                }
            )
        else:
            waiting_milestone = {
                "format_version": "qtail_droid_release_milestone_v1",
                "generated_at": now(),
                "status": "waiting",
                "immutable": False,
                "release": release,
                "dataset_name": expected["dataset_name"],
                "claim_boundary": (
                    "This is a mutable progress placeholder, not a sealed "
                    "release milestone. It may transition once from WAITING "
                    "to immutable COMPLETE only after every release check "
                    "passes."
                ),
                "target": expected,
                "observed": {
                    "verified_objects": len(ledger_release),
                    "decoded_tfrecords": int(
                        closure_release.get("objects", 0)
                    ),
                    "decoded_records": int(
                        closure_release.get("records", 0)
                    ),
                    "release_partial_files": len(release_partials),
                },
                "checks": checks,
            }
            if milestone_path.exists():
                invalid_existing.append(str(milestone_path))
            else:
                atomic_write_json(progress_path, waiting_milestone)
            statuses.append(
                {
                    "release": release,
                    "dataset_name": expected["dataset_name"],
                    "status": "waiting",
                    "target": expected,
                    "progress": str(progress_path),
                    "progress_sha256": file_sha256(progress_path),
                    "milestone": None,
                    "milestone_sha256": None,
                    "observed": waiting_milestone["observed"],
                    "checks": checks,
                }
            )

    payload = {
        "format_version": "qtail_droid_release_milestone_status_v1",
        "generated_at": now(),
        "status": (
            "failed"
            if invalid_existing
            else "complete"
            if all(item["status"] == "complete" for item in statuses)
            else "in_progress"
        ),
        "claim_boundary": (
            "Mutable waiting evidence is written only to *_progress.json. "
            "A *_complete.json path is created only after all release checks "
            "pass and is then application-level append-once: later runs compare "
            "stable fields and fail on mismatch instead of replacing the sealed "
            "payload. This is not an APFS immutable flag, external timestamp, or "
            "WORM guarantee. Release milestones are input-closure evidence only; "
            "they do not prove Q-Tail model quality or robot-policy success."
        ),
        "immutability_scope": (
            "application_append_once_stable_field_comparison_not_apfs_flag_"
            "external_timestamp_or_worm"
        ),
        "release_count": len(statuses),
        "completed_release_count": sum(
            item["status"] == "complete" for item in statuses
        ),
        "releases": statuses,
        "invalid_existing_milestones": invalid_existing,
    }
    atomic_write_json(args.out, payload)
    if invalid_existing:
        raise SystemExit(
            "immutable release milestone mismatch: "
            + ", ".join(invalid_existing)
        )


if __name__ == "__main__":
    main()
