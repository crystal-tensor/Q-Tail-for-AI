#!/usr/bin/env python3
"""Quarantine downloader transport artifacts after the official mirror is complete."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def official_target_for(relative: str) -> str | None:
    if ".qtail.part" in relative:
        return relative.split(".qtail.part", 1)[0]
    if ".invalid-" in relative:
        return relative.split(".invalid-", 1)[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        str(item["relative_path"]): int(item["bytes"])
        for item in manifest.get("objects", [])
    }
    manifest_valid = (
        manifest.get("status") in {"verified", "complete"}
        and int(manifest.get("object_count", -1)) == len(expected)
        and len(expected) > 0
    )

    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    if manifest_valid:
        for relative, expected_bytes in expected.items():
            target = args.data_dir / relative
            if not target.is_file():
                missing.append(relative)
                continue
            actual_bytes = target.stat().st_size
            if actual_bytes != expected_bytes:
                size_mismatches.append(
                    {
                        "relative_path": relative,
                        "expected_bytes": expected_bytes,
                        "actual_bytes": actual_bytes,
                    }
                )

    candidates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for directory, _, names in os.walk(args.data_dir):
        for name in names:
            path = Path(directory) / name
            relative = str(path.relative_to(args.data_dir))
            if relative in expected or name == ".DS_Store" or name.startswith("._"):
                continue
            official_relative = official_target_for(relative)
            if official_relative is None:
                unresolved.append(
                    {
                        "relative_path": relative,
                        "bytes": path.stat().st_size,
                        "official_target": None,
                        "reason": "unrecognized_extra_file",
                    }
                )
                continue
            detail = {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "official_target": official_relative,
            }
            if official_relative not in expected:
                unresolved.append({**detail, "reason": "target_not_in_manifest"})
                continue
            target = args.data_dir / official_relative
            if not target.is_file():
                unresolved.append({**detail, "reason": "official_target_missing"})
                continue
            if target.stat().st_size != expected[official_relative]:
                unresolved.append(
                    {**detail, "reason": "official_target_size_mismatch"}
                )
                continue
            candidates.append(detail)

    preflight_passed = (
        manifest_valid
        and not missing
        and not size_mismatches
        and not unresolved
    )
    moved: list[dict[str, Any]] = []
    if preflight_passed:
        for item in candidates:
            source = args.data_dir / item["relative_path"]
            destination = args.quarantine_dir / item["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(
                    destination.name
                    + ".duplicate-"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                )
            source.replace(destination)
            moved.append(
                {
                    **item,
                    "quarantine_path": str(destination),
                }
            )

    payload = {
        "generated_at": now(),
        "status": "complete" if preflight_passed else "failed",
        "data_dir": str(args.data_dir),
        "manifest": str(args.manifest),
        "manifest_valid": manifest_valid,
        "expected_object_count": len(expected),
        "missing_object_count": len(missing),
        "missing_object_sample": missing[:20],
        "size_mismatch_count": len(size_mismatches),
        "size_mismatch_sample": size_mismatches[:20],
        "transport_candidate_count": len(candidates),
        "transport_candidate_bytes": sum(item["bytes"] for item in candidates),
        "unresolved_candidate_count": len(unresolved),
        "unresolved_candidate_sample": unresolved[:20],
        "quarantined_count": len(moved),
        "quarantined_bytes": sum(item["bytes"] for item in moved),
        "quarantined": moved,
        "policy": (
            "Only Q-Tail .qtail.part* and .invalid-* artifacts whose official "
            "target exists at the manifest size are moved. Unknown files are "
            "never removed automatically."
        ),
    }
    atomic_write_json(args.out, payload)
    if not preflight_passed:
        raise SystemExit(
            "DROID transport cleanup preflight failed: "
            f"manifest_valid={manifest_valid} missing={len(missing)} "
            f"mismatched={len(size_mismatches)} unresolved={len(unresolved)}"
        )


if __name__ == "__main__":
    main()
