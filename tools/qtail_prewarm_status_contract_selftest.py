#!/usr/bin/env python3
"""Positive and negative controls for DROID prewarm status semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qtail_train_droid_full import prewarm_snapshot_status


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = {
        "partial_snapshot_is_caught_up_not_complete": (
            prewarm_snapshot_status(
                shard_count=3_766,
                coverage_quality_passed=True,
                full_official_tfrecord_coverage=False,
            )
            == "prewarm_caught_up_current_snapshot"
        ),
        "exact_official_snapshot_is_complete": (
            prewarm_snapshot_status(
                shard_count=4_096,
                coverage_quality_passed=True,
                full_official_tfrecord_coverage=True,
            )
            == "prewarm_full_official_shard_snapshot_complete"
        ),
        "coverage_error_is_never_complete": (
            prewarm_snapshot_status(
                shard_count=4_096,
                coverage_quality_passed=False,
                full_official_tfrecord_coverage=True,
            )
            == "prewarm_current_snapshot_with_coverage_errors"
        ),
        "excess_shards_are_rejected": (
            prewarm_snapshot_status(
                shard_count=4_097,
                coverage_quality_passed=True,
                full_official_tfrecord_coverage=False,
            )
            == "prewarm_invalid_official_shard_coverage"
        ),
        "empty_snapshot_is_rejected": (
            prewarm_snapshot_status(
                shard_count=0,
                coverage_quality_passed=True,
                full_official_tfrecord_coverage=False,
            )
            == "prewarm_invalid_official_shard_coverage"
        ),
        "wrong_release_composition_is_rejected": (
            prewarm_snapshot_status(
                shard_count=4_096,
                coverage_quality_passed=True,
                full_official_tfrecord_coverage=False,
            )
            == "prewarm_invalid_official_shard_coverage"
        ),
    }
    payload = {
        "format_version": "qtail_prewarm_status_contract_selftest_v1",
        "status": "passed" if all(cases.values()) else "failed",
        "checks": cases,
        "passed": sum(bool(value) for value in cases.values()),
        "total": len(cases),
    }
    if args.out:
        atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
