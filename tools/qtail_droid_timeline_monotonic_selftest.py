#!/usr/bin/env python3
"""Exercise legacy and committed feature-counter timeline semantics."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qtail_verify_droid_timeline import (
    TIMELINE_VERSION,
    canonical_sha256,
    verify_timeline,
)


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


def sample(
    *,
    sequence: int,
    previous_hash: str | None,
    generated_at: str,
    processed_shards: int,
    counter_semantics: str | None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "processed_shards": processed_shards,
        "total_shards": 4_096,
        "records_decoded": processed_shards * 40,
        "record_scan_complete_shards": processed_shards,
        "parse_errors": 0,
    }
    if counter_semantics is not None:
        features["counter_semantics"] = counter_semantics
    payload: dict[str, Any] = {
        "sequence": sequence,
        "previous_sample_sha256": previous_hash,
        "generated_at": generated_at,
        "kind": "full_pipeline_sample",
        "stage": "droid_full_download",
        "status": "in_progress",
        "download": {
            "physical_bytes": 1_000 + sequence,
            "completed_objects": 10 + sequence,
        },
        "object_checksums": {
            "verified_objects": 10 + sequence,
            "checksum_errors": 0,
        },
        "feature_extraction": features,
        "transport_isolation": {
            "status": "passed",
            "core_running": True,
            "tun_enabled": False,
            "guard_samples": 100 + sequence,
            "blocked_samples": 0,
            "forbidden_socket_observations": 0,
            "wrong_route_observations": 0,
            "guard_generated_at": generated_at,
            "guard_age_seconds": 0.2,
            "guard_process_count": 1,
        },
        "completion": {
            "passed_requirements": 4,
            "total_requirements": 9,
        },
        "runtime": {
            "healthy": True,
            "heartbeat_gate_passed": True,
            "mount_gate_passed": True,
            "web_gate_passed": True,
        },
    }
    payload["sample_sha256"] = canonical_sha256(payload)
    return payload


def write_timeline(
    path: Path,
    *,
    first_count: int,
    second_count: int,
    counter_semantics: str | None,
) -> None:
    first = sample(
        sequence=0,
        previous_hash=None,
        generated_at="2026-07-29T00:00:00+00:00",
        processed_shards=first_count,
        counter_semantics=counter_semantics,
    )
    second = sample(
        sequence=1,
        previous_hash=first["sample_sha256"],
        generated_at="2026-07-29T00:01:00+00:00",
        processed_shards=second_count,
        counter_semantics=counter_semantics,
    )
    atomic_write_json(
        path,
        {
            "version": TIMELINE_VERSION,
            "status": "in_progress",
            "retention": "full_pipeline_history",
            "sample_count": 2,
            "first_generated_at": first["generated_at"],
            "last_generated_at": second["generated_at"],
            "chain_head_sha256": second["sample_sha256"],
            "samples": [first, second],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    monotonic = "monotonic_committed_prewarm_snapshot_v1"

    with tempfile.TemporaryDirectory(prefix="qtail-timeline-selftest-") as tmp:
        root = Path(tmp)
        legacy_reset_path = root / "legacy-reset.json"
        committed_reset_path = root / "committed-reset.json"
        committed_growth_path = root / "committed-growth.json"
        write_timeline(
            legacy_reset_path,
            first_count=100,
            second_count=20,
            counter_semantics=None,
        )
        write_timeline(
            committed_reset_path,
            first_count=100,
            second_count=20,
            counter_semantics=monotonic,
        )
        write_timeline(
            committed_growth_path,
            first_count=100,
            second_count=120,
            counter_semantics=monotonic,
        )
        legacy_reset = verify_timeline(
            legacy_reset_path,
            require_final=False,
        )
        committed_reset = verify_timeline(
            committed_reset_path,
            require_final=False,
        )
        committed_growth = verify_timeline(
            committed_growth_path,
            require_final=False,
        )

    checks = {
        "legacy_scan_reset_is_disclosed_and_accepted": bool(
            legacy_reset["status"] == "passed"
            and legacy_reset["data_continuity"][
                "feature_pass_reset_events"
            ]
            == 1
        ),
        "committed_counter_decrease_is_rejected": bool(
            committed_reset["status"] == "failed"
            and committed_reset["data_continuity"][
                "committed_feature_counter_decrease_events"
            ]
            == 1
            and any(
                "committed feature counter decreased" in error
                for error in committed_reset["errors"]
            )
        ),
        "committed_counter_growth_is_accepted": bool(
            committed_growth["status"] == "passed"
            and committed_growth["data_continuity"][
                "committed_feature_counter_decrease_events"
            ]
            == 0
        ),
    }
    payload = {
        "generated_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "control": "droid_timeline_monotonic_feature_counter_v1",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "claim_boundary": (
            "This proves verifier behavior on controlled hash-chained "
            "timelines. The live timeline is verified separately."
        ),
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
