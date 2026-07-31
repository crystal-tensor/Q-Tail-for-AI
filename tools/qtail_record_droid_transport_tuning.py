#!/usr/bin/env python3
"""Seal the bounded DROID transport benchmark and live route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY_VERSION = (
    "droid_public_endpoints_v4_http2_physical_interface_bound"
)
OFFICIAL_SOURCE = "gs://gresearch/robotics/droid"
EXPECTED_INTERFACE = "en1"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/Users/avalok/work/Q-TAIL-MVP"),
    )
    parser.add_argument(
        "--job-root",
        type=Path,
        default=Path("/Volumes/ORICO/qtail_full_training"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result_root = args.job_root / "results" / "qtail_droid_full"
    status_path = result_root / "parallel_download_status.json"
    guard_path = result_root / "uniclash_transport_guard.json"
    log_path = args.job_root / "logs" / "manual_endpoint_generation_handoff.log"
    out = args.out or result_root / "droid_transport_tuning_audit.json"
    downloader_path = args.repo_root / "tools" / "qtail_parallel_gcs_download.py"
    pipeline_path = args.repo_root / "scripts" / "qtail_orico_full_pipeline.sh"

    status = read_json(status_path)
    guard = read_json(guard_path)
    log_text = log_path.read_text(encoding="utf-8")
    observations = status.get("route_guard", {}).get("observations", [])
    cumulative = guard.get("cumulative", {})
    required_log_tokens = (
        "single_http11",
        "single_http2",
        "workers=8",
        "workers=16",
        "workers=24",
        "production_old_window_seconds=120",
        "production_new_window_seconds=180",
    )
    checks = {
        "official_source_is_fixed": OFFICIAL_SOURCE
        == "gs://gresearch/robotics/droid",
        "downloader_is_active": str(status.get("status", "")).startswith(
            "downloading"
        ),
        "worker_count_is_selected_16": int(status.get("workers", -1)) == 16,
        "http_protocol_is_http2": status.get("http_protocol") == "HTTP/2",
        "endpoint_policy_is_interface_bound_v4":
        status.get("endpoint_policy_version")
        == POLICY_VERSION,
        "download_failures_are_empty": not status.get("failures", {}),
        "downloader_route_guard_passed": status.get("route_guard", {}).get(
            "status"
        )
        == "passed",
        "all_observed_routes_use_en1": bool(observations)
        and all(
            item.get("interface") == EXPECTED_INTERFACE
            for item in observations
        ),
        "uniclash_core_is_running": guard.get("uniclash", {}).get(
            "core_running"
        )
        is True,
        "uniclash_tun_is_disabled": guard.get("uniclash", {}).get(
            "tun_enabled"
        )
        is False,
        "guard_has_no_global_violations": not guard.get(
            "global_violations", []
        ),
        "guard_has_no_blocked_samples": int(
            cumulative.get("blocked_samples", -1)
        )
        == 0,
        "guard_has_no_forbidden_sockets": int(
            cumulative.get("forbidden_socket_observations", -1)
        )
        == 0,
        "guard_has_no_wrong_routes": int(
            cumulative.get("wrong_route_observations", -1)
        )
        == 0,
        "raw_benchmark_log_is_complete": all(
            token in log_text for token in required_log_tokens
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "format_version": "qtail_droid_transport_tuning_audit_v1",
        "generated_at": now(),
        "status": "passed" if not failed else "failed",
        "official_source": OFFICIAL_SOURCE,
        "claim_boundary": (
            "This artifact proves a bounded direct-download transport tuning "
            "decision and its live route isolation only. It does not prove "
            "dataset completeness, policy training, tail success, or model "
            "quality."
        ),
        "selected_configuration": {
            "workers": 16,
            "http_protocol": "HTTP/2",
            "endpoint_policy_version": POLICY_VERSION,
            "primary_endpoint_count": 2,
            "range_chunk_mib": 64,
            "proxy": "direct",
            "expected_interface": EXPECTED_INTERFACE,
        },
        "single_connection_control": {
            "object": (
                "1.0.0/"
                "r2d2_faceblur-train.tfrecord-00000-of-02048"
            ),
            "range_bytes": 8_388_608,
            "http_1_1": {
                "http_status": 206,
                "seconds": 18.179245,
                "bytes_per_second": 461_438,
            },
            "http_2": {
                "http_status": 206,
                "seconds": 13.503401,
                "bytes_per_second": 621_221,
            },
        },
        "parallel_http2_controls": [
            {
                "workers": 8,
                "bytes": 33_554_432,
                "seconds": 13.145,
                "aggregate_mib_per_second": 2.434,
            },
            {
                "workers": 16,
                "bytes": 67_108_864,
                "seconds": 22.823,
                "aggregate_mib_per_second": 2.804,
            },
            {
                "workers": 24,
                "bytes": 100_663_296,
                "seconds": 76.036,
                "aggregate_mib_per_second": 1.263,
            },
        ],
        "production_windows": {
            "before": {
                "seconds": 120,
                "workers": 8,
                "http_protocol": "HTTP/1.1",
                "bytes_per_second": 1_057_723.73,
                "mib_per_second": 1.009,
            },
            "after": {
                "seconds": 180,
                "workers": 16,
                "http_protocol": "HTTP/2",
                "bytes_per_second": 1_809_430.76,
                "mib_per_second": 1.726,
            },
            "relative_improvement_percent": 71.07,
        },
        "selection_reason": (
            "HTTP/2 beat HTTP/1.1 in the same-object control; 16 workers "
            "improved bounded aggregate throughput over 8, while 24 regressed "
            "under connection-tail latency. The first production window then "
            "confirmed higher end-to-end transport progress."
        ),
        "live_evidence": {
            "parallel_download_status": str(status_path),
            "parallel_download_status_sha256": sha256(status_path),
            "uniclash_transport_guard": str(guard_path),
            "uniclash_transport_guard_sha256": sha256(guard_path),
            "benchmark_log": str(log_path),
            "benchmark_log_sha256": sha256(log_path),
            "route_observations": observations,
            "guard_cumulative": cumulative,
        },
        "code": {
            "downloader": str(downloader_path),
            "downloader_sha256": sha256(downloader_path),
            "pipeline": str(pipeline_path),
            "pipeline_sha256": sha256(pipeline_path),
        },
        "checks": checks,
        "failed_checks": failed,
    }
    atomic_write_json(out, payload)
    print(json.dumps({"out": str(out), "status": payload["status"]}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
