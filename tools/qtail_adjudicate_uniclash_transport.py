#!/usr/bin/env python3
"""Bind UniClash guard epochs to preserved raw route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


V3 = "droid_transport_root_environment_v3"
V4 = "droid_transport_downloader_descendants_v4"
V5 = "droid_transport_downloader_descendants_v5_interface_bound"
V6 = "droid_transport_downloader_descendants_v6_interface_bound_live"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def no_network_violation(payload: dict[str, Any]) -> bool:
    cumulative = payload.get("cumulative", {})
    return bool(
        int(cumulative.get("forbidden_socket_observations", -1)) == 0
        and int(cumulative.get("wrong_route_observations", -1)) == 0
    )


def archive_row(
    path: Path,
    payload: dict[str, Any],
    *,
    coverage_gap: bool,
    policy_pause: bool,
) -> dict[str, Any]:
    cumulative = payload.get("cumulative", {})
    return {
        "path": str(path),
        "sha256": sha256(path),
        "classifier_version": payload.get("policy", {}).get(
            "process_classifier_version"
        ),
        "raw_blocked_samples": int(cumulative.get("blocked_samples", 0)),
        "raw_forbidden_socket_observations": int(
            cumulative.get("forbidden_socket_observations", 0)
        ),
        "raw_wrong_route_observations": int(
            cumulative.get("wrong_route_observations", 0)
        ),
        "coverage_gap": coverage_gap,
        "policy_pause": policy_pause,
        "data_transfer_violation": not no_network_violation(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-root",
        type=Path,
        default=Path("/Volumes/ORICO/qtail_full_training"),
    )
    parser.add_argument(
        "--live-guard",
        type=Path,
        default=Path(
            "/Users/avalok/work/Q-TAIL-MVP/.tmp/"
            "qtail-uniclash-transport-guard.json"
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result_root = args.job_root / "results" / "qtail_droid_full"
    epoch_root = result_root / "transport_epochs"
    paths = {
        "v1": (
            result_root
            / "uniclash_transport_guard_v1_classifier_false_positive.json"
        ),
        "v2": (
            result_root
            / "uniclash_transport_guard_v2_descendant_environment_false_positive.json"
        ),
        "v3": (
            result_root
            / "uniclash_transport_guard_v3_encoded_path_underobservation.json"
        ),
        "v4": epoch_root / "uniclash_transport_guard_v4_core_restart_pause.json",
        "v5": (
            epoch_root
            / "uniclash_transport_guard_v5_interface_migration_pause.json"
        ),
        "selftest": (
            result_root
            / "uniclash_transport_guard_classifier_v6_selftest.json"
        ),
        "parallel": result_root / "parallel_download_status.json",
    }
    out = args.out or (
        result_root / "uniclash_transport_guard_adjudication.json"
    )
    evidence = {name: read_json(path) for name, path in paths.items()}
    live = read_json(args.live_guard)
    transfers = live.get("transfers", [])
    routes = [
        route
        for transfer in transfers
        for route in transfer.get("routes", [])
    ]
    sockets = [
        socket
        for transfer in transfers
        for socket in transfer.get("sockets", [])
    ]
    expected_active = len(evidence["parallel"].get("active", []))
    v4_events = evidence["v4"].get("cumulative", {}).get(
        "violation_events", []
    )
    v5_events = evidence["v5"].get("cumulative", {}).get(
        "violation_events", []
    )
    checks = {
        "v1_raw_false_positive_preserved": int(
            evidence["v1"].get("cumulative", {}).get("blocked_samples", 0)
        )
        > 0,
        "v2_raw_false_positive_preserved": int(
            evidence["v2"].get("cumulative", {}).get("blocked_samples", 0)
        )
        > 0,
        "v3_coverage_gap_preserved": (
            evidence["v3"].get("policy", {}).get(
                "process_classifier_version"
            )
            == V3
            and no_network_violation(evidence["v3"])
        ),
        "v4_core_restart_pause_preserved": (
            evidence["v4"].get("policy", {}).get(
                "process_classifier_version"
            )
            == V4
            and no_network_violation(evidence["v4"])
            and any(
                "UniClashCore is not running"
                in event.get("global_violations", [])
                for event in v4_events
            )
        ),
        "v5_interface_migration_pause_preserved": (
            evidence["v5"].get("policy", {}).get(
                "process_classifier_version"
            )
            == V5
            and no_network_violation(evidence["v5"])
            and any(
                any(
                    "physical interface binding differs"
                    in violation
                    for transfer in event.get("transfer_violations", [])
                    for violation in transfer.get("violations", [])
                )
                for event in v5_events
            )
        ),
        "v6_classifier_selftest_passed": (
            evidence["selftest"].get("status") == "passed"
            and evidence["selftest"].get("classifier_version") == V6
            and all(evidence["selftest"].get("checks", {}).values())
        ),
        "v6_live_guard_passed_and_clean": (
            live.get("status") == "passed"
            and live.get("policy", {}).get("process_classifier_version")
            == V6
            and int(
                live.get("cumulative", {}).get("blocked_samples", -1)
            )
            == 0
            and no_network_violation(live)
        ),
        "v6_covers_every_active_worker": (
            expected_active > 0
            and int(live.get("active_droid_transfers", -1))
            == expected_active
        ),
        "v6_every_curl_is_interface_bound": bool(transfers)
        and all(
            transfer.get("direct_flag") is True
            and transfer.get("explicit_proxy") is False
            and transfer.get("bound_interface") == "en1"
            and not transfer.get("violations", [])
            for transfer in transfers
            if transfer.get("transport_kind") == "curl"
        ),
        "v6_every_socket_routes_direct": (
            len(sockets) == len(transfers)
            and len(routes) == len(transfers)
            and all(route.get("interface") == "en1" for route in routes)
            and all(
                socket.get("remote_address")
                not in {"127.0.0.1", "::1", "localhost"}
                and int(socket.get("remote_port", -1)) != 7993
                for socket in sockets
            )
        ),
        "uniclash_core_online_tun_off": (
            live.get("uniclash", {}).get("core_running") is True
            and live.get("uniclash", {}).get("tun_enabled") is False
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    archive_specs = [
        ("v1", False, False),
        ("v2", False, False),
        ("v3", True, False),
        ("v4", False, True),
        ("v5", False, True),
    ]
    archives = [
        archive_row(
            paths[name],
            evidence[name],
            coverage_gap=coverage_gap,
            policy_pause=policy_pause,
        )
        for name, coverage_gap, policy_pause in archive_specs
    ]
    archive_hashes_actual = {
        row["path"]: sha256(Path(row["path"])) for row in archives
    }
    payload = {
        "generated_at": now(),
        "status": (
            "adjudicated_transport_epochs_v6" if not failed else "failed"
        ),
        "claim_boundary": (
            "This adjudicates process-classifier coverage, conservative "
            "policy pauses, curl interface binding, and observed socket "
            "routes. It does not prove mirror completeness or model quality."
        ),
        "findings": [
            {
                "guard_epoch": "classifier_v1",
                "classification": "shell-launcher classifier false positive",
                "coverage_gap": False,
                "policy_pause": True,
                "data_transfer_violation": False,
            },
            {
                "guard_epoch": "droid_transport_executable_v2",
                "classification": "descendant environment false positive",
                "coverage_gap": False,
                "policy_pause": True,
                "data_transfer_violation": False,
            },
            {
                "guard_epoch": V3,
                "classification": "encoded JSON media URL under-observation",
                "coverage_gap": True,
                "policy_pause": False,
                "data_transfer_violation": False,
            },
            {
                "guard_epoch": V4,
                "classification": (
                    "UniClashCore restart caused fail-closed termination; all "
                    "observed sockets still routed directly over en1"
                ),
                "coverage_gap": False,
                "policy_pause": True,
                "data_transfer_violation": False,
            },
            {
                "guard_epoch": V5,
                "classification": (
                    "interface-binding migration rejected old downloader "
                    "children before the resumable v6 process generation"
                ),
                "coverage_gap": False,
                "policy_pause": True,
                "data_transfer_violation": False,
            },
        ],
        "remediation": {
            "classifier_version": V6,
            "curl_contract": ["--noproxy '*'", "--interface en1"],
            "route_contract": "every observed remote route must equal en1",
            "selftest": str(paths["selftest"]),
            "selftest_sha256": sha256(paths["selftest"]),
        },
        "live_v6_evidence": {
            "guard": str(args.live_guard),
            "guard_sha256": sha256(args.live_guard),
            "parallel_download_status": str(paths["parallel"]),
            "parallel_download_status_sha256": sha256(paths["parallel"]),
            "expected_active_workers": expected_active,
            "observed_transfers": len(transfers),
            "observed_sockets": len(sockets),
            "observed_routes": len(routes),
            "bound_interfaces": sorted(
                {
                    str(item.get("bound_interface"))
                    for item in transfers
                    if item.get("bound_interface")
                }
            ),
            "route_interfaces": sorted(
                {str(route.get("interface")) for route in routes}
            ),
        },
        "preservation": {"archives": archives},
        "archive_hashes_actual": archive_hashes_actual,
        "checks": checks,
        "failed_checks": failed,
    }
    atomic_write_json(out, payload)
    print(json.dumps({"out": str(out), "status": payload["status"]}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
