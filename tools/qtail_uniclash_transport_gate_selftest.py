#!/usr/bin/env python3
"""Destructive controls for the UniClash pre-checksum transport gate."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qtail_assert_uniclash_transport_gate import validate_guard


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def base_payload(reference: datetime) -> dict[str, Any]:
    return {
        "generated_at": reference.isoformat(),
        "status": "passed",
        "policy": {
            "uniclash_core_must_continue": True,
            "droid_must_bypass_uniclash": True,
            "guarded_transports": ["curl", "gsutil"],
            "expected_interface": "en1",
        },
        "uniclash": {
            "core_running": True,
            "core_pids": [13626],
            "tun_enabled": False,
        },
        "system_proxy_bypass": {
            "passed": True,
            "required_domains": [
                "*.googleapis.com",
                "*.storage.googleapis.com",
                "storage.googleapis.com",
            ],
        },
        "cumulative": {
            "blocked_samples": 0,
            "forbidden_socket_observations": 0,
            "wrong_route_observations": 0,
            "violation_events": [],
        },
        "global_violations": [],
        "transfers": [
            {
                "pid": 100,
                "routes": [{"interface": "en1"}],
                "violations": [],
            }
        ],
    }


def is_passed(
    payload: dict[str, Any],
    reference: datetime,
) -> bool:
    result = validate_guard(
        payload,
        expected_interface="en1",
        max_age_seconds=10.0,
        reference_time=reference,
    )
    return result["status"] == "passed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = datetime.now(timezone.utc)
    valid = base_payload(reference)
    cases: dict[str, tuple[dict[str, Any], bool]] = {
        "positive_clean_direct_guard": (valid, True),
    }

    core_off = copy.deepcopy(valid)
    core_off["uniclash"]["core_running"] = False
    cases["core_off_rejected"] = (core_off, False)

    tun_on = copy.deepcopy(valid)
    tun_on["uniclash"]["tun_enabled"] = True
    cases["tun_on_rejected"] = (tun_on, False)

    stale = copy.deepcopy(valid)
    stale["generated_at"] = (reference - timedelta(seconds=11)).isoformat()
    cases["stale_guard_rejected"] = (stale, False)

    missing_gsutil = copy.deepcopy(valid)
    missing_gsutil["policy"]["guarded_transports"] = ["curl"]
    cases["missing_gsutil_policy_rejected"] = (missing_gsutil, False)

    bypass_failed = copy.deepcopy(valid)
    bypass_failed["system_proxy_bypass"]["passed"] = False
    cases["system_bypass_failure_rejected"] = (bypass_failed, False)

    blocked_history = copy.deepcopy(valid)
    blocked_history["cumulative"]["blocked_samples"] = 1
    cases["blocked_history_rejected"] = (blocked_history, False)

    wrong_history = copy.deepcopy(valid)
    wrong_history["cumulative"]["wrong_route_observations"] = 1
    cases["wrong_route_history_rejected"] = (wrong_history, False)

    wrong_live_route = copy.deepcopy(valid)
    wrong_live_route["transfers"][0]["routes"][0]["interface"] = "utun4"
    cases["live_tunnel_route_rejected"] = (wrong_live_route, False)

    global_violation = copy.deepcopy(valid)
    global_violation["global_violations"] = ["UniClashCore is not running"]
    cases["global_violation_rejected"] = (global_violation, False)

    checks = {
        name: is_passed(payload, reference) is expected
        for name, (payload, expected) in cases.items()
    }
    passed = all(checks.values())
    report = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "checks": checks,
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "gate_checks_per_evaluation": 10,
        "claim_boundary": (
            "Synthetic controls verify gate adjudication only; the live guard "
            "artifact supplies machine-specific transport evidence."
        ),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
