#!/usr/bin/env python3
"""Require a fresh, clean UniClash bypass audit before DROID gsutil work."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GATE_VERSION = "qtail_uniclash_pre_checksum_gate_v2"
ADJUDICATED_IDLE_POLICY_VIOLATIONS = frozenset(
    {"UniClashCore is not running"}
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_guard(
    payload: dict[str, Any],
    *,
    expected_interface: str,
    max_age_seconds: float,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    reference = reference_time or datetime.now(timezone.utc)
    generated_at = parse_timestamp(payload.get("generated_at"))
    age_seconds = (
        max(0.0, (reference - generated_at).total_seconds())
        if generated_at is not None
        else None
    )
    policy = payload.get("policy", {})
    uniclash = payload.get("uniclash", {})
    bypass = payload.get("system_proxy_bypass", {})
    cumulative = payload.get("cumulative", {})
    cumulative_events = cumulative.get("violation_events", [])
    if not isinstance(cumulative_events, list):
        cumulative_events = []
    transports = policy.get("guarded_transports", [])
    global_violations = payload.get("global_violations", [])
    transfers = payload.get("transfers", [])

    live_route_interfaces = sorted(
        {
            str(route.get("interface"))
            for transfer in transfers
            if isinstance(transfer, dict)
            for route in transfer.get("routes", [])
            if isinstance(route, dict) and route.get("interface")
        }
    )
    transfer_violations = [
        {
            "pid": transfer.get("pid"),
            "violations": transfer.get("violations"),
        }
        for transfer in transfers
        if isinstance(transfer, dict) and transfer.get("violations")
    ]
    adjudicated_policy_pause_events = [
        event
        for event in cumulative_events
        if isinstance(event, dict)
        and bool(event.get("global_violations"))
        and set(event.get("global_violations", [])).issubset(
            ADJUDICATED_IDLE_POLICY_VIOLATIONS
        )
        and not event.get("blocked_processes")
        and not event.get("transfer_violations")
    ]
    unadjudicated_cumulative_events = [
        event
        for event in cumulative_events
        if event not in adjudicated_policy_pause_events
    ]
    blocked_samples = int(cumulative.get("blocked_samples", -1))
    checks = {
        "guard_status_passed": payload.get("status")
        in {"passed", "passed_idle"},
        "guard_heartbeat_fresh": (
            age_seconds is not None and age_seconds <= max_age_seconds
        ),
        "uniclash_core_running": uniclash.get("core_running") is True,
        "uniclash_tun_disabled": uniclash.get("tun_enabled") is False,
        "droid_bypass_policy_enabled": (
            policy.get("uniclash_core_must_continue") is True
            and policy.get("droid_must_bypass_uniclash") is True
        ),
        "curl_and_gsutil_guarded": (
            isinstance(transports, list)
            and {"curl", "gsutil"}.issubset(set(transports))
        ),
        "expected_interface_bound": (
            policy.get("expected_interface") == expected_interface
        ),
        "system_proxy_bypass_passed": (
            bypass.get("passed") is True
            and "storage.googleapis.com"
            in set(bypass.get("required_domains", []))
        ),
        "cumulative_history_clean": (
            blocked_samples == len(cumulative_events)
            and not unadjudicated_cumulative_events
            and int(
                cumulative.get("forbidden_socket_observations", -1)
            )
            == 0
            and int(cumulative.get("wrong_route_observations", -1)) == 0
            and not cumulative.get("blocked_pids")
        ),
        "live_transfers_clean_and_direct": (
            not global_violations
            and not transfer_violations
            and all(
                interface == expected_interface
                for interface in live_route_interfaces
            )
        ),
    }
    passed = all(checks.values())
    return {
        "generated_at": now(),
        "version": GATE_VERSION,
        "status": "passed" if passed else "blocked",
        "purpose": (
            "Immediate preflight before DROID gsutil checksum traffic. "
            "The continuous transport guard remains authoritative while "
            "gsutil is running."
        ),
        "expected_interface": expected_interface,
        "guard_generated_at": payload.get("generated_at"),
        "guard_age_seconds": age_seconds,
        "max_guard_age_seconds": max_age_seconds,
        "checks": checks,
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "live_route_interfaces": live_route_interfaces,
        "global_violations": global_violations,
        "transfer_violations": transfer_violations,
        "adjudicated_idle_policy_pause_count": len(
            adjudicated_policy_pause_events
        ),
        "unadjudicated_cumulative_violation_count": len(
            unadjudicated_cumulative_events
        ),
        "uniclash_core_pids": uniclash.get("core_pids", []),
        "claim_boundary": (
            "This is a point-in-time launch gate. Continuous route/socket "
            "observation and final timeline verification are still required. "
            "A historical UniClashCore-off heartbeat is accepted only when "
            "the same event has no DROID process or transfer violation and "
            "the cumulative history has zero forbidden sockets, wrong routes, "
            "or blocked PIDs; the event remains disclosed as an idle policy "
            "pause."
        ),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-interface", default="en1")
    parser.add_argument("--max-age-seconds", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        payload = json.loads(args.guard.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result = {
            "generated_at": now(),
            "version": GATE_VERSION,
            "status": "blocked",
            "checks": {},
            "checks_passed": 0,
            "checks_total": 10,
            "error": str(error),
        }
    else:
        if not isinstance(payload, dict):
            payload = {}
        result = validate_guard(
            payload,
            expected_interface=args.expected_interface,
            max_age_seconds=args.max_age_seconds,
        )

    if args.out:
        atomic_write_json(args.out, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
