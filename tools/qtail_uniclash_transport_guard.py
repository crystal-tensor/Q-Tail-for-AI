#!/usr/bin/env python3
"""Continuously prove DROID transfers bypass the running UniClash core."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote


LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DROID_SOURCE_TOKEN = "gresearch/robotics/droid"
DROID_OBJECT_PATH_TOKEN = "robotics/droid"
PROCESS_CLASSIFIER_VERSION = (
    "droid_transport_downloader_descendants_v6_interface_bound_live"
)
GOOGLE_STORAGE_HOSTS = {
    "storage.googleapis.com",
    "*.storage.googleapis.com",
    "*.googleapis.com",
}
UNICLASH_APP_NAME = "UniClash"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def process_rows() -> list[tuple[int, int, str]]:
    result = run(["/bin/ps", "-axo", "pid=,ppid=,command="])
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if match:
            rows.append(
                (int(match.group(1)), int(match.group(2)), match.group(3))
            )
    return rows


def uniclash_state(rows: list[tuple[int, int, str]]) -> dict[str, Any]:
    core_pids = [
        pid
        for pid, _, command in rows
        if re.search(r"/UniClashCore(?:\s|$)", command)
    ]
    app_pids = [
        pid
        for pid, _, command in rows
        if re.search(r"/UniClash(?:\s|$)", command)
        and "UniClashCore" not in command
    ]
    config_result = run(
        ["/usr/bin/defaults", "read", "com.follow.clash", "flutter.config"]
    )
    tun_enabled: bool | None = None
    system_proxy: bool | None = None
    allow_bypass: bool | None = None
    if config_result.returncode == 0:
        try:
            config = json.loads(config_result.stdout)
            tun_enabled = bool(
                config.get("patchClashConfig", {})
                .get("tun", {})
                .get("enable")
            )
            system_proxy = bool(
                config.get("networkProps", {}).get("systemProxy")
            )
            allow_bypass = bool(
                config.get("vpnProps", {}).get("allowBypass")
            )
        except json.JSONDecodeError:
            pass
    return {
        "app_running": bool(app_pids),
        "app_pids": app_pids,
        "core_running": bool(core_pids),
        "core_pids": core_pids,
        "tun_enabled": tun_enabled,
        "system_proxy_enabled": system_proxy,
        "allow_bypass": allow_bypass,
    }


def ensure_uniclash_core(
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    before = uniclash_state(process_rows())
    result: dict[str, Any] = {
        "enabled": True,
        "attempted": False,
        "open_returncode": None,
        "core_running_before": before["core_running"],
        "core_running_after": before["core_running"],
        "waited_seconds": 0.0,
    }
    if before["core_running"]:
        return result

    result["attempted"] = True
    opened = run(["/usr/bin/open", "-g", "-a", UNICLASH_APP_NAME])
    result["open_returncode"] = opened.returncode
    if opened.stderr.strip():
        result["open_stderr"] = opened.stderr.strip()

    started = time.monotonic()
    deadline = started + max(0.5, timeout_seconds)
    while time.monotonic() < deadline:
        current = uniclash_state(process_rows())
        if current["core_running"]:
            result["core_running_after"] = True
            break
        time.sleep(0.5)
    result["waited_seconds"] = round(time.monotonic() - started, 3)
    return result


def proxy_bypass_state(service: str) -> dict[str, Any]:
    result = run(
        ["/usr/sbin/networksetup", "-getproxybypassdomains", service]
    )
    domains = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("There aren't")
    }
    flattened: set[str] = set()
    for domain in domains:
        flattened.update(item.strip() for item in domain.split(","))
    return {
        "service": service,
        "domains": sorted(flattened),
        "required_domains": sorted(GOOGLE_STORAGE_HOSTS),
        "passed": GOOGLE_STORAGE_HOSTS.issubset(flattened),
    }


def ensure_proxy_bypass(service: str) -> dict[str, Any]:
    state = proxy_bypass_state(service)
    state["repair_attempted"] = False
    state["repair_returncode"] = None
    state["repaired"] = False
    if state["passed"]:
        return state
    state["repair_attempted"] = True
    required = sorted(set(state["domains"]) | GOOGLE_STORAGE_HOSTS)
    repair = run(
        [
            "/usr/sbin/networksetup",
            "-setproxybypassdomains",
            service,
            *required,
        ]
    )
    refreshed = proxy_bypass_state(service)
    refreshed.update(
        {
            "repair_attempted": True,
            "repair_returncode": repair.returncode,
            "repair_stderr": repair.stderr.strip()[:500],
            "repaired": bool(repair.returncode == 0 and refreshed["passed"]),
        }
    )
    return refreshed


def socket_rows_by_pid(
    pids: list[int],
) -> dict[int, list[dict[str, Any]]]:
    if not pids:
        return {}
    result = run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            ",".join(str(pid) for pid in pids),
            "-iTCP",
            "-sTCP:ESTABLISHED",
        ]
    )
    sockets: dict[int, list[dict[str, Any]]] = {
        pid: [] for pid in pids
    }
    for line in result.stdout.splitlines()[1:]:
        pid_match = re.match(r"^\S+\s+(\d+)\s+", line)
        match = re.search(r"\s(\S+):(\d+)->(\S+):(\d+)\s+\(ESTABLISHED\)", line)
        if not pid_match or not match:
            continue
        pid = int(pid_match.group(1))
        sockets.setdefault(pid, []).append(
            {
                "local_address": match.group(1),
                "local_port": int(match.group(2)),
                "remote_address": match.group(3),
                "remote_port": int(match.group(4)),
            }
        )
    return sockets


def route_for(address: str) -> dict[str, str]:
    result = run(["/sbin/route", "-n", "get", address])
    interface = re.search(
        r"(?m)^\s*interface:\s*(\S+)\s*$", result.stdout
    )
    gateway = re.search(r"(?m)^\s*gateway:\s*(\S+)\s*$", result.stdout)
    return {
        "address": address,
        "interface": interface.group(1) if interface else "",
        "gateway": gateway.group(1) if gateway else "",
    }


def process_environment_state(pid: int) -> dict[str, bool]:
    result = run(
        ["/bin/ps", "eww", "-p", str(pid), "-o", "command="]
    )
    process_text = result.stdout
    no_proxy_all = bool(
        re.search(
            r"(?:^|\s)(?:NO_PROXY|no_proxy)=\*(?:\s|$)",
            process_text,
        )
    )
    proxy_environment_present = bool(
        re.search(
            r"(?:^|\s)(?:HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|"
            r"http_proxy|https_proxy|all_proxy)=\S+",
            process_text,
        )
    )
    return {
        "no_proxy_all": no_proxy_all,
        "proxy_environment_present": proxy_environment_present,
    }


def monitored_processes(
    rows: list[tuple[int, int, str]],
) -> list[tuple[int, int, str, str, int]]:
    by_parent: dict[int, list[int]] = {}
    by_pid = {pid: (ppid, command) for pid, ppid, command in rows}
    for pid, ppid, _ in rows:
        by_parent.setdefault(ppid, []).append(pid)

    def contains_droid_source(command: str) -> bool:
        decoded = unquote(command).lower()
        return (
            DROID_SOURCE_TOKEN in decoded
            or (
                "gresearch" in decoded
                and DROID_OBJECT_PATH_TOKEN in decoded
            )
        )

    def command_tokens(command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def is_curl_process(command: str) -> bool:
        tokens = command_tokens(command)
        return bool(tokens and Path(tokens[0]).name.lower() == "curl")

    def is_downloader_process(command: str) -> bool:
        tokens = command_tokens(command)
        if len(tokens) < 2:
            return False
        executable = Path(tokens[0]).name.lower()
        script = Path(tokens[1]).name.lower()
        return (
            "python" in executable
            and script == "qtail_parallel_gcs_download.py"
        )

    monitored: dict[int, tuple[str, int]] = {}
    for pid, _, command in rows:
        if is_curl_process(command) and contains_droid_source(command):
            monitored[pid] = ("curl", pid)

    downloader_roots = [
        pid
        for pid, _, command in rows
        if is_downloader_process(command)
    ]
    for root_pid in downloader_roots:
        pending = list(by_parent.get(root_pid, []))
        while pending:
            pid = pending.pop()
            if pid not in by_pid:
                continue
            command = by_pid[pid][1]
            if is_curl_process(command):
                monitored[pid] = ("curl", root_pid)
            pending.extend(by_parent.get(pid, []))

    def is_gsutil_process(command: str) -> bool:
        tokens = command_tokens(command)
        if not tokens:
            return False
        executable = Path(tokens[0]).name.lower()
        if executable in {"gsutil", "gsutil.py"}:
            return True
        return bool(
            ("python" in executable)
            and len(tokens) > 1
            and Path(tokens[1]).name.lower() in {"gsutil", "gsutil.py"}
        )

    gsutil_roots = [
        pid
        for pid, _, command in rows
        if contains_droid_source(command) and is_gsutil_process(command)
    ]
    for root_pid in gsutil_roots:
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            if pid not in by_pid:
                continue
            monitored.setdefault(pid, ("gsutil", root_pid))
            pending.extend(by_parent.get(pid, []))

    return [
        (pid, by_pid[pid][0], by_pid[pid][1], kind, root_pid)
        for pid, (kind, root_pid) in sorted(monitored.items())
    ]


def classifier_selftest() -> dict[str, Any]:
    rows = [
        (
            100,
            1,
            (
                "/usr/bin/curl --noproxy '*' --interface en1 "
                "https://storage-download.googleapis.com/gresearch/"
                "robotics/droid/1.0.0/a"
            ),
        ),
        (
            101,
            1,
            (
                "/usr/bin/curl --noproxy '*' --interface en1 "
                "https://storage.googleapis.com/download/storage/v1/b/"
                "gresearch/o/robotics%2Fdroid%2F1.0.0%2Fa?alt=media"
            ),
        ),
        (
            102,
            1,
            (
                "/usr/bin/curl --noproxy '*' --interface en1 "
                "https://example.com/unrelated"
            ),
        ),
        (
            103,
            1,
            (
                "/bin/zsh -lc '/usr/bin/curl "
                "https://storage.googleapis.com/gresearch/robotics/droid/a'"
            ),
        ),
        (
            200,
            1,
            (
                "/usr/bin/python3 /repo/tools/"
                "qtail_parallel_gcs_download.py --manifest manifest.json"
            ),
        ),
        (
            201,
            200,
            (
                "/usr/bin/curl --noproxy '*' --interface en1 "
                "https://opaque.example/media"
            ),
        ),
        (
            300,
            1,
            "/env/bin/gsutil rsync -r gs://gresearch/robotics/droid /data",
        ),
        (301, 300, "/usr/bin/python3 gsutil_worker.py"),
    ]
    observed = monitored_processes(rows)
    by_pid = {
        pid: {"kind": kind, "root_pid": root_pid}
        for pid, _, _, kind, root_pid in observed
    }
    checks = {
        "plain_droid_curl_included": by_pid.get(100)
        == {"kind": "curl", "root_pid": 100},
        "encoded_droid_curl_included": by_pid.get(101)
        == {"kind": "curl", "root_pid": 101},
        "unrelated_curl_excluded": 102 not in by_pid,
        "shell_wrapper_excluded": 103 not in by_pid,
        "opaque_downloader_child_included": by_pid.get(201)
        == {"kind": "curl", "root_pid": 200},
        "gsutil_root_included": by_pid.get(300)
        == {"kind": "gsutil", "root_pid": 300},
        "gsutil_descendant_included": by_pid.get(301)
        == {"kind": "gsutil", "root_pid": 300},
    }
    return {
        "generated_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "classifier_version": PROCESS_CLASSIFIER_VERSION,
        "checks": checks,
        "observed": by_pid,
    }


def update_cumulative(
    previous: dict[str, Any],
    payload: dict[str, Any],
    expected_interface: str,
) -> dict[str, Any]:
    old = previous.get("cumulative", {})
    if not isinstance(old, dict):
        old = {}
    transfers = payload.get("transfers", [])
    blocked = bool(
        payload.get("global_violations")
        or payload.get("blocked_processes")
    )
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
    forbidden_sockets = sum(
        socket.get("remote_address") in LOOPBACK
        or socket.get("remote_port") == 7993
        for socket in sockets
    )
    wrong_routes = sum(
        route.get("interface") != expected_interface for route in routes
    )
    events = old.get("violation_events", [])
    if not isinstance(events, list):
        events = []
    if blocked:
        events.append(
            {
                "at": payload["generated_at"],
                "global_violations": payload.get("global_violations", []),
                "blocked_processes": payload.get("blocked_processes", []),
                "transfer_violations": [
                    {
                        "pid": transfer.get("pid"),
                        "kind": transfer.get("transport_kind"),
                        "violations": transfer.get("violations", []),
                    }
                    for transfer in transfers
                    if transfer.get("violations")
                ],
            }
        )
    old_blocked_pids = old.get("blocked_pids", [])
    blocked_pids = sorted(
        {
            int(pid)
            for pid in [*old_blocked_pids, *payload.get("blocked_processes", [])]
        }
    )
    return {
        "first_sample_at": old.get(
            "first_sample_at", payload["generated_at"]
        ),
        "last_sample_at": payload["generated_at"],
        "samples": int(old.get("samples", 0)) + 1,
        "passed_samples": int(old.get("passed_samples", 0))
        + int(not blocked),
        "blocked_samples": int(old.get("blocked_samples", 0))
        + int(blocked),
        "observed_process_samples": int(
            old.get("observed_process_samples", 0)
        )
        + len(transfers),
        "curl_process_samples": int(old.get("curl_process_samples", 0))
        + sum(
            transfer.get("transport_kind") == "curl"
            for transfer in transfers
        ),
        "gsutil_process_samples": int(
            old.get("gsutil_process_samples", 0)
        )
        + sum(
            transfer.get("transport_kind") == "gsutil"
            for transfer in transfers
        ),
        "observed_socket_samples": int(
            old.get("observed_socket_samples", 0)
        )
        + len(sockets),
        "direct_route_socket_samples": int(
            old.get("direct_route_socket_samples", 0)
        )
        + sum(
            route.get("interface") == expected_interface for route in routes
        ),
        "forbidden_socket_observations": int(
            old.get("forbidden_socket_observations", 0)
        )
        + forbidden_sockets,
        "wrong_route_observations": int(
            old.get("wrong_route_observations", 0)
        )
        + wrong_routes,
        "blocked_pids": blocked_pids,
        "last_violation_at": (
            payload["generated_at"]
            if blocked
            else old.get("last_violation_at")
        ),
        "violation_events": events[-100:],
    }


def audit(
    expected_interface: str,
    network_service: str,
    previous: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    rows = process_rows()
    uniclash = uniclash_state(rows)
    bypass = ensure_proxy_bypass(network_service)
    monitored_rows = monitored_processes(rows)
    sockets_by_pid = socket_rows_by_pid(
        [pid for pid, _, _, _, _ in monitored_rows]
    )
    gsutil_root_environments = {
        root_pid: process_environment_state(root_pid)
        for _, _, _, transport_kind, root_pid in monitored_rows
        if transport_kind == "gsutil"
    }
    transfers: list[dict[str, Any]] = []
    violating_pids: list[int] = []
    global_violations: list[str] = []
    if not uniclash["core_running"]:
        global_violations.append("UniClashCore is not running")
    if uniclash["tun_enabled"] is not False:
        global_violations.append(
            f"UniClash TUN must be false, got {uniclash['tun_enabled']}"
        )
    if not bypass["passed"]:
        global_violations.append("Google Storage proxy bypass is incomplete")

    for pid, ppid, command, transport_kind, root_pid in monitored_rows:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        environment = (
            gsutil_root_environments[root_pid]
            if transport_kind == "gsutil"
            else process_environment_state(pid)
        )
        direct_flag = (
            any(
                tokens[index] == "--noproxy"
                and index + 1 < len(tokens)
                and tokens[index + 1] == "*"
                for index in range(len(tokens))
            )
            if transport_kind == "curl"
            else (
                environment["no_proxy_all"]
                and not environment["proxy_environment_present"]
            )
        )
        explicit_proxy = "--proxy" in tokens or "-x" in tokens
        bound_interface = ""
        if transport_kind == "curl":
            for index, token in enumerate(tokens):
                if token == "--interface" and index + 1 < len(tokens):
                    bound_interface = tokens[index + 1]
                    break
        sockets = sockets_by_pid.get(pid, [])
        routes = [
            route_for(item["remote_address"])
            for item in sockets
            if item["remote_address"] not in LOOPBACK
        ]
        violations: list[str] = []
        if not direct_flag:
            violations.append(
                "missing --noproxy '*'"
                if transport_kind == "curl"
                else "gsutil process lacks clean NO_PROXY=* environment"
            )
        if explicit_proxy:
            violations.append("explicit proxy option present")
        if (
            transport_kind == "curl"
            and bound_interface != expected_interface
        ):
            violations.append(
                "curl physical interface binding differs from "
                f"{expected_interface}: {bound_interface or 'missing'}"
            )
        if any(
            item["remote_address"] in LOOPBACK
            or item["remote_port"] == 7993
            for item in sockets
        ):
            violations.append("socket reached UniClash loopback/7993")
        if any(
            route["interface"] != expected_interface for route in routes
        ):
            violations.append(
                f"route interface differs from {expected_interface}"
            )
        if global_violations:
            violations.extend(global_violations)
        if violations:
            violating_pids.append(pid)
        transfers.append(
            {
                "pid": pid,
                "parent_pid": ppid,
                "root_pid": root_pid,
                "transport_kind": transport_kind,
                "direct_flag": direct_flag,
                "explicit_proxy": explicit_proxy,
                "bound_interface": (
                    bound_interface if transport_kind == "curl" else None
                ),
                "no_proxy_all": (
                    direct_flag
                    if transport_kind == "curl"
                    else environment["no_proxy_all"]
                ),
                "curl_no_proxy_flag": (
                    direct_flag if transport_kind == "curl" else None
                ),
                "no_proxy_environment_all": environment["no_proxy_all"],
                "proxy_environment_present": environment[
                    "proxy_environment_present"
                ],
                "sockets": sockets,
                "routes": routes,
                "violations": violations,
            }
        )

    status = "passed"
    if global_violations or violating_pids:
        status = "blocked"
    elif not monitored_rows:
        status = "passed_idle"
    payload = {
        "generated_at": now(),
        "status": status,
        "policy": {
            "download_must_continue": True,
            "uniclash_core_must_continue": True,
            "droid_must_bypass_uniclash": True,
            "guarded_transports": ["curl", "gsutil"],
            "process_classifier_version": PROCESS_CLASSIFIER_VERSION,
            "curl_classification": [
                "decoded official DROID URL",
                "descendant of qtail_parallel_gcs_download.py",
            ],
            "curl_transport_contract": [
                "--noproxy '*'",
                f"--interface {expected_interface}",
            ],
            "expected_interface": expected_interface,
            "forbidden_proxy_port": 7993,
        },
        "uniclash": uniclash,
        "system_proxy_bypass": bypass,
        "active_droid_transfers": len(transfers),
        "transfers": transfers,
        "global_violations": global_violations,
        "blocked_processes": violating_pids,
    }
    previous_payload = previous or {}
    previous_classifier = previous_payload.get("policy", {}).get(
        "process_classifier_version"
    )
    epoch_previous = (
        previous_payload
        if previous_classifier == PROCESS_CLASSIFIER_VERSION
        else {}
    )
    previous_epoch = previous_payload.get("epoch", {})
    reset_from_classifier = (
        previous_epoch.get("reset_from_classifier_version")
        if previous_classifier == PROCESS_CLASSIFIER_VERSION
        and isinstance(previous_epoch, dict)
        else (
            previous_classifier
            if previous_classifier
            and previous_classifier != PROCESS_CLASSIFIER_VERSION
            else None
        )
    )
    payload["epoch"] = {
        "classifier_version": PROCESS_CLASSIFIER_VERSION,
        "reset_from_classifier_version": reset_from_classifier,
    }
    payload["cumulative"] = update_cumulative(
        epoch_previous,
        payload,
        expected_interface,
    )
    return payload, violating_pids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--expected-interface", default="en1")
    parser.add_argument("--network-service", default="Wi-Fi")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-terminate", action="store_true")
    parser.add_argument("--no-restart-uniclash", action="store_true")
    parser.add_argument("--classifier-selftest-out", type=Path)
    args = parser.parse_args()

    if args.classifier_selftest_out:
        payload = classifier_selftest()
        atomic_write_json(args.classifier_selftest_out, payload)
        print(json.dumps(payload, indent=2))
        if payload["status"] != "passed":
            raise SystemExit(1)
        return
    if not args.status:
        parser.error("--status is required unless --classifier-selftest-out is used")

    while True:
        keepalive = (
            {
                "enabled": False,
                "attempted": False,
            }
            if args.no_restart_uniclash
            else ensure_uniclash_core()
        )
        payload, violating_pids = audit(
            args.expected_interface,
            args.network_service,
            read_json(args.status),
        )
        payload["uniclash_keepalive"] = keepalive
        if not args.no_terminate:
            for pid in violating_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        atomic_write_json(args.status, payload)
        if args.once:
            break
        time.sleep(max(0.5, args.interval_seconds))


if __name__ == "__main__":
    main()
