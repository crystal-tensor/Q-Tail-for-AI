#!/usr/bin/env python3
"""Download a public GCS manifest with resumable, independent curl workers."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

FORBIDDEN_ROUTE_INTERFACE = re.compile(
    r"^(?:utun|tun|tap|ppp|ipsec|gif|stf|lo)\d*$",
    re.IGNORECASE,
)
PUBLIC_ENDPOINT_POLICY_VERSION = (
    "droid_public_endpoints_v4_http2_physical_interface_bound"
)
PUBLIC_HTTP_PROTOCOL = "HTTP/2"
DEFAULT_RESERVE_FREE_BYTES = 185_037_263_258


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def public_url(uri: str, endpoint_index: int) -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    object_name = quote(parsed.path.lstrip("/"), safe="/")
    encoded_object_name = quote(parsed.path.lstrip("/"), safe="")
    endpoints = (
        f"https://storage-download.googleapis.com/{bucket}/{object_name}",
        (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{bucket}/o/{encoded_object_name}?alt=media"
        ),
        f"https://storage.googleapis.com/{bucket}/{object_name}",
        f"https://{bucket}.storage.googleapis.com/{object_name}",
    )
    return endpoints[endpoint_index % len(endpoints)]


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def file_md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def content_range(path: Path) -> tuple[int, int, int] | None:
    try:
        headers = path.read_text(encoding="iso-8859-1", errors="replace")
    except OSError:
        return None
    matches = re.findall(
        r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$",
        headers,
    )
    if not matches:
        return None
    start, end, total = matches[-1]
    return int(start), int(end), int(total)


def retry_delay_seconds(base: int, consecutive_failures: int, attempts: int) -> int:
    exponential = max(1, base) * (
        2 ** min(max(0, consecutive_failures - 1), 4)
    )
    return min(60, exponential + attempts % 5)


def resolve_proxy(configured: str) -> str:
    if configured.strip().lower() in {"", "direct", "none", "off"}:
        return ""
    if configured.lower() != "auto":
        return configured
    result = subprocess.run(
        ["/usr/sbin/scutil", "--proxy"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    enabled = re.search(r"HTTPSEnable\s*:\s*1\b", result.stdout)
    host = re.search(r"HTTPSProxy\s*:\s*(\S+)", result.stdout)
    port = re.search(r"HTTPSPort\s*:\s*(\d+)", result.stdout)
    if not (enabled and host and port):
        return ""
    return f"http://{host.group(1)}:{port.group(1)}"


def direct_route_for_url(
    url: str,
    expected_interface: str | None = None,
) -> dict[str, str]:
    host = urlparse(url).hostname
    if not host:
        raise RuntimeError(f"Cannot resolve route for malformed URL: {url}")
    addresses = sorted(
        {
            item[4][0]
            for item in socket.getaddrinfo(
                host,
                443,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        }
    )
    if not addresses:
        raise RuntimeError(f"No IPv4 address resolved for direct endpoint: {host}")
    address = addresses[0]
    result = subprocess.run(
        ["/sbin/route", "-n", "get", address],
        text=True,
        capture_output=True,
        check=False,
    )
    interface_match = re.search(r"(?m)^\s*interface:\s*(\S+)\s*$", result.stdout)
    gateway_match = re.search(r"(?m)^\s*gateway:\s*(\S+)\s*$", result.stdout)
    interface = interface_match.group(1) if interface_match else ""
    gateway = gateway_match.group(1) if gateway_match else ""
    if result.returncode != 0 or not interface:
        raise RuntimeError(
            f"Cannot prove direct route for {host} ({address}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if FORBIDDEN_ROUTE_INTERFACE.match(interface):
        raise RuntimeError(
            f"VPN/tunnel route blocked for {host} ({address}): "
            f"interface={interface} gateway={gateway or 'unknown'}"
        )
    if expected_interface and interface != expected_interface:
        raise RuntimeError(
            f"Unexpected route blocked for {host} ({address}): "
            f"expected_interface={expected_interface} "
            f"interface={interface} gateway={gateway or 'unknown'}"
        )
    if address.startswith("127.") or gateway.startswith("127."):
        raise RuntimeError(
            f"Loopback proxy route blocked for {host}: "
            f"address={address} gateway={gateway or 'unknown'}"
        )
    return {
        "checked_at": now(),
        "host": host,
        "resolved_ipv4": address,
        "interface": interface,
        "gateway": gateway,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path)
    parser.add_argument("--checksum-ledger", type=Path)
    parser.add_argument("--checksum-quarantine", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--curl", type=Path, default=Path("/usr/bin/curl"))
    parser.add_argument("--retry-delay", type=int, default=30)
    parser.add_argument("--heartbeat-seconds", type=int, default=15)
    parser.add_argument("--attempt-retry-seconds", type=int, default=5)
    parser.add_argument("--stall-timeout-seconds", type=int, default=1800)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--primary-endpoints", type=int, default=2)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--forbid-tunnel-route", action="store_true")
    parser.add_argument("--expected-interface")
    parser.add_argument("--required-mount", type=Path)
    parser.add_argument("--process-lock", type=Path)
    parser.add_argument(
        "--reserve-free-bytes",
        type=int,
        default=DEFAULT_RESERVE_FREE_BYTES,
    )
    args = parser.parse_args()

    if not 1 <= args.primary_endpoints <= 4:
        raise SystemExit("--primary-endpoints must be between 1 and 4")
    if args.forbid_tunnel_route and resolve_proxy(args.proxy):
        raise SystemExit(
            "--forbid-tunnel-route requires direct/none/off proxy mode"
        )
    if args.expected_interface and not args.forbid_tunnel_route:
        raise SystemExit(
            "--expected-interface requires --forbid-tunnel-route"
        )
    if args.forbid_tunnel_route and not args.expected_interface:
        raise SystemExit(
            "--forbid-tunnel-route requires --expected-interface"
        )
    if args.required_mount and not os.path.ismount(args.required_mount):
        raise SystemExit(f"Required mount is unavailable: {args.required_mount}")
    if args.reserve_free_bytes < 0:
        raise SystemExit("--reserve-free-bytes must be non-negative")
    process_started_at = now()
    process_lock_path = args.process_lock or args.status.with_name(
        ".qtail_parallel_gcs_download.lock"
    )
    process_lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock_handle = process_lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            process_lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        process_lock_handle.seek(0)
        owner = process_lock_handle.read().strip() or "unknown owner"
        raise SystemExit(
            "DROID downloader single-writer lock is already held: "
            f"{process_lock_path} owner={owner}"
        )
    process_lock_handle.seek(0)
    process_lock_handle.truncate()
    json.dump(
        {
            "lock_version": "qtail_droid_downloader_single_writer_v1",
            "pid": os.getpid(),
            "started_at": process_started_at,
            "status": str(args.status),
            "target": str(args.target),
        },
        process_lock_handle,
        ensure_ascii=False,
    )
    process_lock_handle.write("\n")
    process_lock_handle.flush()
    os.fsync(process_lock_handle.fileno())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    objects = manifest["objects"]
    expected_total = int(manifest["total_bytes"])
    checksum_by_path: dict[str, dict[str, Any]] = {}
    checksum_ledger: dict[str, Any] = {
        "format_version": 1,
        "generated_at": now(),
        "checksum_manifest": (
            str(args.checksum_manifest)
            if args.checksum_manifest
            else None
        ),
        "objects": {},
    }
    if args.checksum_manifest:
        checksum_manifest = json.loads(
            args.checksum_manifest.read_text(encoding="utf-8")
        )
        checksum_by_path = {
            str(item["relative_path"]): item
            for item in checksum_manifest.get("objects", [])
        }
        manifest_paths = {
            str(item["relative_path"]): int(item["bytes"])
            for item in objects
        }
        checksum_paths = {
            path: int(item["bytes"])
            for path, item in checksum_by_path.items()
        }
        if (
            checksum_manifest.get("status") != "verified"
            or checksum_paths != manifest_paths
            or any(
                not item.get("md5_base64")
                for item in checksum_by_path.values()
            )
        ):
            raise SystemExit(
                "Checksum manifest does not exactly match the size manifest"
            )
        if not args.checksum_ledger:
            raise SystemExit(
                "--checksum-ledger is required with --checksum-manifest"
            )
        if not args.checksum_quarantine:
            raise SystemExit(
                "--checksum-quarantine is required with --checksum-manifest"
            )
        args.checksum_quarantine.mkdir(parents=True, exist_ok=True)
        if args.checksum_ledger.exists():
            try:
                loaded_ledger = json.loads(
                    args.checksum_ledger.read_text(encoding="utf-8")
                )
                if (
                    loaded_ledger.get("format_version") == 1
                    and isinstance(loaded_ledger.get("objects"), dict)
                ):
                    checksum_ledger = loaded_ledger
            except (OSError, json.JSONDecodeError):
                pass
    args.target.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    active: dict[str, dict[str, Any]] = {}
    route_observations: dict[str, dict[str, str]] = {}
    failures: dict[str, str] = {}
    completed_this_run = 0
    stop_heartbeat = threading.Event()
    stall_abort = threading.Event()
    last_transport_progress_bytes = -1
    last_transport_change = time.monotonic()

    def disk_headroom(requested_bytes: int = 0) -> dict[str, Any]:
        root = args.required_mount or args.target
        usage = shutil.disk_usage(root)
        required_free_bytes = args.reserve_free_bytes + max(
            0, int(requested_bytes)
        )
        return {
            "root": str(root),
            "free_bytes": usage.free,
            "reserve_free_bytes": args.reserve_free_bytes,
            "next_request_bytes": max(0, int(requested_bytes)),
            "required_free_bytes": required_free_bytes,
            "headroom_bytes": usage.free - required_free_bytes,
            "passed": usage.free >= required_free_bytes,
            "activation_boundary": (
                "Applies to this downloader process generation."
            ),
        }

    def assert_disk_headroom(requested_bytes: int) -> None:
        headroom = disk_headroom(requested_bytes)
        if not headroom["passed"]:
            raise RuntimeError(
                "ORICO reserve-free-bytes gate blocked the next Range request: "
                f"free={headroom['free_bytes']} "
                f"requested={headroom['next_request_bytes']} "
                f"reserve={headroom['reserve_free_bytes']} "
                f"headroom={headroom['headroom_bytes']}"
            )

    def ledger_entry_is_current(
        relative: str,
        target: Path,
        checksum: dict[str, Any],
    ) -> bool:
        entry = checksum_ledger.get("objects", {}).get(relative, {})
        try:
            stat = target.stat()
        except OSError:
            return False
        return (
            int(entry.get("bytes", -1)) == stat.st_size
            and int(entry.get("mtime_ns", -1)) == stat.st_mtime_ns
            and int(entry.get("ctime_ns", -1)) == stat.st_ctime_ns
            and entry.get("official_md5_base64") == checksum["md5_base64"]
            and entry.get("local_md5_base64") == checksum["md5_base64"]
            and entry.get("generation") == checksum.get("generation")
        )

    def verify_completed_target(
        item: dict[str, Any],
        target: Path,
    ) -> tuple[bool, str]:
        relative = str(item["relative_path"])
        checksum = checksum_by_path.get(relative)
        if not checksum:
            return True, "size_only"
        if ledger_entry_is_current(relative, target, checksum):
            return True, "ledger_hit"
        local_md5 = file_md5_base64(target)
        if local_md5 != checksum["md5_base64"]:
            invalid = (
                args.checksum_quarantine
                / Path(relative).parent
                / (
                    Path(relative).name
                    + f".checksum-mismatch-{int(time.time())}"
                )
            )
            invalid.parent.mkdir(parents=True, exist_ok=True)
            target.replace(invalid)
            with lock:
                failures[relative] = (
                    f"official_md5={checksum['md5_base64']} "
                    f"local_md5={local_md5} isolated={invalid}"
                )
            return False, str(invalid)
        stat = target.stat()
        with lock:
            checksum_ledger.setdefault("objects", {})[relative] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "official_md5_base64": checksum["md5_base64"],
                "local_md5_base64": local_md5,
                "official_crc32c_base64": checksum.get("crc32c_base64"),
                "generation": checksum.get("generation"),
                "verified_at": now(),
            }
            checksum_ledger["generated_at"] = now()
            atomic_write_json(args.checksum_ledger, checksum_ledger)
        return True, "md5_verified"

    def snapshot(status: str) -> None:
        nonlocal last_transport_progress_bytes, last_transport_change
        completed_objects = 0
        completed_bytes = 0
        partial_bytes = 0
        for item in objects:
            target = args.target / item["relative_path"]
            part = target.with_name(target.name + ".qtail.part")
            if file_size(target) == int(item["bytes"]):
                completed_objects += 1
                completed_bytes += int(item["bytes"])
            else:
                partial_bytes += min(file_size(part), int(item["bytes"]))
        with lock:
            active_items = []
            for entry in active.values():
                current = file_size(Path(entry["part_path"]))
                active_items.append(
                    {
                        key: value
                        for key, value in entry.items()
                        if key not in {"part_path", "inflight_path", "headers_path"}
                    }
                    | {
                        "current_bytes": current,
                        "inflight_bytes": file_size(Path(entry["inflight_path"])),
                        "downloaded_this_attempt": max(
                            0, current - int(entry["resumed_from"])
                        ),
                    }
                )
            transport_progress_bytes = (
                completed_bytes
                + partial_bytes
                + sum(int(entry["inflight_bytes"]) for entry in active_items)
            )
            monotonic_now = time.monotonic()
            if transport_progress_bytes > last_transport_progress_bytes:
                last_transport_progress_bytes = transport_progress_bytes
                last_transport_change = monotonic_now
            transport_stalled_seconds = max(
                0.0,
                monotonic_now - last_transport_change,
            )
            if (
                args.stall_timeout_seconds > 0
                and transport_stalled_seconds >= args.stall_timeout_seconds
            ):
                stall_abort.set()
            payload = {
                "generated_at": now(),
                "status": status,
                "workers": args.workers,
                "primary_endpoints": args.primary_endpoints,
                "endpoint_policy_version": PUBLIC_ENDPOINT_POLICY_VERSION,
                "endpoint_priority": [
                    "storage-download.googleapis.com",
                    "storage.googleapis.com JSON media API",
                    "storage.googleapis.com path API",
                    "<bucket>.storage.googleapis.com",
                ],
                "http_protocol": PUBLIC_HTTP_PROTOCOL,
                "manifest": str(args.manifest),
                "checksum_manifest": (
                    str(args.checksum_manifest)
                    if args.checksum_manifest
                    else None
                ),
                "checksum_ledger": (
                    str(args.checksum_ledger)
                    if args.checksum_ledger
                    else None
                ),
                "checksum_expected_objects": len(checksum_by_path),
                "checksum_verified_objects": len(
                    checksum_ledger.get("objects", {})
                ),
                "target": str(args.target),
                "required_mount": str(args.required_mount) if args.required_mount else None,
                "disk_headroom": disk_headroom(),
                "process_lock": {
                    "version": "qtail_droid_downloader_single_writer_v1",
                    "path": str(process_lock_path),
                    "owner_pid": os.getpid(),
                    "started_at": process_started_at,
                    "held": True,
                },
                "object_count": len(objects),
                "expected_bytes": expected_total,
                "completed_objects": completed_objects,
                "completed_bytes": completed_bytes,
                "partial_bytes": partial_bytes,
                "progress_bytes": completed_bytes + partial_bytes,
                "transport_progress_bytes": transport_progress_bytes,
                "progress_percent": (completed_bytes + partial_bytes) / expected_total * 100.0,
                "transport_stalled_seconds": transport_stalled_seconds,
                "stall_timeout_seconds": args.stall_timeout_seconds,
                "stall_abort_requested": stall_abort.is_set(),
                "route_guard": {
                    "enabled": args.forbid_tunnel_route,
                    "expected_interface": args.expected_interface,
                    "curl_interface_bound": bool(args.expected_interface),
                    "status": (
                        "passed"
                        if args.forbid_tunnel_route
                        else "not_enabled"
                    ),
                    "forbidden_interface_pattern": (
                        FORBIDDEN_ROUTE_INTERFACE.pattern
                    ),
                    "observations": sorted(
                        route_observations.values(),
                        key=lambda item: item["host"],
                    ),
                },
                "active": active_items,
                "failures": failures,
                "completed_this_run": completed_this_run,
            }
        atomic_write_json(args.status, payload)

    def heartbeat() -> None:
        while not stop_heartbeat.wait(max(1, args.heartbeat_seconds)):
            snapshot(f"downloading_round_{round_number}")

    def download(
        item: dict[str, Any], endpoint_index: int
    ) -> tuple[str, bool, str]:
        nonlocal completed_this_run
        if args.required_mount and not os.path.ismount(args.required_mount):
            raise RuntimeError(f"Required mount disappeared: {args.required_mount}")
        relative = item["relative_path"]
        expected = int(item["bytes"])
        target = args.target / relative
        part = target.with_name(target.name + ".qtail.part")
        inflight = part.with_name(part.name + ".inflight")
        headers = inflight.with_name(inflight.name + ".headers")
        target.parent.mkdir(parents=True, exist_ok=True)
        if file_size(target) == expected:
            return relative, True, "already_complete"
        if target.exists() and file_size(target) != expected:
            invalid = target.with_name(target.name + f".invalid-{int(time.time())}")
            target.replace(invalid)
        if file_size(part) > expected:
            invalid = part.with_name(part.name + f".invalid-{int(time.time())}")
            part.replace(invalid)
        inflight.unlink(missing_ok=True)
        headers.unlink(missing_ok=True)

        with lock:
            active[relative] = {
                "relative_path": relative,
                "expected_bytes": expected,
                "resumed_from": file_size(part),
                "started_at": now(),
                "part_path": str(part),
                "inflight_path": str(inflight),
                "headers_path": str(headers),
                "endpoint_index": endpoint_index,
                "endpoint_url": public_url(item["uri"], endpoint_index),
            }
        chunk_bytes = max(1, args.chunk_mib) * 1024 * 1024
        attempts = 0
        consecutive_failures = 0
        current_endpoint_index = endpoint_index
        while file_size(part) < expected:
            if stall_abort.is_set():
                raise RuntimeError(
                    f"Transport made no byte progress for "
                    f"{args.stall_timeout_seconds} seconds"
                )
            if args.required_mount and not os.path.ismount(args.required_mount):
                raise RuntimeError(f"Required mount disappeared: {args.required_mount}")
            attempts += 1
            start = file_size(part)
            end = min(expected, start + chunk_bytes) - 1
            requested = end - start + 1
            assert_disk_headroom(requested)
            inflight.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            command = [
                str(args.curl),
                "--location",
                "--silent",
                "--show-error",
                "--http2",
                "--connect-timeout",
                "30",
                "--max-time",
                "240",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--range",
                f"{start}-{end}",
                "--output",
                str(inflight),
                "--dump-header",
                str(headers),
                "--write-out",
                "%{http_code}",
            ]
            proxy = resolve_proxy(args.proxy)
            if proxy:
                command.extend(["--proxy", proxy])
            else:
                # Keep direct mode direct even if proxy environment variables
                # or desktop proxy settings are enabled later.
                command.extend(["--noproxy", "*"])
                if args.expected_interface:
                    command.extend(
                        ["--interface", args.expected_interface]
                    )
            endpoint_url = public_url(item["uri"], current_endpoint_index)
            route_observation: dict[str, str] | None = None
            if args.forbid_tunnel_route:
                route_observation = direct_route_for_url(
                    endpoint_url,
                    expected_interface=args.expected_interface,
                )
            with lock:
                active[relative]["proxy"] = proxy or "direct"
                active[relative]["endpoint_index"] = current_endpoint_index
                active[relative]["endpoint_url"] = endpoint_url
                if route_observation:
                    active[relative]["route_interface"] = route_observation[
                        "interface"
                    ]
                    active[relative]["route_gateway"] = route_observation[
                        "gateway"
                    ]
                    active[relative]["route_ipv4"] = route_observation[
                        "resolved_ipv4"
                    ]
                    route_observations[route_observation["host"]] = (
                        route_observation
                    )
            command.append(endpoint_url)
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            received = file_size(inflight)
            response_range = content_range(headers)
            try:
                http_code = int(result.stdout.strip()[-3:])
            except (TypeError, ValueError):
                http_code = 0

            accepted = (
                http_code == 206
                and 0 < received <= requested
                and response_range == (start, end, expected)
            )
            if http_code == 200 and start == 0 and received == expected:
                accepted = True
            if accepted:
                with part.open("ab") as destination, inflight.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())
                inflight.unlink(missing_ok=True)
                headers.unlink(missing_ok=True)
                with lock:
                    failures.pop(relative, None)
                    active[relative]["consecutive_failures"] = 0
                    active[relative]["retry_delay_seconds"] = 0
                consecutive_failures = 0
                continue

            error = (
                f"attempt={attempts} endpoint={current_endpoint_index} "
                f"curl={result.returncode} http={http_code} "
                f"range={start}-{end} received={received} expected={expected} "
                f"content_range={response_range} "
                f"stderr={result.stderr.strip()[-500:]}"
            )
            inflight.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            with lock:
                failures[relative] = error
            consecutive_failures += 1
            current_endpoint_index = (current_endpoint_index + 1) % 4
            retry_delay = retry_delay_seconds(
                args.attempt_retry_seconds,
                consecutive_failures,
                attempts,
            )
            with lock:
                active[relative]["consecutive_failures"] = consecutive_failures
                active[relative]["retry_delay_seconds"] = retry_delay
            time.sleep(retry_delay)

        actual = file_size(part)
        if actual == expected:
            part.replace(target)
            verified, checksum_detail = verify_completed_target(item, target)
            if not verified:
                raise RuntimeError(
                    f"Official MD5 mismatch for {relative}: {checksum_detail}"
                )
            with lock:
                active.pop(relative, None)
                completed_this_run += 1
                failures.pop(relative, None)
            return relative, True, f"downloaded:{checksum_detail}"
        error = f"actual={actual} expected={expected}"
        with lock:
            failures[relative] = error
        return relative, False, error

    round_number = 0
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="qtail-download-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        if checksum_by_path:
            for item in objects:
                target = args.target / item["relative_path"]
                if file_size(target) != int(item["bytes"]):
                    continue
                verified, checksum_detail = verify_completed_target(
                    item,
                    target,
                )
                if not verified:
                    with lock:
                        failures[str(item["relative_path"])] = (
                            f"checksum backfill isolated {checksum_detail}"
                        )
            snapshot("checksum_backfill_complete")
        while True:
            pending = [
                item
                for item in objects
                if file_size(args.target / item["relative_path"]) != int(item["bytes"])
            ]
            if not pending:
                stop_heartbeat.set()
                heartbeat_thread.join(
                    timeout=max(1, args.heartbeat_seconds + 1)
                )
                snapshot("complete")
                return
            round_number += 1
            snapshot(f"downloading_round_{round_number}")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(download, item, index % args.primary_endpoints)
                    for index, item in enumerate(pending)
                ]
                for future in as_completed(futures):
                    future.result()
                    snapshot(f"downloading_round_{round_number}")
            snapshot(f"round_{round_number}_complete")
            time.sleep(args.retry_delay)
    except Exception as error:
        with lock:
            failures["__pipeline__"] = (
                f"{type(error).__name__}: {error}"
            )
        snapshot("failed")
        raise
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=max(1, args.heartbeat_seconds + 1))


if __name__ == "__main__":
    main()
