#!/usr/bin/env python3
"""Build a direct-network, dataset-complete Open X expansion manifest."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


BUCKET = "gdm-robotics-open-x-embodiment"
API_HOST = "storage.googleapis.com"
DEFAULT_TARGET_BYTES = 1_099_511_627_776  # 1 TiB
QUANTUM_BYTES = 64 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def direct_route(expected_interface: str) -> dict[str, str]:
    addresses = sorted(
        {
            item[4][0]
            for item in socket.getaddrinfo(
                API_HOST,
                443,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        }
    )
    if not addresses:
        raise RuntimeError(f"No IPv4 address resolved for {API_HOST}")
    result = subprocess.run(
        ["/sbin/route", "-n", "get", addresses[0]],
        text=True,
        capture_output=True,
        check=False,
    )
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    interface = fields.get("interface", "")
    if result.returncode != 0 or interface != expected_interface:
        raise RuntimeError(
            f"Direct route gate failed: expected={expected_interface} "
            f"actual={interface or 'unknown'} address={addresses[0]}"
        )
    return {
        "host": API_HOST,
        "ipv4": addresses[0],
        "interface": interface,
        "gateway": fields.get("gateway", ""),
    }


def list_objects(curl: Path, interface: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    objects: list[dict[str, Any]] = []
    routes: list[dict[str, str]] = []
    token = ""
    page = 0
    while True:
        page += 1
        route = direct_route(interface)
        routes.append({**route, "page": str(page)})
        params = {
            "maxResults": "1000",
            "projection": "full",
            "fields": "nextPageToken,items(name,size,md5Hash,crc32c,generation)",
        }
        if token:
            params["pageToken"] = token
        url = f"https://{API_HOST}/storage/v1/b/{BUCKET}/o?{urlencode(params)}"
        environment = os.environ.copy()
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            environment.pop(key, None)
        environment["NO_PROXY"] = "*"
        environment["no_proxy"] = "*"
        payload: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(1, 11):
            result = subprocess.run(
                [
                    str(curl),
                    "--fail-with-body",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "300",
                    "--noproxy",
                    "*",
                    "--interface",
                    interface,
                    url,
                ],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            if result.returncode == 0:
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError as error:
                    last_error = f"invalid JSON: {error}"
                else:
                    if isinstance(parsed, dict):
                        payload = parsed
                        break
            else:
                last_error = result.stderr.strip()[-1000:]
            time.sleep(min(30, attempt * 2))
        if payload is None:
            raise RuntimeError(f"Open X object listing failed on page {page}: {last_error}")
        for item in payload.get("items", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            objects.append(
                {
                    "name": str(item["name"]),
                    "bytes": int(item.get("size", 0)),
                    "md5_base64": item.get("md5Hash"),
                    "crc32c_base64": item.get("crc32c"),
                    "generation": item.get("generation"),
                }
            )
        token = str(payload.get("nextPageToken", ""))
        if not token:
            break
    objects.sort(key=lambda item: item["name"])
    return objects, routes


def choose_datasets(
    candidates: list[dict[str, Any]],
    target_bytes: int,
    max_single_fraction: float,
) -> list[dict[str, Any]]:
    maximum_single = int(target_bytes * max_single_fraction)
    eligible = [
        item
        for item in candidates
        if 0 < int(item["bytes"]) <= maximum_single
    ]
    # Quantized knapsack maximizes downloaded bytes. For equal-size states,
    # prefer more complete datasets to preserve cross-embodiment diversity.
    target_units = target_bytes // QUANTUM_BYTES
    states: dict[int, tuple[int, tuple[int, ...], int]] = {0: (0, (), 0)}
    for index, item in enumerate(eligible):
        units = (int(item["bytes"]) + QUANTUM_BYTES - 1) // QUANTUM_BYTES
        updated = dict(states)
        for prior_units, (count, selected, actual_bytes) in states.items():
            new_units = prior_units + units
            new_bytes = actual_bytes + int(item["bytes"])
            if new_units > target_units or new_bytes > target_bytes:
                continue
            proposal = (count + 1, selected + (index,), new_bytes)
            current = updated.get(new_units)
            if current is None or (proposal[2], proposal[0]) > (current[2], current[0]):
                updated[new_units] = proposal
        states = updated
    best = max(states.values(), key=lambda state: (state[2], state[0]))
    return [eligible[index] for index in best[1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--curl", type=Path, default=Path("/usr/bin/curl"))
    parser.add_argument("--interface", default="en1")
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--max-single-fraction", type=float, default=0.40)
    args = parser.parse_args()

    if not 0 < args.max_single_fraction <= 1:
        raise SystemExit("--max-single-fraction must be in (0, 1]")
    objects, routes = list_objects(args.curl, args.interface)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        top = item["name"].split("/", 1)[0]
        if top:
            grouped[top].append(item)

    existing = {
        path.name
        for path in args.data_dir.iterdir()
        if path.is_dir()
    } if args.data_dir.exists() else set()
    datasets: list[dict[str, Any]] = []
    for name, entries in sorted(grouped.items()):
        names = {entry["name"].rsplit("/", 1)[-1] for entry in entries}
        tfrecords = [
            entry for entry in entries
            if "tfrecord" in entry["name"].lower() and int(entry["bytes"]) > 0
        ]
        missing_md5 = sum(
            1 for entry in entries
            if int(entry["bytes"]) > 0 and not entry.get("md5_base64")
        )
        is_dataset = bool(tfrecords) and {
            "dataset_info.json",
            "features.json",
        }.issubset(names)
        datasets.append(
            {
                "dataset": name,
                "bytes": sum(int(entry["bytes"]) for entry in entries),
                "gib": round(sum(int(entry["bytes"]) for entry in entries) / 2**30, 3),
                "object_count": len(entries),
                "tfrecord_count": len(tfrecords),
                "missing_md5_count": missing_md5,
                "dataset_signature": is_dataset,
                "already_local": name in existing,
            }
        )

    candidates = [
        item for item in datasets
        if item["dataset_signature"]
        and not item["already_local"]
        and item["missing_md5_count"] == 0
    ]
    selected = choose_datasets(
        candidates,
        args.target_bytes,
        args.max_single_fraction,
    )
    selected_names = {item["dataset"] for item in selected}
    selected_objects = [
        entry
        for name in sorted(selected_names)
        for entry in grouped[name]
        if int(entry["bytes"]) > 0
    ]
    download_objects = [
        {
            "uri": f"gs://{BUCKET}/{entry['name']}",
            "relative_path": entry["name"],
            "bytes": int(entry["bytes"]),
        }
        for entry in selected_objects
    ]
    checksum_objects = [
        {
            "relative_path": entry["name"],
            "bytes": int(entry["bytes"]),
            "md5_base64": entry["md5_base64"],
            "crc32c_base64": entry.get("crc32c_base64"),
            "generation": entry.get("generation"),
        }
        for entry in selected_objects
    ]
    total_bytes = sum(item["bytes"] for item in download_objects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "format_version": "qtail_openx_direct_catalog_v1",
        "generated_at": now(),
        "status": "verified",
        "source": f"gs://{BUCKET}",
        "transport": {
            "mode": "direct_physical_interface_bound",
            "interface": args.interface,
            "proxy": "disabled",
            "routes": routes,
        },
        "bucket_object_count": len(objects),
        "dataset_signature_count": sum(bool(item["dataset_signature"]) for item in datasets),
        "existing_dataset_count": len(existing),
        "candidate_dataset_count": len(candidates),
        "target_bytes": args.target_bytes,
        "selected_bytes": total_bytes,
        "selected_gib": round(total_bytes / 2**30, 3),
        "selected_dataset_count": len(selected),
        "selected_datasets": sorted(selected, key=lambda item: item["dataset"]),
        "datasets": datasets,
    }
    manifest = {
        "format_version": "qtail_openx_expansion_manifest_v1",
        "generated_at": now(),
        "status": "verified",
        "source": f"gs://{BUCKET}",
        "target_bytes": args.target_bytes,
        "total_bytes": total_bytes,
        "object_count": len(download_objects),
        "dataset_count": len(selected),
        "datasets": sorted(selected_names),
        "objects": download_objects,
    }
    checksum = {
        "format_version": "qtail_openx_expansion_checksum_manifest_v1",
        "generated_at": now(),
        "status": "verified",
        "source": f"gs://{BUCKET}",
        "total_bytes": total_bytes,
        "object_count": len(checksum_objects),
        "objects": checksum_objects,
    }
    atomic_json(args.out_dir / "openx_bucket_catalog.json", catalog)
    atomic_json(args.out_dir / "openx_1t_object_manifest.json", manifest)
    atomic_json(args.out_dir / "openx_1t_checksum_manifest.json", checksum)
    print(
        json.dumps(
            {
                "status": "verified",
                "dataset_signatures": catalog["dataset_signature_count"],
                "selected_datasets": len(selected),
                "selected_objects": len(download_objects),
                "selected_gib": catalog["selected_gib"],
                "interface": args.interface,
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
