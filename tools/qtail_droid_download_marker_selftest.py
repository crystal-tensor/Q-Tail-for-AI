#!/usr/bin/env python3
"""Positive and mutation controls for the DROID download marker contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def md5_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


def run(
    *,
    python: Path,
    verifier: Path,
    root: Path,
    write: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(python),
        str(verifier),
        "--data-dir",
        str(root / "data"),
        "--manifest",
        str(root / "manifest.json"),
        "--checksum-manifest",
        str(root / "checksums.json"),
        "--checksum-ledger",
        str(root / "ledger.json"),
        "--transport-status",
        str(root / "transport.json"),
        "--marker",
        str(root / "DROID_DOWNLOAD_COMPLETE"),
        "--expected-bytes",
        "9",
        "--expected-objects",
        "2",
        "--expected-tfrecords",
        "2",
    ]
    if write:
        command.append("--write")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def fixture(root: Path) -> None:
    rows = [
        ("1.0.0/a.tfrecord-00000-of-00001", b"abcd", "g1"),
        ("1.0.1/b.tfrecord-00000-of-00001", b"efghi", "g2"),
    ]
    manifest_items = []
    checksum_items = []
    ledger_objects = {}
    for relative, payload, generation in rows:
        path = root / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        stat = path.stat()
        md5 = md5_base64(payload)
        manifest_items.append(
            {"relative_path": relative, "bytes": len(payload)}
        )
        checksum_items.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "md5_base64": md5,
                "generation": generation,
            }
        )
        ledger_objects[relative] = {
            "bytes": len(payload),
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "official_md5_base64": md5,
            "local_md5_base64": md5,
            "generation": generation,
        }
    atomic_write(
        root / "manifest.json",
        {
            "status": "verified",
            "object_count": 2,
            "total_bytes": 9,
            "objects": manifest_items,
        },
    )
    atomic_write(
        root / "checksums.json",
        {
            "status": "verified",
            "object_count": 2,
            "total_bytes": 9,
            "objects": checksum_items,
        },
    )
    atomic_write(
        root / "ledger.json",
        {"format_version": 1, "objects": ledger_objects},
    )
    atomic_write(
        root / "transport.json",
        {
            "status": "complete",
            "object_count": 2,
            "expected_bytes": 9,
            "completed_objects": 2,
            "completed_bytes": 9,
            "checksum_expected_objects": 2,
            "checksum_verified_objects": 2,
            "active": [],
            "failures": {},
            "route_guard": {"enabled": True, "status": "passed"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    controls: list[dict[str, Any]] = []

    def record(name: str, result: subprocess.CompletedProcess[str], passed: bool):
        controls.append(
            {
                "name": name,
                "passed": passed,
                "returncode": result.returncode,
                "stderr": result.stderr.strip()[-1000:],
            }
        )

    with tempfile.TemporaryDirectory(prefix="qtail-download-marker-") as tmp:
        root = Path(tmp)
        fixture(root)
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
            write=True,
        )
        record("positive_write", result, result.returncode == 0)

        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
        )
        record("positive_verify", result, result.returncode == 0)

        marker_path = root / "DROID_DOWNLOAD_COMPLETE"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["binding"]["object_count"] = 1
        atomic_write(marker_path, marker)
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
        )
        record("tampered_marker_rejected", result, result.returncode != 0)

        fixture(root)
        marker_path.write_bytes(b"")
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
            write=True,
        )
        record("legacy_empty_marker_upgraded", result, result.returncode == 0)

        transport_path = root / "transport.json"
        transport = json.loads(transport_path.read_text(encoding="utf-8"))
        transport["route_guard"]["status"] = "failed"
        atomic_write(transport_path, transport)
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
        )
        record("wrong_route_rejected", result, result.returncode != 0)

        fixture(root)
        (root / "DROID_DOWNLOAD_COMPLETE").unlink(missing_ok=True)
        ledger_path = root / "ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        first = next(iter(ledger["objects"].values()))
        first["mtime_ns"] += 1
        atomic_write(ledger_path, ledger)
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
            write=True,
        )
        record("ledger_identity_rejected", result, result.returncode != 0)

        fixture(root)
        transport = json.loads(transport_path.read_text(encoding="utf-8"))
        transport["completed_objects"] = 1
        atomic_write(transport_path, transport)
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
            write=True,
        )
        record("transport_count_rejected", result, result.returncode != 0)

        fixture(root)
        changed = root / "data" / "1.0.0" / "a.tfrecord-00000-of-00001"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        expected_mtime_ns = int(
            ledger["objects"][
                "1.0.0/a.tfrecord-00000-of-00001"
            ]["mtime_ns"]
        )
        original_atime_ns = changed.stat().st_atime_ns
        changed.write_bytes(b"wxyz")
        os.utime(
            changed,
            ns=(original_atime_ns, expected_mtime_ns),
        )
        result = run(
            python=args.python,
            verifier=args.verifier,
            root=root,
            write=True,
        )
        record(
            "same_size_change_with_restored_mtime_rejected",
            result,
            result.returncode != 0,
        )

    passed = sum(bool(item["passed"]) for item in controls)
    payload = {
        "generated_at": now(),
        "status": "passed" if passed == len(controls) else "failed",
        "controls_passed": passed,
        "controls_total": len(controls),
        "controls": controls,
    }
    atomic_write(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit("DROID download marker self-test failed")


if __name__ == "__main__":
    main()
