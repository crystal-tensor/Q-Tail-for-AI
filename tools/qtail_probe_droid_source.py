#!/usr/bin/env python3
"""Probe the official DROID bucket and persist an auditable source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_probe(
    payload: dict,
    *,
    source: str,
    expected_bytes: int,
    job_root: Path,
) -> None:
    storage = payload.get("storage", {})
    if (
        payload.get("status") != "verified"
        or payload.get("source") != source
        or payload.get("remote_bytes") != expected_bytes
        or payload.get("job_root") != str(job_root)
        or not isinstance(storage, dict)
        or storage.get("capacity_gate_passed") is not True
        or storage.get("required_with_5_percent_slack")
        != int(expected_bytes * 1.05)
    ):
        raise SystemExit("stored DROID source probe is stale or invalid")


def write_marker(
    *,
    marker_dir: Path,
    report: Path,
    payload: dict,
) -> Path:
    marker = marker_dir / "DROID_SOURCE_PROBED"
    marker_payload = {
        "format_version": "qtail_droid_source_probe_marker_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "source": payload["source"],
        "remote_bytes": payload["remote_bytes"],
        "job_root": payload["job_root"],
        "report": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "capacity_gate_passed_at_probe": payload["storage"][
            "capacity_gate_passed"
        ],
        "required_with_5_percent_slack": payload["storage"][
            "required_with_5_percent_slack"
        ],
        "claim_boundary": (
            "This binds the official URI and observed byte count to the "
            "stored source-probe report. Current free-space sufficiency is "
            "evaluated separately from this historical probe."
        ),
    }
    atomic_write_json(marker, marker_payload)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsutil", type=Path, required=True)
    parser.add_argument("--source", default="gs://gresearch/robotics/droid")
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--marker-dir", type=Path)
    parser.add_argument("--seal-existing", action="store_true")
    args = parser.parse_args()

    if args.seal_existing:
        try:
            payload = json.loads(args.out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"stored DROID source probe is unreadable: {error}"
            ) from error
        validate_probe(
            payload,
            source=args.source,
            expected_bytes=args.expected_bytes,
            job_root=args.job_root,
        )
        if not args.marker_dir:
            raise SystemExit("--seal-existing requires --marker-dir")
        marker = write_marker(
            marker_dir=args.marker_dir,
            report=args.out,
            payload=payload,
        )
        print(
            json.dumps(
                {
                    "status": "sealed_existing",
                    "report": str(args.out),
                    "marker": str(marker),
                },
                indent=2,
            )
        )
        return

    size = run([str(args.gsutil), "du", "-s", args.source])
    if size.returncode != 0:
        raise SystemExit(f"gsutil du failed: {size.stderr.strip()}")
    fields = size.stdout.strip().split()
    if len(fields) < 2 or not fields[0].isdigit():
        raise SystemExit(f"Unexpected gsutil du output: {size.stdout!r}")
    remote_bytes = int(fields[0])
    if remote_bytes != args.expected_bytes:
        raise SystemExit(
            f"Official DROID size changed: expected {args.expected_bytes}, observed {remote_bytes}"
        )

    version = run([str(args.gsutil), "version", "-l"])
    usage = shutil.disk_usage(args.job_root)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "source": args.source,
        "remote_bytes": remote_bytes,
        "remote_tib": remote_bytes / (1024**4),
        "probe_command": f"{args.gsutil} du -s {args.source}",
        "probe_stdout": size.stdout.strip(),
        "gsutil_version": version.stdout.strip(),
        "job_root": str(args.job_root),
        "storage": {
            "capacity_bytes": usage.total,
            "free_bytes_at_probe": usage.free,
            "required_with_5_percent_slack": int(remote_bytes * 1.05),
            "capacity_gate_passed": usage.free >= int(remote_bytes * 1.05),
        },
    }
    if not payload["storage"]["capacity_gate_passed"]:
        raise SystemExit("ORICO capacity gate failed at source probe")

    validate_probe(
        payload,
        source=args.source,
        expected_bytes=args.expected_bytes,
        job_root=args.job_root,
    )
    atomic_write_json(args.out, payload)
    if args.marker_dir:
        write_marker(
            marker_dir=args.marker_dir,
            report=args.out,
            payload=payload,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
