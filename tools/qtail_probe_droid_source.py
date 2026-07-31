#!/usr/bin/env python3
"""Probe the official DROID bucket and persist an auditable source manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsutil", type=Path, required=True)
    parser.add_argument("--source", default="gs://gresearch/robotics/droid")
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--marker-dir", type=Path)
    args = parser.parse_args()

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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    if args.marker_dir:
        args.marker_dir.mkdir(parents=True, exist_ok=True)
        (args.marker_dir / "DROID_SOURCE_PROBED").touch()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
