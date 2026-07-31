#!/usr/bin/env python3
"""Build a size-audited object manifest for the public DROID bucket."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OBJECT_LINE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S+)\s*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsutil", type=Path, required=True)
    parser.add_argument("--source", default="gs://gresearch/robotics/droid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    args = parser.parse_args()

    command = [str(args.gsutil), "ls", "-l", "-r", f"{args.source}/**"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Object listing failed: {result.stderr.strip()}")

    prefix = args.source.rstrip("/") + "/"
    objects = []
    for line in result.stdout.splitlines():
        match = OBJECT_LINE.match(line)
        if not match:
            continue
        size = int(match.group(1))
        uri = match.group(2)
        if not uri.startswith(prefix):
            continue
        relative_path = uri[len(prefix) :]
        objects.append({"uri": uri, "relative_path": relative_path, "bytes": size})

    objects.sort(key=lambda item: item["relative_path"])
    total = sum(item["bytes"] for item in objects)
    if total != args.expected_bytes:
        raise SystemExit(
            f"Manifest bytes do not match source probe: {total} != {args.expected_bytes}"
        )
    if not objects:
        raise SystemExit("No DROID objects found")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "source": args.source,
        "listing_command": " ".join(command),
        "object_count": len(objects),
        "total_bytes": total,
        "objects": objects,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "objects": len(objects), "bytes": total}))


if __name__ == "__main__":
    main()
