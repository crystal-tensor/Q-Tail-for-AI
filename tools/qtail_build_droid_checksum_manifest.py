#!/usr/bin/env python3
"""Build an official GCS hash manifest for every DROID object."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--curl", type=Path, default=Path("/usr/bin/curl"))
    parser.add_argument("--bucket", default="gresearch")
    parser.add_argument("--prefix", default="robotics/droid/")
    args = parser.parse_args()

    size_manifest = json.loads(
        args.size_manifest.read_text(encoding="utf-8")
    )
    expected = {
        str(item["relative_path"]): int(item["bytes"])
        for item in size_manifest.get("objects", [])
    }
    if not expected:
        raise SystemExit("Size manifest contains no objects")

    fields = "items(name,size,md5Hash,crc32c,generation),nextPageToken"
    objects: list[dict[str, Any]] = []
    page_token = ""
    page_count = 0
    while True:
        query = {
            "prefix": args.prefix,
            "maxResults": 1000,
            "fields": fields,
        }
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{args.bucket}/o?"
            f"{urlencode(query)}"
        )
        result = subprocess.run(
            [
                str(args.curl),
                "--noproxy",
                "*",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "20",
                "--max-time",
                "120",
                url,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"GCS JSON listing failed on page {page_count + 1}: "
                f"{result.stderr.strip()}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"GCS JSON listing returned invalid JSON: {error}"
            ) from error
        page_count += 1
        for item in payload.get("items", []):
            name = str(item.get("name", ""))
            if not name.startswith(args.prefix):
                continue
            relative_path = name[len(args.prefix) :]
            objects.append(
                {
                    "relative_path": relative_path,
                    "bytes": int(item["size"]),
                    "md5_base64": str(item.get("md5Hash", "")),
                    "crc32c_base64": str(item.get("crc32c", "")),
                    "generation": str(item.get("generation", "")),
                }
            )
        page_token = str(payload.get("nextPageToken", ""))
        if not page_token:
            break

    objects.sort(key=lambda item: item["relative_path"])
    actual = {
        item["relative_path"]: int(item["bytes"])
        for item in objects
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    size_mismatches = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path] != actual[path]
    )
    missing_hashes = sorted(
        item["relative_path"]
        for item in objects
        if not item["md5_base64"] or not item["crc32c_base64"]
    )
    if missing or extra or size_mismatches or missing_hashes:
        raise SystemExit(
            "Official checksum listing does not match the size manifest: "
            f"missing={len(missing)} extra={len(extra)} "
            f"size_mismatches={len(size_mismatches)} "
            f"missing_hashes={len(missing_hashes)}"
        )
    if len(objects) != int(size_manifest.get("object_count", -1)):
        raise SystemExit(
            f"Checksum object count {len(objects)} does not match "
            f"{size_manifest.get('object_count')}"
        )

    atomic_write_json(
        args.out,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "verified",
            "source": size_manifest.get("source"),
            "api_bucket": args.bucket,
            "api_prefix": args.prefix,
            "page_count": page_count,
            "size_manifest": str(args.size_manifest),
            "object_count": len(objects),
            "total_bytes": sum(item["bytes"] for item in objects),
            "objects": objects,
        },
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "pages": page_count,
                "objects": len(objects),
                "bytes": sum(item["bytes"] for item in objects),
            }
        )
    )


if __name__ == "__main__":
    main()
