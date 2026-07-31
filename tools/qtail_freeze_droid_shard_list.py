#!/usr/bin/env python3
"""Freeze an audited relative-path list from DROID shard feature rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_dir.resolve()
    rows: list[dict[str, Any]] = []
    with args.rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["path"])
            resolved = source.resolve()
            if not source.is_file() or not resolved.is_relative_to(data_root):
                raise SystemExit(f"Invalid shard source: {source}")
            relative = str(resolved.relative_to(data_root))
            rows.append(
                {
                    "relative_path": relative,
                    "bytes": int(row["bytes"]),
                    "records_decoded": int(row["records_decoded"]),
                    "boundary_sha256": row["boundary_sha256"],
                }
            )
    rows.sort(key=lambda item: item["relative_path"])
    relative_paths = [item["relative_path"] for item in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise SystemExit("Duplicate shard paths in feature rows.")
    membership_digest = hashlib.sha256(
        "\n".join(relative_paths).encode("utf-8")
    ).hexdigest()
    payload = {
        "version": "qtail_bounded_droid_shard_list_v1",
        "source_rows": str(args.rows),
        "source_rows_sha256": sha256(args.rows),
        "data_dir": str(args.data_dir),
        "shard_count": len(rows),
        "records_decoded": sum(
            int(item["records_decoded"]) for item in rows
        ),
        "represented_bytes": sum(int(item["bytes"]) for item in rows),
        "relative_paths_sha256": membership_digest,
        "relative_paths": relative_paths,
        "entries": rows,
        "claim_boundary": (
            "Frozen membership for a bounded engineering canary only; "
            "never a formal full-mirror input."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
