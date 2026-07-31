#!/usr/bin/env python3
"""Positive/negative controls for bounded DROID shard-list selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from qtail_train_droid_full import load_bounded_shard_list


def digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def write_manifest(path: Path, relative_paths: list[str], **extra: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "qtail_bounded_droid_shard_list_v1",
                "relative_paths": relative_paths,
                "relative_paths_sha256": digest(relative_paths),
                **extra,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def rejected(action: Callable[[], Any], expected: str) -> bool:
    try:
        action()
    except SystemExit as error:
        return expected in str(error)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    controls: list[dict[str, Any]] = []

    def record(name: str, passed: bool) -> None:
        controls.append({"name": name, "passed": passed})

    with tempfile.TemporaryDirectory(
        prefix="qtail-droid-shard-list-selftest."
    ) as temporary:
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        for name in ("a.tfrecord", "b.tfrecord"):
            (data / name).write_bytes(name.encode("ascii"))
        valid = root / "valid.json"
        write_manifest(valid, ["a.tfrecord", "b.tfrecord"])
        selected = load_bounded_shard_list(
            data_dir=data,
            shard_list_path=valid,
        )
        record(
            "valid_sorted_digest_bound_list_accepted",
            [path.name for path in selected]
            == ["a.tfrecord", "b.tfrecord"],
        )

        duplicate = root / "duplicate.json"
        write_manifest(duplicate, ["a.tfrecord", "a.tfrecord"])
        record(
            "duplicate_paths_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=duplicate,
                ),
                "duplicate",
            ),
        )

        unsorted = root / "unsorted.json"
        write_manifest(unsorted, ["b.tfrecord", "a.tfrecord"])
        record(
            "unsorted_paths_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=unsorted,
                ),
                "sorted",
            ),
        )

        tampered = root / "tampered.json"
        write_manifest(
            tampered,
            ["a.tfrecord"],
        )
        payload = json.loads(tampered.read_text(encoding="utf-8"))
        payload["relative_paths_sha256"] = "0" * 64
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        record(
            "membership_digest_tamper_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=tampered,
                ),
                "digest mismatch",
            ),
        )

        traversal = root / "traversal.json"
        write_manifest(traversal, ["../outside.tfrecord"])
        record(
            "path_traversal_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=traversal,
                ),
                "unsafe path",
            ),
        )

        outside = root / "outside.tfrecord"
        outside.write_bytes(b"outside")
        (data / "escape.tfrecord").symlink_to(outside)
        escape = root / "escape.json"
        write_manifest(escape, ["escape.tfrecord"])
        record(
            "external_symlink_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=escape,
                ),
                "escapes data-dir",
            ),
        )

        partial = data / "partial.tfrecord.qtail.part"
        partial.write_bytes(b"partial")
        partial_manifest = root / "partial.json"
        write_manifest(
            partial_manifest,
            ["partial.tfrecord.qtail.part"],
        )
        record(
            "partial_download_rejected",
            rejected(
                lambda: load_bounded_shard_list(
                    data_dir=data,
                    shard_list_path=partial_manifest,
                ),
                "not a complete TFRecord",
            ),
        )

        conflict = subprocess.run(
            [
                sys.executable,
                str(args.trainer),
                "--data-dir",
                str(data),
                "--out",
                str(root / "out"),
                "--shard-list",
                str(valid),
                "--max-shards",
                "1",
                "--features-only",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        record(
            "max_shards_conflict_rejected_before_scan",
            conflict.returncode != 0
            and "mutually exclusive" in (
                conflict.stdout + conflict.stderr
            ),
        )

    report = json.loads(args.canary_report.read_text(encoding="utf-8"))
    record(
        "live_frozen_canary_is_bounded_not_formal",
        report.get("training_scope") == "bounded_test_subset"
        and report.get("formal_protocol", {}).get("locked") is False
        and report.get("input_audit", {}).get("verified") is False,
    )

    payload = {
        "version": "qtail_droid_shard_list_selftest_v1",
        "status": (
            "passed"
            if all(item["passed"] for item in controls)
            else "failed"
        ),
        "controls": controls,
        "passed": sum(item["passed"] for item in controls),
        "total": len(controls),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary_out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_out.replace(args.out)
    print(json.dumps(payload))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
