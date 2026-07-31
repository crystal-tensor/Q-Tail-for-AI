#!/usr/bin/env python3
"""Exercise the final DROID mirror verifier with destructive controls."""

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
from typing import Any, Callable


OFFICIAL_SOURCE = "gs://gresearch/robotics/droid"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_fixture(root: Path) -> dict[str, Path | int]:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    payloads = {
        "1.0.0/example.tfrecord-00000-of-00001": b"record-data-v1",
        "1.0.0/dataset_info.json": b'{"name":"fixture"}\n',
    }
    objects: list[dict[str, Any]] = []
    checksum_objects: list[dict[str, Any]] = []
    ledger_objects: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for index, (relative, payload) in enumerate(payloads.items(), start=1):
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        stat = path.stat()
        checksum = md5_base64(payload)
        generation = str(1_000 + index)
        objects.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
            }
        )
        checksum_objects.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "md5_base64": checksum,
                "generation": generation,
            }
        )
        ledger_objects[relative] = {
            "bytes": len(payload),
            "official_md5_base64": checksum,
            "local_md5_base64": checksum,
            "generation": generation,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        total_bytes += len(payload)

    manifest = root / "manifest.json"
    checksum_manifest = root / "checksum_manifest.json"
    ledger = root / "ledger.json"
    write_json(
        manifest,
        {
            "status": "verified",
            "source": OFFICIAL_SOURCE,
            "object_count": len(objects),
            "total_bytes": total_bytes,
            "objects": objects,
        },
    )
    write_json(
        checksum_manifest,
        {
            "status": "verified",
            "source": OFFICIAL_SOURCE,
            "object_count": len(checksum_objects),
            "total_bytes": total_bytes,
            "objects": checksum_objects,
        },
    )
    write_json(
        ledger,
        {
            "format_version": 1,
            "objects": ledger_objects,
        },
    )
    return {
        "data_dir": data_dir,
        "manifest": manifest,
        "checksum_manifest": checksum_manifest,
        "ledger": ledger,
        "expected_bytes": total_bytes,
    }


def run_case(
    *,
    case_root: Path,
    python: Path,
    verifier: Path,
    expected_acceptance: bool,
    mutation: Callable[[dict[str, Path | int]], None] | None = None,
    checksum_returncode: int = 0,
    expected_tfrecords: int = 1,
) -> dict[str, Any]:
    fixture = build_fixture(case_root)
    if mutation is not None:
        mutation(fixture)
    report = case_root / "verification.json"
    command = [
        str(python),
        str(verifier),
        "--data-dir",
        str(fixture["data_dir"]),
        "--manifest",
        str(fixture["manifest"]),
        "--checksum-manifest",
        str(fixture["checksum_manifest"]),
        "--checksum-ledger",
        str(fixture["ledger"]),
        "--expected-bytes",
        str(fixture["expected_bytes"]),
        "--expected-objects",
        "2",
        "--expected-tfrecords",
        str(expected_tfrecords),
        "--checksum-returncode",
        str(checksum_returncode),
        "--rehash-local",
        "--out",
        str(report),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    observed_acceptance = completed.returncode == 0
    payload = (
        json.loads(report.read_text(encoding="utf-8"))
        if report.exists()
        else {}
    )
    return {
        "expected_acceptance": expected_acceptance,
        "observed_acceptance": observed_acceptance,
        "passed": observed_acceptance is expected_acceptance,
        "returncode": completed.returncode,
        "verifier_status": payload.get("status"),
        "stderr_tail": completed.stderr.strip()[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def missing_object(fixture: dict[str, Path | int]) -> None:
        (Path(fixture["data_dir"]) / "1.0.0/dataset_info.json").unlink()

    def same_size_change(fixture: dict[str, Path | int]) -> None:
        path = (
            Path(fixture["data_dir"])
            / "1.0.0/example.tfrecord-00000-of-00001"
        )
        stat = path.stat()
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed = path.stat()
        ledger_path = Path(fixture["ledger"])
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        entry = ledger["objects"][
            "1.0.0/example.tfrecord-00000-of-00001"
        ]
        entry["mtime_ns"] = changed.st_mtime_ns
        entry["ctime_ns"] = changed.st_ctime_ns
        write_json(ledger_path, ledger)

    def generation_mismatch(fixture: dict[str, Path | int]) -> None:
        ledger_path = Path(fixture["ledger"])
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["objects"][
            "1.0.0/example.tfrecord-00000-of-00001"
        ]["generation"] = "wrong-generation"
        write_json(ledger_path, ledger)

    def duplicate_manifest_path(fixture: dict[str, Path | int]) -> None:
        manifest_path = Path(fixture["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["objects"].append(dict(manifest["objects"][0]))
        write_json(manifest_path, manifest)

    def extra_file(fixture: dict[str, Path | int]) -> None:
        (Path(fixture["data_dir"]) / "unexpected.bin").write_bytes(b"extra")

    definitions = [
        ("positive_control_passes", True, None, 0, 1),
        ("nonzero_checksum_returncode_rejected", False, None, 9, 1),
        ("missing_object_rejected", False, missing_object, 0, 1),
        (
            "same_size_change_with_forged_ledger_rejected",
            False,
            same_size_change,
            0,
            1,
        ),
        (
            "ledger_generation_mismatch_rejected",
            False,
            generation_mismatch,
            0,
            1,
        ),
        (
            "duplicate_manifest_path_rejected",
            False,
            duplicate_manifest_path,
            0,
            1,
        ),
        ("extra_file_rejected", False, extra_file, 0, 1),
        ("wrong_expected_tfrecord_count_rejected", False, None, 0, 2),
    ]

    controls: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="qtail-droid-mirror-verifier-"
    ) as temporary:
        root = Path(temporary)
        for index, (
            name,
            expected_acceptance,
            mutation,
            checksum_returncode,
            expected_tfrecords,
        ) in enumerate(definitions):
            result = run_case(
                case_root=root / f"case-{index:02d}",
                python=args.python,
                verifier=args.verifier,
                expected_acceptance=expected_acceptance,
                mutation=mutation,
                checksum_returncode=checksum_returncode,
                expected_tfrecords=expected_tfrecords,
            )
            controls.append({"name": name, **result})

    controls_passed = sum(control["passed"] for control in controls)
    payload = {
        "generated_at": now(),
        "version": "qtail_droid_mirror_verifier_selftest_v1",
        "status": (
            "passed" if controls_passed == len(controls) else "failed"
        ),
        "verifier": str(args.verifier),
        "controls_passed": controls_passed,
        "controls_total": len(controls),
        "controls": controls,
        "claim_boundary": (
            "Tiny synthetic fixtures test the verifier's fail-closed "
            "behavior. They do not replace the formal 3.700 TB mirror audit."
        ),
    }
    write_json(args.out, payload)
    if payload["status"] != "passed":
        raise SystemExit("DROID mirror verifier controls failed")


if __name__ == "__main__":
    main()
