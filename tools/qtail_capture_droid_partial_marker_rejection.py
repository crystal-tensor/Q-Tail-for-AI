#!/usr/bin/env python3
"""Capture a real partial-mirror rejection by the DROID marker gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--transport-status", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    transport = read_json(args.transport_status)
    manifest = read_json(args.manifest)
    ledger = read_json(args.checksum_ledger)
    expected_objects = int(manifest.get("object_count", -1))
    completed_objects = int(transport.get("completed_objects", -1))
    verified_objects = len(ledger.get("objects", {}))
    partial_precondition = bool(
        expected_objects == 4_102
        and 0 <= completed_objects < expected_objects
        and 0 <= verified_objects < expected_objects
        and transport.get("status") != "complete"
    )
    if not partial_precondition:
        raise SystemExit(
            "live partial-mirror control requires an incomplete 4,102-object "
            "transport and ledger"
        )

    with tempfile.TemporaryDirectory(
        prefix="qtail-live-partial-marker-"
    ) as temporary:
        marker = Path(temporary) / "DROID_DOWNLOAD_COMPLETE"
        command = [
            str(args.python),
            str(args.verifier),
            "--data-dir",
            str(args.data_dir),
            "--manifest",
            str(args.manifest),
            "--checksum-manifest",
            str(args.checksum_manifest),
            "--checksum-ledger",
            str(args.checksum_ledger),
            "--transport-status",
            str(args.transport_status),
            "--marker",
            str(marker),
            "--expected-bytes",
            str(args.expected_bytes),
            "--write",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        marker_created = marker.exists()

    passed = result.returncode != 0 and not marker_created
    payload = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "control": "real_partial_mirror_cannot_commit_download_marker",
        "scope": "live_official_droid_partial_mirror",
        "formal_completion_evidence": False,
        "precondition": {
            "passed": partial_precondition,
            "manifest_objects": expected_objects,
            "completed_objects": completed_objects,
            "checksum_verified_objects": verified_objects,
            "transport_status": transport.get("status"),
        },
        "result": {
            "verifier_returncode": result.returncode,
            "marker_created": marker_created,
            "rejected": result.returncode != 0,
            "stderr": result.stderr.strip(),
            "stdout": result.stdout.strip(),
        },
        "reproducibility": {
            "verifier": str(args.verifier),
            "verifier_sha256": sha256(args.verifier),
            "python": str(args.python),
            "data_dir": str(args.data_dir),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "checksum_manifest": str(args.checksum_manifest),
            "checksum_manifest_sha256": sha256(args.checksum_manifest),
            "checksum_ledger": str(args.checksum_ledger),
            "checksum_ledger_sha256_at_capture": sha256(
                args.checksum_ledger
            ),
            "transport_status": str(args.transport_status),
            "transport_status_sha256_at_capture": sha256(
                args.transport_status
            ),
            "command": command,
        },
        "claim_boundary": (
            "This proves the live partial official mirror could not open the "
            "download-completion gate. It is not evidence that the full mirror "
            "or formal training is complete."
        ),
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("live partial-mirror marker rejection failed")


if __name__ == "__main__":
    main()
