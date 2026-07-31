#!/usr/bin/env python3
"""Run positive and destructive negative controls for incremental closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument("--checksum-ledger", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--record-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = {
        "checksum_manifest": args.checksum_manifest,
        "checksum_ledger": args.checksum_ledger,
        "cache_manifest": args.cache_manifest,
        "record_audit": args.record_audit,
    }

    def run_case(
        *,
        name: str,
        paths: dict[str, Path],
        expected_success: bool,
        expected_failed_check: str | None,
        directory: Path,
        expected_deferred_count: int | None = None,
        expected_formal_gate: bool | None = None,
        require_formal: bool = False,
    ) -> dict[str, Any]:
        output = directory / f"{name}.json"
        command = [
            str(args.python),
            str(args.auditor),
            "--data-dir",
            str(args.data_dir),
            "--checksum-manifest",
            str(paths["checksum_manifest"]),
            "--checksum-ledger",
            str(paths["checksum_ledger"]),
            "--cache-manifest",
            str(paths["cache_manifest"]),
            "--record-audit",
            str(paths["record_audit"]),
            "--out",
            str(output),
        ]
        if require_formal:
            command.append("--require-formal")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        artifact = read_json(output) if output.is_file() else {}
        failed_checks = artifact.get("failed_checks", [])
        deferred_count = int(
            artifact.get("deferred_after_snapshot_count", -1)
        )
        formal_gate = artifact.get("formal_full_mirror_gate")
        passed = bool(
            (completed.returncode == 0) == expected_success
            and (
                expected_failed_check is None
                or expected_failed_check in failed_checks
            )
            and (
                expected_deferred_count is None
                or deferred_count == expected_deferred_count
            )
            and (
                expected_formal_gate is None
                or formal_gate is expected_formal_gate
            )
        )
        return {
            "name": name,
            "passed": passed,
            "expected_success": expected_success,
            "returncode": completed.returncode,
            "artifact_status": artifact.get("status"),
            "expected_failed_check": expected_failed_check,
            "failed_checks": failed_checks,
            "deferred_after_snapshot_count": deferred_count,
            "expected_deferred_after_snapshot_count": (
                expected_deferred_count
            ),
            "formal_full_mirror_gate": formal_gate,
            "expected_formal_full_mirror_gate": expected_formal_gate,
            "stderr": completed.stderr.strip(),
        }

    with tempfile.TemporaryDirectory(
        prefix="qtail-droid-closure-selftest-"
    ) as raw_directory:
        directory = Path(raw_directory)
        frozen_base: dict[str, Path] = {}
        source_artifacts: dict[str, dict[str, Any]] = {}
        for name, path in base.items():
            source_bytes = path.read_bytes()
            frozen_path = directory / f"base-{name}.json"
            frozen_path.write_bytes(source_bytes)
            frozen_base[name] = frozen_path
            source_artifacts[name] = {
                "path": str(path),
                "snapshot_sha256": hashlib.sha256(
                    source_bytes
                ).hexdigest(),
            }

        cases = [
            run_case(
                name="positive_current_closure",
                paths=frozen_base,
                expected_success=True,
                expected_failed_check=None,
                directory=directory,
            )
        ]
        baseline_deferred_count = int(
            cases[0]["deferred_after_snapshot_count"]
        )
        baseline_formal_gate = bool(
            cases[0]["formal_full_mirror_gate"]
        )
        cases.append(
            run_case(
                name="require_formal_matches_exact_full_gate",
                paths=frozen_base,
                expected_success=baseline_formal_gate,
                expected_failed_check=None,
                expected_formal_gate=baseline_formal_gate,
                require_formal=True,
                directory=directory,
            )
        )

        bad_records = read_json(frozen_base["record_audit"])
        bad_records["verified_decoded_records"] = (
            int(bad_records.get("verified_decoded_records", 0)) + 1
        )
        bad_records_path = directory / "bad-record-audit.json"
        write_json(bad_records_path, bad_records)
        cases.append(
            run_case(
                name="record_count_tamper_rejected",
                paths={
                    **frozen_base,
                    "record_audit": bad_records_path,
                },
                expected_success=False,
                expected_failed_check=(
                    "record_audit_decoded_records_match"
                ),
                directory=directory,
            )
        )

        bad_ledger = read_json(frozen_base["checksum_ledger"])
        first_key = sorted(bad_ledger.get("objects", {}))[0]
        bad_ledger["objects"][first_key]["local_md5_base64"] = "tampered"
        bad_ledger_path = directory / "bad-checksum-ledger.json"
        write_json(bad_ledger_path, bad_ledger)
        cases.append(
            run_case(
                name="md5_ledger_tamper_rejected",
                paths={
                    **frozen_base,
                    "checksum_ledger": bad_ledger_path,
                },
                expected_success=False,
                expected_failed_check=(
                    "ledger_entries_match_official_md5_and_live_files"
                ),
                directory=directory,
            )
        )

        late_bad_ledger = read_json(frozen_base["checksum_ledger"])
        late_key = sorted(late_bad_ledger.get("objects", {}))[-1]
        late_bad_ledger["objects"][late_key]["local_md5_base64"] = "tampered"
        late_bad_ledger_path = directory / "late-bad-checksum-ledger.json"
        write_json(late_bad_ledger_path, late_bad_ledger)
        noisy_cache = read_json(frozen_base["cache_manifest"])
        noisy_cache["artifacts"] = [
            {
                "path": str(directory / f"missing-cache-{index:03d}.json"),
                "bytes": 0,
                "sha256": "0" * 64,
            }
            for index in range(205)
        ] + noisy_cache.get("artifacts", [])
        noisy_cache_path = directory / "noisy-cache-manifest.json"
        write_json(noisy_cache_path, noisy_cache)
        cases.append(
            run_case(
                name="md5_after_error_sample_limit_rejected",
                paths={
                    **frozen_base,
                    "checksum_ledger": late_bad_ledger_path,
                    "cache_manifest": noisy_cache_path,
                },
                expected_success=False,
                expected_failed_check=(
                    "ledger_entries_match_official_md5_and_live_files"
                ),
                directory=directory,
            )
        )

        bad_cache = read_json(frozen_base["cache_manifest"])
        bad_cache["artifacts"] = bad_cache.get("artifacts", [])[1:]
        bad_cache["cache_count"] = len(bad_cache["artifacts"])
        bad_cache_path = directory / "bad-cache-manifest.json"
        write_json(bad_cache_path, bad_cache)
        cases.append(
            run_case(
                name="missing_listed_cache_rejected",
                paths={
                    **frozen_base,
                    "cache_manifest": bad_cache_path,
                },
                expected_success=False,
                expected_failed_check=(
                    "every_completed_tfrecord_has_one_listed_cache"
                ),
                directory=directory,
            )
        )

        deferred_cache = read_json(frozen_base["cache_manifest"])
        deferred_artifact = deferred_cache["artifacts"].pop(0)
        deferred_payload = read_json(Path(deferred_artifact["path"]))
        deferred_identity = deferred_payload["identity"]
        deferred_row = deferred_payload["row"]
        deferred_relative = str(deferred_identity["relative_path"])
        deferred_records = int(deferred_row["records_decoded"])
        deferred_source_bytes = int(deferred_identity["bytes"])
        deferred_release = Path(deferred_relative).parts[0]
        deferred_cache["cache_count"] = len(deferred_cache["artifacts"])
        deferred_cache["expected_shard_count"] = len(
            deferred_cache["artifacts"]
        )
        deferred_cache["represented_bytes"] = (
            int(deferred_cache["represented_bytes"])
            - deferred_source_bytes
        )
        deferred_cache["source_shard_count"] = max(
            0, int(deferred_cache.get("source_shard_count", 1)) - 1
        )
        deferred_cache_path = directory / "deferred-cache-manifest.json"
        write_json(deferred_cache_path, deferred_cache)

        deferred_record_audit = read_json(frozen_base["record_audit"])
        deferred_record_audit["verified_cache_count"] = (
            int(deferred_record_audit["verified_cache_count"]) - 1
        )
        deferred_record_audit["verified_decoded_records"] = (
            int(deferred_record_audit["verified_decoded_records"])
            - deferred_records
        )
        deferred_record_audit["represented_bytes"] = (
            int(deferred_record_audit["represented_bytes"])
            - deferred_source_bytes
        )
        for release_row in deferred_record_audit.get(
            "release_record_audit", []
        ):
            if release_row.get("release") != deferred_release:
                continue
            release_row["verified_cache_count"] = (
                int(release_row["verified_cache_count"]) - 1
            )
            release_row["verified_expected_records"] = (
                int(release_row["verified_expected_records"])
                - deferred_records
            )
            release_row["verified_decoded_records"] = (
                int(release_row["verified_decoded_records"])
                - deferred_records
            )
        deferred_record_audit_path = (
            directory / "deferred-record-audit.json"
        )
        write_json(deferred_record_audit_path, deferred_record_audit)

        deferred_ledger = read_json(frozen_base["checksum_ledger"])
        snapshot_at = datetime.fromisoformat(
            str(deferred_cache["source_snapshot_at"]).replace("Z", "+00:00")
        )
        deferred_ledger["objects"][deferred_relative]["verified_at"] = (
            snapshot_at + timedelta(seconds=1)
        ).isoformat()
        deferred_ledger_path = directory / "deferred-checksum-ledger.json"
        write_json(deferred_ledger_path, deferred_ledger)
        cases.append(
            run_case(
                name="post_snapshot_tfrecord_is_deferred",
                paths={
                    **frozen_base,
                    "checksum_ledger": deferred_ledger_path,
                    "cache_manifest": deferred_cache_path,
                    "record_audit": deferred_record_audit_path,
                },
                expected_success=True,
                expected_failed_check=None,
                expected_deferred_count=baseline_deferred_count + 1,
                expected_formal_gate=False,
                directory=directory,
            )
        )

    checks = {case["name"]: case["passed"] for case in cases}
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "format_version": "qtail_droid_incremental_closure_selftest_v2",
        "generated_at": now(),
        "status": "passed" if not failed else "failed",
        "claim_boundary": (
            "These controls validate rejection behavior and post-snapshot "
            "deferral semantics of the incremental closure auditor. They do "
            "not prove full mirror completion or model quality."
        ),
        "checks": checks,
        "failed_checks": failed,
        "cases": cases,
        "source_artifacts": source_artifacts,
        "auditor": {
            "path": str(args.auditor),
            "sha256": sha256(args.auditor),
        },
    }
    atomic_write_json(args.out, payload)
    if failed:
        raise SystemExit("closure self-test failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
