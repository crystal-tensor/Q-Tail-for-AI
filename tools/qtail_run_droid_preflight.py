#!/usr/bin/env python3
"""Run and publish a current-generation bounded DROID training preflight.

The preflight uses eight locally MD5-verified official TFRecords, trains the
same Source/Q-Tail AllocationHead stages twice, and proves terminal checkpoint
resume. It is engineering evidence only and never writes formal markers.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASES = ("1.0.0", "1.0.1")
STAGES = (
    "evaluation_source",
    "evaluation_qtail",
    "deployment_source",
    "deployment_qtail",
)
FORBIDDEN_FORMAL_MARKERS = (
    "DROID_MODEL_TRAINING_STARTED",
    "DROID_MODEL_TRAINING_COMPLETE",
    "DROID_TRAINING_COMPLETE",
    "FINAL_PAGE_QA_COMPLETE",
)
CHECKPOINT_FORMAT_VERSION = 6
CHECKPOINT_CHAIN_VERSION = "sha256_parent_v1"
STEPS = 25
CHECKPOINT_EVERY_STEPS = 10
RECORDS_PER_SHARD = 2
SHARDS_PER_RELEASE = 4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        if sha256(source) != sha256(temporary):
            raise RuntimeError(f"atomic copy hash mismatch: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise SystemExit(f"preflight lock is already held by {owner}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({"pid": os.getpid(), "acquired_at": now()}) + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def marker_snapshot(marker_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in FORBIDDEN_FORMAL_MARKERS:
        path = marker_dir / name
        result[name] = {
            "exists": path.exists(),
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        }
    return result


def select_verified_shards(
    *,
    data_dir: Path,
    ledger_path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    objects = ledger.get("objects")
    if not isinstance(objects, dict):
        raise SystemExit("checksum ledger objects mapping is missing")
    selected: list[str] = []
    evidence: list[dict[str, Any]] = []
    for release in RELEASES:
        candidates = sorted(
            relative
            for relative, item in objects.items()
            if relative.startswith(f"{release}/")
            and "tfrecord" in Path(relative).name.lower()
            and isinstance(item, dict)
            and item.get("official_md5_base64")
            == item.get("local_md5_base64")
        )
        if len(candidates) < SHARDS_PER_RELEASE:
            raise SystemExit(
                f"only {len(candidates)} verified TFRecords for {release}"
            )
        for relative in candidates[:SHARDS_PER_RELEASE]:
            item = objects[relative]
            path = data_dir / relative
            if not path.is_file() or ".part" in path.name:
                raise SystemExit(f"selected shard is incomplete: {relative}")
            expected_bytes = int(item["bytes"])
            if path.stat().st_size != expected_bytes:
                raise SystemExit(f"selected shard size mismatch: {relative}")
            observed_md5 = md5_base64(path)
            if observed_md5 != item["official_md5_base64"]:
                raise SystemExit(f"selected shard MD5 mismatch: {relative}")
            selected.append(relative)
            evidence.append(
                {
                    "relative_path": relative,
                    "release": release,
                    "bytes": expected_bytes,
                    "official_md5_base64": item["official_md5_base64"],
                    "local_md5_base64_recomputed": observed_md5,
                    "ledger_generation": item.get("generation"),
                    "ledger_verified_at": item.get("verified_at"),
                }
            )
    selected.sort()
    evidence.sort(key=lambda item: item["relative_path"])
    return selected, evidence


def run_trainer(
    *,
    command: list[str],
    log_path: Path,
) -> float:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] COMMAND {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log.write(f"[{now()}] RETURN_CODE {completed.returncode}\n")
    if completed.returncode != 0:
        raise SystemExit(
            f"trainer failed with code {completed.returncode}; see {log_path}"
        )
    return time.monotonic() - started


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_report(
    *,
    report: dict[str, Any],
    total_bytes: int,
    expect_resumed: bool,
) -> None:
    require(report.get("status") == "complete", "report is not complete")
    require(
        report.get("training_scope") == "bounded_test_subset",
        "preflight escaped bounded scope",
    )
    require(
        report.get("formal_protocol", {}).get("locked") is False,
        "preflight entered the formal protocol",
    )
    require(int(report.get("shard_count", -1)) == 8, "shard count is not 8")
    require(
        int(report.get("total_bytes", -1)) == total_bytes,
        "selected byte count changed",
    )
    composition = {
        str(item.get("release")): item
        for item in report.get("release_composition", [])
    }
    require(set(composition) == set(RELEASES), "release composition mismatch")
    for release in RELEASES:
        item = composition[release]
        require(
            int(item.get("observed_tfrecord_shards", -1))
            == SHARDS_PER_RELEASE,
            f"{release} shard count mismatch",
        )
        require(
            int(item.get("observed_records_decoded", -1))
            == SHARDS_PER_RELEASE * RECORDS_PER_SHARD,
            f"{release} decoded record count mismatch",
        )
    holdout = report.get("holdout_evaluation", {})
    holdout_by_release = {
        str(item.get("release")): item
        for item in holdout.get("per_release", [])
    }
    require(
        holdout.get("membership_path_scope")
        == "official_release_relative_path"
        and holdout.get("holdout_membership_locked") is True,
        "holdout membership contract is not locked",
    )
    for release in RELEASES:
        item = holdout_by_release.get(release, {})
        require(
            int(item.get("total_shards", -1)) == SHARDS_PER_RELEASE
            and int(item.get("holdout_shards", -1)) == 1,
            f"{release} holdout contract mismatch",
        )
    compute = report.get("compute_audit", {})
    require(
        compute.get("training_device") == "mps"
        and compute.get("mps_available") is True,
        "preflight did not use MPS",
    )
    for key in (
        "same_architecture",
        "same_seed",
        "same_features",
        "same_device",
        "same_parameter_count",
        "same_environment_fingerprint",
    ):
        require(compute.get(key) is True, f"compute contract failed: {key}")
    require(
        int(compute.get("source_optimizer_updates", -1)) == STEPS * 2
        and int(compute.get("qtail_optimizer_updates", -1)) == STEPS * 2,
        "same-compute optimizer update count mismatch",
    )
    runtime_environment_fingerprint = str(
        compute.get("runtime_environment_fingerprint", "")
    )
    require(
        len(runtime_environment_fingerprint) == 64,
        "runtime environment fingerprint is missing",
    )
    checkpoint_environment_fingerprint = str(
        compute.get("checkpoint_environment_fingerprint", "")
    )
    require(
        len(checkpoint_environment_fingerprint) == 64,
        "checkpoint environment fingerprint is missing",
    )
    resumes = compute.get("resume", {})
    require(set(resumes) == set(STAGES), "training stage set mismatch")
    for stage in STAGES:
        item = resumes[stage]
        require(
            item.get("resumed") is expect_resumed,
            f"{stage} resume state mismatch",
        )
        require(
            int(item.get("resumed_from_step", -1))
            == (STEPS if expect_resumed else 0),
            f"{stage} resume step mismatch",
        )
        require(
            int(item.get("target_step", -1)) == STEPS
            and int(item.get("optimizer_updates_completed", -1)) == STEPS,
            f"{stage} optimizer update count mismatch",
        )
        require(
            item.get("device") == "mps"
            and item.get("checkpoint_device")
            in (None, "mps"),
            f"{stage} checkpoint device mismatch",
        )
        require(
            int(item.get("checkpoint_format_version", -1))
            == CHECKPOINT_FORMAT_VERSION
            and item.get("checkpoint_chain_version")
            == CHECKPOINT_CHAIN_VERSION,
            f"{stage} checkpoint generation mismatch",
        )
        require(
            item.get("environment_fingerprint")
            == checkpoint_environment_fingerprint,
            f"{stage} environment fingerprint mismatch",
        )
        require(
            not item.get("resume_rejections"),
            f"{stage} contains resume rejection events",
        )
    checkpoint = report.get("intermediate_checkpoint_audit", {})
    contract = checkpoint.get("contract", {})
    require(
        checkpoint.get("status") == "complete"
        and int(checkpoint.get("actual_checkpoint_count", -1)) == 16
        and int(contract.get("expected_checkpoint_count", -1)) == 16,
        "checkpoint manifest is incomplete",
    )
    require(
        contract.get("expected_steps") == [0, 10, 20, 25]
        and int(contract.get("checkpoint_format_version", -1))
        == CHECKPOINT_FORMAT_VERSION
        and contract.get("checkpoint_chain_version")
        == CHECKPOINT_CHAIN_VERSION
        and contract.get("parent_checkpoint_hash_chains_verified") is True
        and contract.get("checkpoint_content_hashes_recomputed") is True
        and contract.get("paired_feature_signatures_equal") is True
        and contract.get("initialized_state_signatures_equal") is True,
        "checkpoint chain contract failed",
    )


def checkpoint_hashes(run_root: Path) -> dict[str, str]:
    paths = sorted((run_root / "intermediate_checkpoints").glob("*.pt"))
    return {path.name: sha256(path) for path in paths}


def archive_previous(
    *,
    public_summary: Path,
    public_report: Path,
    history_root: Path,
) -> str | None:
    existing = [path for path in (public_summary, public_report) if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = history_root / stamp
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = history_root / f"{stamp}-{suffix}"
    destination.mkdir(parents=True)
    for source in existing:
        atomic_copy(source, destination / source.name)
    inventory = {
        "archived_at": now(),
        "files": {
            path.name: sha256(destination / path.name) for path in existing
        },
    }
    atomic_write_json(destination / "archive_inventory.json", inventory)
    return str(destination)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Volumes/ORICO/qtail_full_training/data/droid"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "/Volumes/ORICO/qtail_full_training/results/qtail_droid_full"
        ),
    )
    parser.add_argument(
        "--run-parent",
        type=Path,
        default=Path(
            "/Volumes/ORICO/qtail_full_training/results/"
            "qtail_droid_preflight_runs"
        ),
    )
    parser.add_argument(
        "--marker-dir",
        type=Path,
        default=Path("/Volumes/ORICO/qtail_full_training/manifests"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "/Volumes/ORICO/qtail_full_training/results/qtail_droid_full/"
            "droid_object_checksum_ledger.json"
        ),
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=repo_root / "tools" / "qtail_train_droid_full.py",
    )
    parser.add_argument(
        "--pt-source",
        type=Path,
        default=repo_root / "data" / "uploaded_data.csv",
    )
    args = parser.parse_args()

    if not os.path.ismount("/Volumes/ORICO"):
        raise SystemExit("ORICO is not mounted")
    lock = acquire_lock(args.result_root / ".droid_preflight_training.lock")
    try:
        selected, selected_evidence = select_verified_shards(
            data_dir=args.data_dir,
            ledger_path=args.ledger,
        )
        membership_sha256 = hashlib.sha256(
            "\n".join(selected).encode("utf-8")
        ).hexdigest()
        trainer_sha256 = sha256(args.trainer)
        generation = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{trainer_sha256[:12]}"
        )
        run_root = args.run_parent / generation
        if run_root.exists():
            raise SystemExit(f"preflight run root already exists: {run_root}")
        run_root.mkdir(parents=True)
        frozen_list = run_root / "frozen_verified_shards.json"
        atomic_write_json(
            frozen_list,
            {
                "version": "qtail_bounded_droid_shard_list_v2",
                "generated_at": now(),
                "selection": (
                    "lexicographically_first_four_ledger_verified_"
                    "tfrecords_per_release"
                ),
                "relative_paths": selected,
                "relative_paths_sha256": membership_sha256,
                "objects": selected_evidence,
            },
        )
        command = [
            sys.executable,
            str(args.trainer),
            "--data-dir",
            str(args.data_dir),
            "--out",
            str(run_root),
            "--shard-list",
            str(frozen_list),
            "--records-per-shard",
            str(RECORDS_PER_SHARD),
            "--min-shards",
            "8",
            "--steps",
            str(STEPS),
            "--checkpoint-every-steps",
            str(CHECKPOINT_EVERY_STEPS),
            "--bootstrap-samples",
            "50",
            "--holdout-fraction",
            "0.25",
            "--min-record-parse-rate",
            "1.0",
            "--min-record-scan-complete-rate",
            "1.0",
            "--status-every-shards",
            "1",
            "--required-mount",
            "/Volumes/ORICO",
            "--pt-source",
            str(args.pt_source),
            "--process-lock",
            str(run_root / ".qtail_train_droid_full.lock"),
        ]
        marker_before = marker_snapshot(args.marker_dir)
        first_seconds = run_trainer(
            command=command,
            log_path=run_root / "first_run.log",
        )
        report_path = run_root / "droid_full_training_report.json"
        first_report = load_json(report_path)
        total_bytes = sum(int(item["bytes"]) for item in selected_evidence)
        validate_report(
            report=first_report,
            total_bytes=total_bytes,
            expect_resumed=False,
        )
        first_report_copy = run_root / "first_run_report.json"
        atomic_copy(report_path, first_report_copy)
        first_checkpoint_hashes = checkpoint_hashes(run_root)
        require(
            len(first_checkpoint_hashes) == 16,
            "first run did not create 16 checkpoints",
        )
        first_final_checkpoint_hash = sha256(
            run_root / "qtail_droid_allocation_head.pt"
        )

        second_seconds = run_trainer(
            command=command,
            log_path=run_root / "resume_run.log",
        )
        resumed_report = load_json(report_path)
        validate_report(
            report=resumed_report,
            total_bytes=total_bytes,
            expect_resumed=True,
        )
        second_checkpoint_hashes = checkpoint_hashes(run_root)
        require(
            first_checkpoint_hashes == second_checkpoint_hashes,
            "terminal resume changed immutable checkpoint bytes",
        )
        marker_after = marker_snapshot(args.marker_dir)
        require(
            marker_before == marker_after,
            "bounded preflight changed a formal marker",
        )

        compute = resumed_report["compute_audit"]
        resume_stages = compute["resume"]
        intermediate_manifest = Path(
            resumed_report["artifacts"]["intermediate_checkpoint_manifest"]
        )
        checkpoint_path = Path(resumed_report["artifacts"]["checkpoint"])
        public_summary = (
            args.result_root / "droid_preflight_training_smoke.json"
        )
        public_report = (
            args.result_root / "droid_preflight_training_smoke_report.json"
        )
        history = archive_previous(
            public_summary=public_summary,
            public_report=public_report,
            history_root=args.result_root / "preflight_history",
        )
        atomic_copy(report_path, public_report)
        public_report_hash = sha256(public_report)
        summary = {
            "format_version": "qtail_droid_engineering_preflight_v2",
            "generated_at": now(),
            "status": "passed_engineering_preflight",
            "scope": "bounded_test_subset_not_scientific_evidence",
            "claim_boundary": (
                "Current-generation engineering preflight only: real "
                "ledger-verified DROID TFRecord I/O, MPS same-compute "
                "Source/Q-Tail execution, checkpoint v6 parent-chain "
                "validation, and terminal resume. It is not_scientific_evidence "
                "and is withheld from every formal DROID effect claim and "
                "completion marker."
            ),
            "input": {
                "shards": len(selected),
                "shards_per_release": SHARDS_PER_RELEASE,
                "releases": {
                    release: sum(
                        item["release"] == release
                        for item in selected_evidence
                    )
                    for release in RELEASES
                },
                "records_decoded": len(selected) * RECORDS_PER_SHARD,
                "record_cap_per_shard": RECORDS_PER_SHARD,
                "bytes": total_bytes,
                "all_local_md5_recomputed_match_official": True,
                "frozen_relative_paths_sha256": membership_sha256,
                "formal_protocol_locked": False,
            },
            "compute": {
                "device": compute["training_device"],
                "mps_available": compute["mps_available"],
                "architecture": compute["architecture"],
                "optimizer": compute["same_optimizer"],
                "runtime_environment_fingerprint": compute[
                    "runtime_environment_fingerprint"
                ],
                "checkpoint_environment_fingerprint": compute[
                    "checkpoint_environment_fingerprint"
                ],
                "same_environment_fingerprint": compute[
                    "same_environment_fingerprint"
                ],
                "same_architecture": compute["same_architecture"],
                "same_seed": compute["same_seed"],
                "same_features": compute["same_features"],
                "same_device": compute["same_device"],
                "same_parameter_count": compute["same_parameter_count"],
                "source_optimizer_updates": compute[
                    "source_optimizer_updates"
                ],
                "qtail_optimizer_updates": compute[
                    "qtail_optimizer_updates"
                ],
            },
            "resume": {
                "resumed_stage_count": sum(
                    item["resumed"] for item in resume_stages.values()
                ),
                "stage_count": len(resume_stages),
                "all_checkpoint_devices_match": all(
                    item["device"] == item["checkpoint_device"]
                    for item in resume_stages.values()
                ),
                "all_checkpoint_optimizers_match": all(
                    item["optimizer"] == item["checkpoint_optimizer"]
                    for item in resume_stages.values()
                ),
                "all_environment_fingerprints_match": all(
                    item["environment_fingerprint"]
                    == item["checkpoint_environment_fingerprint"]
                    == compute["checkpoint_environment_fingerprint"]
                    for item in resume_stages.values()
                ),
                "stages": resume_stages,
            },
            "checkpoint_chain": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "chain_version": CHECKPOINT_CHAIN_VERSION,
                "expected_steps": [0, 10, 20, 25],
                "expected_checkpoint_count": 16,
                "actual_checkpoint_count": len(second_checkpoint_hashes),
                "parent_hash_chains_verified": True,
                "terminal_resume_preserved_checkpoint_hashes": True,
                "checkpoint_hashes": second_checkpoint_hashes,
            },
            "formal_marker_isolation": {
                "marker_dir_argument_passed_to_trainer": False,
                "forbidden_markers": list(FORBIDDEN_FORMAL_MARKERS),
                "before": marker_before,
                "after": marker_after,
                "unchanged": True,
            },
            "scientific_gate": {
                "passed": False,
                "expected_for_completion": False,
                "disposition": (
                    "withheld_from_formal_claim_and_completion_markers"
                ),
            },
            "execution": {
                "run_root": str(run_root),
                "trainer_command": command,
                "first_run_seconds": first_seconds,
                "resume_run_seconds": second_seconds,
                "first_run_report": str(first_report_copy),
                "first_run_log": str(run_root / "first_run.log"),
                "resume_run_log": str(run_root / "resume_run.log"),
                "previous_public_evidence_archive": history,
            },
            "provenance": {
                "runner": str(Path(__file__).resolve()),
                "runner_sha256": sha256(Path(__file__).resolve()),
                "trainer": str(args.trainer.resolve()),
                "trainer_sha256": trainer_sha256,
                "pt_source": str(args.pt_source.resolve()),
                "pt_source_sha256": sha256(args.pt_source),
                "checksum_ledger": str(args.ledger.resolve()),
                "checksum_ledger_sha256_at_selection": sha256(args.ledger),
                "frozen_shard_list": str(frozen_list),
                "frozen_shard_list_sha256": sha256(frozen_list),
            },
            "artifacts": {
                "report_copy": str(public_report),
                "report_sha256": public_report_hash,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "first_run_checkpoint_sha256": first_final_checkpoint_hash,
                "intermediate_checkpoint_manifest": str(
                    intermediate_manifest
                ),
                "intermediate_checkpoint_manifest_sha256": sha256(
                    intermediate_manifest
                ),
                "intermediate_checkpoint_count": len(
                    second_checkpoint_hashes
                ),
                "first_run_log_sha256": sha256(
                    run_root / "first_run.log"
                ),
                "resume_run_log_sha256": sha256(
                    run_root / "resume_run.log"
                ),
            },
        }
        atomic_write_json(public_summary, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
