#!/usr/bin/env python3
"""Write and validate evidence-bound Open X expansion stage markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT_VERSION = "qtail_openx_1t_stage_marker_v1"
MARKERS = {
    "download": "OPENX_1T_DOWNLOAD_COMPLETE",
    "training": "OPENX_1T_TRAINING_COMPLETE",
    "synthesis": "OPENX_1T_SYNTHESIS_COMPLETE",
}
SYNTHESIS_ARTIFACTS = (
    "qtail_service_delivery_report.json",
    "qtail_service_synthetic_plan.csv",
    "qtail_synthetic_data.csv",
    "qtail_service_model_card.json",
    "qtail_data_engine_report.json",
    "README_QTAIL_DELIVERY.md",
    "qtail_delivery_package.zip",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required artifact missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def download_evidence(root: Path) -> dict[str, Any]:
    manifest_path = root / "openx_1t_object_manifest.json"
    checksum_manifest_path = root / "openx_1t_checksum_manifest.json"
    ledger_path = root / "download_checksum_ledger.json"
    verification_path = root / "download_verification.json"
    manifest = read_json(manifest_path)
    verification = read_json(verification_path)
    ledger = read_json(ledger_path)
    expected_objects = int(manifest.get("object_count", 0))
    expected_bytes = int(manifest.get("total_bytes", 0))
    objects = ledger.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("checksum ledger objects must be a mapping")
    valid_ledger_objects = sum(
        1
        for value in objects.values()
        if isinstance(value, dict)
        and value.get("official_md5_base64")
        and value.get("official_md5_base64") == value.get("local_md5_base64")
    )
    checks = (
        verification.get("status") == "verified",
        int(verification.get("expected_objects", 0)) == expected_objects,
        int(verification.get("complete_objects", 0)) == expected_objects,
        int(verification.get("md5_verified_objects", 0)) == expected_objects,
        int(verification.get("expected_bytes", 0)) == expected_bytes,
        int(verification.get("complete_bytes", 0)) == expected_bytes,
        valid_ledger_objects == expected_objects,
        not verification.get("missing"),
        not verification.get("size_mismatch"),
        not verification.get("ledger_mismatch"),
        not verification.get("partials"),
    )
    if not all(checks):
        raise ValueError(
            "download evidence incomplete: "
            f"verification={verification.get('status')} "
            f"objects={verification.get('md5_verified_objects')}/{expected_objects} "
            f"ledger={valid_ledger_objects}/{expected_objects}"
        )
    return {
        "expected_objects": expected_objects,
        "expected_bytes": expected_bytes,
        "valid_ledger_objects": valid_ledger_objects,
        "artifacts": {
            "object_manifest": artifact(manifest_path),
            "checksum_manifest": artifact(checksum_manifest_path),
            "checksum_ledger": artifact(ledger_path),
            "verification": artifact(verification_path),
        },
    }


def training_evidence(root: Path) -> dict[str, Any]:
    download_parent = marker_status(root, "download")
    if not download_parent.get("valid"):
        raise ValueError(
            f"download parent marker invalid: {download_parent.get('error')}"
        )
    training_root = root / "training"
    runtime_path = training_root / "training_runtime_status.json"
    report_path = training_root / "openx_demo_training_report.json"
    checkpoint_path = training_root / "qtail_allocation_head.pt"
    rows_path = training_root / "openx_shard_training_rows.csv"
    cache_usage_path = training_root / "feature_cache_usage.json"
    optimizer_progress_path = training_root / "optimizer_progress.json"
    source_resume_path = training_root / "resume_checkpoints" / "source.pt"
    qtail_resume_path = training_root / "resume_checkpoints" / "qtail.pt"
    runtime = read_json(runtime_path)
    report = read_json(report_path)
    optimizer_progress = read_json(optimizer_progress_path)
    expected_checkpoint_sha = str(
        report.get("model_artifact", {}).get("sha256") or ""
    )
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if not (
        runtime.get("status") == "complete"
        and int(runtime.get("returncode", -1)) == 0
        and report.get("status") == "complete"
        and int(report.get("steps", 0)) == 20000
        and optimizer_progress.get("status") == "phase_complete"
        and optimizer_progress.get("phase") == "qtail"
        and int(optimizer_progress.get("step", 0)) == 20000
        and int(optimizer_progress.get("steps_target", 0)) == 20000
        and int(optimizer_progress.get("overall_completed_updates", 0)) == 40000
        and int(optimizer_progress.get("overall_target_updates", 0)) == 40000
        and expected_checkpoint_sha
        and actual_checkpoint_sha == expected_checkpoint_sha
    ):
        raise ValueError(
            "training evidence incomplete: "
            f"runtime={runtime.get('status')} rc={runtime.get('returncode')} "
            f"report={report.get('status')} steps={report.get('steps')} "
            f"optimizer={optimizer_progress.get('phase')} "
            f"{optimizer_progress.get('overall_completed_updates')}/40000 "
            f"checkpoint_match={actual_checkpoint_sha == expected_checkpoint_sha}"
        )
    return {
        "steps": 20000,
        "optimizer_updates": 40000,
        "shard_count": int(report.get("shard_count", 0)),
        "checkpoint_sha256": actual_checkpoint_sha,
        "parent_download_marker": artifact(root / MARKERS["download"]),
        "artifacts": {
            "runtime": artifact(runtime_path),
            "report": artifact(report_path),
            "checkpoint": artifact(checkpoint_path),
            "training_rows": artifact(rows_path),
            "cache_usage": artifact(cache_usage_path),
            "optimizer_progress": artifact(optimizer_progress_path),
            "source_resume_checkpoint": artifact(source_resume_path),
            "qtail_resume_checkpoint": artifact(qtail_resume_path),
        },
    }


def synthesis_evidence(root: Path) -> dict[str, Any]:
    training_parent = marker_status(root, "training")
    if not training_parent.get("valid"):
        raise ValueError(
            f"training parent marker invalid: {training_parent.get('error')}"
        )
    synthesis_root = root / "synthesis"
    customer_input_path = Path(
        "/Users/avalok/work/Q-TAIL-MVP/data/customer_semifinal_embodied_tasks.csv"
    )
    runtime_path = synthesis_root / "synthesis_runtime_status.json"
    report_path = synthesis_root / "qtail_service_delivery_report.json"
    runtime = read_json(runtime_path)
    report = read_json(report_path)
    validation = report.get("customer_package", {}).get("validation", {})
    if not (
        runtime.get("status") == "complete"
        and int(runtime.get("returncode", -1)) == 0
        and validation.get("valid") is True
        and validation.get("winner") == "qtail_synthetic"
    ):
        raise ValueError(
            "synthesis evidence incomplete: "
            f"runtime={runtime.get('status')} rc={runtime.get('returncode')} "
            f"valid={validation.get('valid')} winner={validation.get('winner')}"
        )
    artifacts = {
        name: artifact(synthesis_root / name) for name in SYNTHESIS_ARTIFACTS
    }
    return {
        "validation_valid": True,
        "winner": "qtail_synthetic",
        "artifact_count": len(artifacts),
        "parent_training_marker": artifact(root / MARKERS["training"]),
        "customer_input": artifact(customer_input_path),
        "artifacts": artifacts,
    }


def build_evidence(root: Path, stage: str) -> dict[str, Any]:
    if stage == "download":
        return download_evidence(root)
    if stage == "training":
        return training_evidence(root)
    if stage == "synthesis":
        return synthesis_evidence(root)
    raise ValueError(f"unknown stage: {stage}")


def marker_status(root: Path, stage: str) -> dict[str, Any]:
    marker_path = root / MARKERS[stage]
    if not marker_path.is_file():
        return {"valid": False, "completed_at": None, "error": "marker missing"}
    try:
        marker = read_json(marker_path)
        evidence = build_evidence(root, stage)
        valid = bool(
            marker.get("format_version") == FORMAT_VERSION
            and marker.get("stage") == stage
            and marker.get("evidence") == evidence
        )
        return {
            "valid": valid,
            "completed_at": marker.get("completed_at") if valid else None,
            "error": None if valid else "marker evidence mismatch",
            "evidence": evidence,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "completed_at": None, "error": str(error)}


def write_marker(root: Path, stage: str) -> dict[str, Any]:
    evidence = build_evidence(root, stage)
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "completed_at": now(),
        "evidence": evidence,
    }
    path = root / MARKERS[stage]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "write"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MARKERS), required=True)
    args = parser.parse_args()
    if args.action == "write":
        payload = write_marker(args.root, args.stage)
        print(json.dumps({"valid": True, **payload}, ensure_ascii=False))
        return
    status = marker_status(args.root, args.stage)
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status["valid"] else 1)


if __name__ == "__main__":
    main()
