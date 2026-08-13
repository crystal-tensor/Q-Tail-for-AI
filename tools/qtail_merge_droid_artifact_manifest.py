#!/usr/bin/env python3
"""Atomically refresh and extend the DROID artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAL_STATIC_ARTIFACTS = (
    "droid_full_run_manifest.json",
    "droid_feature_extraction_status.json",
    "droid_shard_features.csv",
    "droid_feature_cache_manifest.json",
    "droid_incremental_closure_audit.json",
    "droid_incremental_closure_selftest.json",
    "droid_release_milestone_status.json",
    "droid_feature_cache_verification.json",
    "droid_shard_training_rows.csv",
    "droid_training_curve.csv",
    "droid_intermediate_checkpoint_manifest.json",
    "droid_rare_instruction_fingerprint_coverage.json",
    "droid_model_training_status.json",
    "qtail_droid_allocation_head.pt",
    "empirical_pt_source.csv",
    "droid_full_training_report.json",
    "training_status.json",
    "droid_environment_manifest.json",
    "droid_environment_contract_selftest.json",
    "droid_download_marker_selftest.json",
    "droid_mirror_verifier_selftest.json",
    "droid_downloader_single_writer_selftest.json",
    "droid_runtime_process_contract_selftest.json",
    "pipeline_generation_gate.json",
    "droid_stage_marker_hardening_selftest.json",
    "droid_progress_preview_selftest.json",
    "droid_artifact_manifest_merge_selftest.json",
    "droid_pipeline_shell_contract_selftest.json",
    "droid_training_gate_order_selftest.json",
    "droid_timeline_monotonic_selftest.json",
    "uniclash_pre_checksum_gate.json",
    "uniclash_pre_checksum_gate_selftest.json",
    "uniclash_checksum_handoff_gate.json",
    "uniclash_pre_environment_gate.json",
    "uniclash_pre_training_gate.json",
    "droid_object_manifest.json",
    "droid_release_metadata_audit.json",
    "uniclash_transport_guard_classifier_v6_selftest.json",
    "transport_epochs/uniclash_transport_guard_v4_core_restart_pause.json",
    "transport_epochs/uniclash_transport_guard_v5_interface_migration_pause.json",
    "download_verification.json",
    "droid_protocol_selftest.json",
)
HISTORICAL_OPTIONAL_ARTIFACTS = (
    "droid_checksum_stat_continuity.json",
    "live_page_smoke.json",
    "droid_live_partial_marker_rejection.json",
    "droid_transport_tuning_audit.json",
    "uniclash_transport_guard_v3_encoded_path_underobservation.json",
    "droid_preflight_training_smoke.json",
    "droid_preflight_training_smoke_report.json",
    "droid_forecast_908_summary.json",
    "droid_scalability_canary_summary.json",
    "droid_scalability_canary_full_report.json",
    "droid_scalability_canary_frozen_shard_list.json",
    "droid_shard_list_selftest.json",
)
FORMAL_CHECKPOINT_LABELS = (
    "evaluation_source",
    "evaluation_qtail",
    "deployment_source",
    "deployment_qtail",
)
FORMAL_CHECKPOINT_STEPS = (0, 5_000, 10_000, 15_000, 20_000)
MANIFEST_CONTROL_FILENAMES = {
    "droid_artifact_manifest.json",
    "droid_training_artifact_manifest.json",
    "qtail_orchestration_snapshot_sync_audit.json",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def formal_droid_artifact_paths(result_root: Path) -> list[Path]:
    paths = [result_root / name for name in FORMAL_STATIC_ARTIFACTS]
    paths.extend(
        result_root
        / "intermediate_checkpoints"
        / f"{label}_step_{step:06d}.pt"
        for label in FORMAL_CHECKPOINT_LABELS
        for step in FORMAL_CHECKPOINT_STEPS
    )
    paths.extend(
        result_root
        / "release_milestones"
        / f"droid_release_{release}_complete.json"
        for release in ("1.0.0", "1.0.1")
    )
    return paths


def path_set_sha256(paths: set[Path]) -> str:
    canonical = "\n".join(sorted(str(path) for path in paths)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_formal_artifact_path(path: Path, formal_root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"formal artifact must not be a symlink: {path}")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"formal artifact is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"formal artifact must be a regular file: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(formal_root)
    except ValueError as error:
        raise ValueError(
            f"formal artifact escapes formal root {formal_root}: {path}"
        ) from error
    return resolved


def validate_existing_formal_contract(
    contract: Any,
    *,
    formal_root: Path,
    required_paths: set[Path],
    manifest_paths: list[Path],
) -> None:
    if not contract:
        return
    if not isinstance(contract, dict):
        raise ValueError("existing formal artifact contract is invalid")
    existing_root = contract.get("result_root")
    if existing_root and Path(str(existing_root)).resolve() != formal_root:
        raise ValueError("formal artifact root drift detected")
    existing_count = contract.get("required_artifact_count")
    if existing_count is not None:
        try:
            count_matches = int(existing_count) == len(required_paths)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "formal required artifact count is invalid"
            ) from error
        if not count_matches:
            raise ValueError("formal required artifact count drift detected")
    existing_paths = contract.get("required_artifact_paths")
    if existing_paths is not None:
        if not isinstance(existing_paths, list):
            raise ValueError("formal required artifact path set is invalid")
        observed = {Path(str(path)).resolve() for path in existing_paths}
        if observed != required_paths or len(existing_paths) != len(required_paths):
            raise ValueError("formal required artifact path set drift detected")
    existing_set_sha256 = contract.get("required_artifact_set_sha256")
    expected_set_sha256 = path_set_sha256(required_paths)
    if (
        existing_set_sha256 is not None
        and existing_set_sha256 != expected_set_sha256
    ):
        raise ValueError("formal required artifact set hash drift detected")
    manifest_path_set = {
        validate_formal_artifact_path(path, formal_root)
        for path in manifest_paths
    }
    if not required_paths.issubset(manifest_path_set):
        raise ValueError("formal required artifact manifest set drift detected")


def build_manifest_payload(
    *,
    manifest: dict[str, Any],
    additions: list[Path],
    formal_root: Path | None,
) -> dict[str, Any]:
    formal_root_resolved = formal_root.resolve() if formal_root else None
    existing_paths = [
        Path(str(entry.get("path", "")))
        for entry in manifest.get("artifacts", [])
        if isinstance(entry, dict) and str(entry.get("path", ""))
    ]
    formal_paths = (
        formal_droid_artifact_paths(formal_root_resolved)
        if formal_root_resolved
        else []
    )
    required_paths = (
        {
            validate_formal_artifact_path(path, formal_root_resolved)
            for path in formal_paths
        }
        if formal_root_resolved
        else set()
    )
    if formal_root_resolved and len(required_paths) != len(formal_paths):
        raise ValueError("formal required artifact path set contains aliases")
    if formal_root_resolved:
        validate_existing_formal_contract(
            manifest.get("formal_droid_contract"),
            formal_root=formal_root_resolved,
            required_paths=required_paths,
            manifest_paths=existing_paths,
        )
    optional_names = set(HISTORICAL_OPTIONAL_ARTIFACTS)
    retained_paths = [
        path
        for path in existing_paths
        if path.name not in MANIFEST_CONTROL_FILENAMES
        and (path.is_file() or path.name not in optional_names)
    ]
    optional_history = [
        formal_root_resolved / name
        for name in HISTORICAL_OPTIONAL_ARTIFACTS
        if formal_root_resolved
        and (formal_root_resolved / name).is_file()
    ]
    source_paths = [
        *retained_paths,
        *(
            path
            for path in additions
            if path.name not in MANIFEST_CONTROL_FILENAMES
        ),
        *formal_paths,
        *optional_history,
    ]
    if formal_root_resolved:
        paths = sorted(
            {
                validate_formal_artifact_path(path, formal_root_resolved)
                for path in source_paths
            },
            key=str,
        )
        observed_required_paths = required_paths.intersection(paths)
        if observed_required_paths != required_paths:
            raise ValueError("formal required artifact path set drift detected")
    else:
        paths = sorted({path.resolve() for path in source_paths}, key=str)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(
                "artifact manifest merge has missing inputs: "
                + ", ".join(missing[:20])
            )
    return {
        **manifest,
        "generated_at": now(),
        "status": "complete",
        "merge_contract": (
            "Every retained and newly required artifact is re-read and "
            "re-hashed immediately before the training marker is committed. "
            "Historical bounded/preflight evidence is retained when present "
            "but cannot permanently block a fresh formal run."
        ),
        "formal_droid_contract": (
            {
                "result_root": str(formal_root_resolved),
                "required_artifact_count": len(required_paths),
                "required_artifact_paths": [
                    str(path) for path in sorted(required_paths, key=str)
                ],
                "required_artifact_set_sha256": path_set_sha256(
                    required_paths
                ),
                "optional_history_retained": len(optional_history),
                "all_required_present": True,
            }
            if formal_root_resolved
            else None
        ),
        "artifacts": [artifact_entry(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--add", type=Path, action="append", default=[])
    parser.add_argument("--formal-droid-root", type=Path)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    try:
        payload = build_manifest_payload(
            manifest=manifest,
            additions=args.add,
            formal_root=args.formal_droid_root,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    atomic_write_json(args.manifest, payload)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "status": "complete",
                "artifact_count": len(payload["artifacts"]),
                "required_additions": (
                    len(args.add)
                    + int(
                        payload.get("formal_droid_contract", {}).get(
                            "required_artifact_count", 0
                        )
                        if payload.get("formal_droid_contract")
                        else 0
                    )
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
