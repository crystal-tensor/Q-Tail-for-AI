#!/usr/bin/env python3
"""Audit and summarize a bounded DROID scalability/resume canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_copy(source: Path, destination: Path) -> None:
    require(source.is_file(), f"Missing evidence source: {source}")
    require(not source.is_symlink(), f"Evidence source must not be a symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_hash_ledger(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        require(
            bool(separator) and len(digest) == 64 and bool(name),
            f"Malformed hash ledger line in {path}: {line!r}",
        )
        entries.append((digest, name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-hashes-before", type=Path, required=True)
    parser.add_argument("--checkpoint-hashes-after", type=Path, required=True)
    parser.add_argument("--model-hash-before", type=Path, required=True)
    parser.add_argument("--model-hash-after", type=Path, required=True)
    parser.add_argument("--shard-list", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help=(
            "Optional formal-result root where immutable bounded-evidence "
            "copies are sealed. This does not promote the canary to formal."
        ),
    )
    args = parser.parse_args()

    report = read_json(args.report)
    require(report.get("status") == "complete", "Canary report is incomplete.")
    require(
        report.get("training_scope") == "bounded_test_subset",
        "Canary report is not explicitly bounded.",
    )
    require(
        report.get("formal_protocol", {}).get("locked") is False,
        "Canary unexpectedly claims the formal protocol.",
    )
    require(
        report.get("input_audit", {}).get("verified") is False,
        "Canary must not claim a verified full mirror.",
    )
    trajectory = report["trajectory_evidence"]
    shard_list = read_json(args.shard_list)
    require(
        shard_list.get("version") == "qtail_bounded_droid_shard_list_v1"
        and int(shard_list["shard_count"]) == args.expected_shards
        and int(shard_list["records_decoded"]) == args.expected_records,
        "Frozen shard-list count contract failed.",
    )
    run_manifest_path = Path(report["artifacts"]["run_manifest"])
    run_manifest = read_json(run_manifest_path)
    require(
        run_manifest.get("bounded_shard_list") == str(args.shard_list)
        and run_manifest.get("source_shard_paths_sha256")
        == shard_list.get("relative_paths_sha256"),
        "Training run is not bound to the frozen shard-list membership.",
    )
    require(
        int(report["shard_count"]) == args.expected_shards,
        "Canary shard count changed.",
    )
    require(
        int(trajectory["records_decoded"]) == args.expected_records,
        "Canary decoded-record count changed.",
    )
    require(
        trajectory.get("full_record_mode") is True
        and float(trajectory["record_parse_rate"]) == 1.0
        and float(trajectory["record_scan_complete_rate"]) == 1.0,
        "Canary did not fully decode every selected shard.",
    )

    releases = {
        str(item["release"]): item
        for item in report["release_composition"]
    }
    require(
        releases.get("1.0.0", {}).get("full_record_count_match") is True,
        "DROID 1.0.0 closure is missing.",
    )
    require(
        releases.get("1.0.1", {}).get("full_record_count_match") is False,
        "Canary boundary no longer reflects an incomplete DROID 1.0.1.",
    )

    compute = report["compute_audit"]
    require(
        compute.get("same_parameter_count") is True
        and compute.get("same_features") is True
        and compute.get("same_device") is True
        and compute.get("same_seed") is True
        and int(compute["source_optimizer_updates"])
        == int(compute["qtail_optimizer_updates"]),
        "Source/Q-Tail compute contract failed.",
    )
    resume = compute["resume"]
    require(
        set(resume)
        == {
            "evaluation_source",
            "evaluation_qtail",
            "deployment_source",
            "deployment_qtail",
        },
        "Canary resume stage set is incomplete.",
    )
    require(
        all(
            item.get("resumed") is True
            and int(item["resumed_from_step"]) == int(report["steps"])
            and not item.get("resume_rejections")
            for item in resume.values()
        ),
        "One or more canary stages did not resume cleanly.",
    )

    checkpoint_audit = report["intermediate_checkpoint_audit"]
    checkpoint_manifest_path = Path(checkpoint_audit["manifest"])
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    require(
        checkpoint_manifest.get("status") == "complete"
        and not checkpoint_manifest.get("errors")
        and int(checkpoint_manifest["actual_checkpoint_count"]) == 20
        and checkpoint_manifest["contract"][
            "parent_checkpoint_hash_chains_verified"
        ]
        is True,
        "Checkpoint manifest contract failed.",
    )
    for entry in checkpoint_manifest["entries"]:
        path = Path(entry["path"])
        require(path.is_file(), f"Missing checkpoint: {path}")
        require(
            sha256(path) == entry["sha256"],
            f"Checkpoint hash mismatch: {path}",
        )

    before_checkpoints = read_hash_ledger(
        args.checkpoint_hashes_before
    )
    after_checkpoints = read_hash_ledger(args.checkpoint_hashes_after)
    before_model = read_hash_ledger(args.model_hash_before)
    after_model = read_hash_ledger(args.model_hash_after)
    require(
        before_checkpoints == after_checkpoints
        and len(before_checkpoints) == 20,
        "Checkpoint bytes changed after resume replay.",
    )
    require(
        before_model == after_model and len(before_model) == 1,
        "Final model bytes changed after resume replay.",
    )

    effect = report["effect_metrics"]
    bootstrap = effect["paired_bootstrap"]
    coverage = report["rare_instruction_fingerprint_coverage"]
    curve = list(coverage["curve"])
    sealed_artifacts: dict[str, dict[str, Any]] = {}
    if args.evidence_root:
        evidence_root = args.evidence_root.resolve()
        require(
            args.out.parent.resolve() == evidence_root,
            "--out must be directly inside --evidence-root.",
        )
        sealed_report = (
            evidence_root / "droid_scalability_canary_full_report.json"
        )
        sealed_shard_list = (
            evidence_root
            / "droid_scalability_canary_frozen_shard_list.json"
        )
        atomic_copy(args.report, sealed_report)
        atomic_copy(args.shard_list, sealed_shard_list)
        sealed_artifacts = {
            "sealed_training_report": artifact(sealed_report),
            "sealed_frozen_shard_list": artifact(sealed_shard_list),
        }
    payload = {
        "generated_at": report["generated_at"],
        "status": "passed_bounded_scalability_canary",
        "evidence_role": "engineering_only_not_formal_effect_evidence",
        "formal_result_publishable": False,
        "claim_boundary": [
            (
                f"This canary uses {int(report['shard_count']):,} frozen "
                f"shards and {int(trajectory['records_decoded']):,} decoded "
                "DROID records."
            ),
            (
                "DROID 1.0.0 is complete, but DROID 1.0.1 and the official "
                "4,096-shard mirror are incomplete."
            ),
            (
                "The run validates feature-cache scalability, equal-compute "
                "training, checkpoint chains, and deterministic resume."
            ),
            (
                "Observed allocation metrics are retained for diagnosis but "
                "cannot be promoted to the final full-data claim."
            ),
            (
                "Allocation-head evidence is not end-to-end robot-policy "
                "tail success."
            ),
        ],
        "input": {
            "shards": int(report["shard_count"]),
            "records_decoded": int(trajectory["records_decoded"]),
            "represented_bytes": int(report["total_bytes"]),
            "releases": report["datasets"],
            "droid_1_0_0_complete": True,
            "droid_1_0_1_complete": False,
            "full_record_mode": True,
            "parse_rate": float(trajectory["record_parse_rate"]),
        },
        "equal_compute": {
            "device": compute["training_device"],
            "architecture": compute["architecture"],
            "optimizer": compute["same_optimizer"],
            "steps_per_stage": int(report["steps"]),
            "source_optimizer_updates": int(
                compute["source_optimizer_updates"]
            ),
            "qtail_optimizer_updates": int(
                compute["qtail_optimizer_updates"]
            ),
            "same_parameter_count": compute["same_parameter_count"],
        },
        "checkpoint_resume": {
            "stage_count": len(resume),
            "resumed_stage_count": sum(
                item["resumed"] is True for item in resume.values()
            ),
            "checkpoint_count": len(before_checkpoints),
            "checkpoint_hashes_stable": True,
            "final_model_hash_stable": True,
            "parent_hash_chains_verified": True,
        },
        "observed_allocation": {
            "role": "diagnostic_only",
            "source_tail_share": float(
                effect["source_pred_tail_share"]
            ),
            "qtail_tail_share": float(
                effect["qtail_pred_tail_share"]
            ),
            "gain_pp": float(effect["predicted_tail_share_gain_pp"]),
            "ci95_pp": [
                float(bootstrap["ci95_low_pp"]),
                float(bootstrap["ci95_high_pp"]),
            ],
            "extreme_underallocation_reduction_pp": float(
                effect["extreme_underallocation_reduction_pp"]
            ),
        },
        "rare_instruction_fingerprint": {
            "role": "diagnostic_only_not_semantic_coverage",
            "minimum_gain_pp": min(float(item["gain_pp"]) for item in curve),
            "maximum_gain_pp": max(float(item["gain_pp"]) for item in curve),
            "observed_direction": "mixed",
            "negative_at_larger_budgets": any(
                int(item["draw_budget"]) >= 200
                and float(item["gain_pp"]) < 0.0
                for item in curve
            ),
        },
        "artifacts": {
            "training_report": artifact(args.report),
            "checkpoint_manifest": artifact(checkpoint_manifest_path),
            "checkpoint_hashes_before": artifact(
                args.checkpoint_hashes_before
            ),
            "checkpoint_hashes_after": artifact(
                args.checkpoint_hashes_after
            ),
            "model_hash_before": artifact(args.model_hash_before),
            "model_hash_after": artifact(args.model_hash_after),
            "frozen_shard_list": artifact(args.shard_list),
            "run_manifest": artifact(run_manifest_path),
            **sealed_artifacts,
        },
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
