#!/usr/bin/env python3
"""Run deterministic positive/negative controls for the DROID protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tfrecord.writer import TFRecordWriter

import qtail_train_droid_full as protocol
import qtail_train_openx_demo as base


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release in ("1.0.0", "1.0.1"):
        for index in range(20):
            rows.append(
                {
                    "dataset": release,
                    "path": f"/fixture/{release}/part-{index:05d}-of-00020",
                    "bytes": 1_000_000 + index * 1_003,
                    "shard_idx": index,
                    "shard_total": 20,
                    "records_decoded": 30 + index % 5,
                    "mean_episode_steps": 100 + index * 3,
                    "mean_reward_max": (index % 7) / 7,
                    "mean_action_std": 0.2 + index / 100,
                    "mean_action_abs_mean": 0.3 + index / 200,
                    "mean_instruction_units": 10 + index % 4,
                    "mean_instruction_unique": 1 + index % 3,
                    "mean_terminal_rate": 0.01,
                    "instruction_hashes": [
                        f"instruction-{index % 9}",
                        f"release-{release}",
                    ],
                }
            )
    return rows


def stable_resume_audit(audit: dict[str, Any]) -> dict[str, Any]:
    checkpoint = audit.get("checkpoint")
    return {
        "resumed": audit.get("resumed"),
        "checkpoint_name": Path(checkpoint).name if checkpoint else None,
        "resumed_from_step": audit.get("resumed_from_step"),
        "target_step": audit.get("target_step"),
        "optimizer_updates_completed": audit.get(
            "optimizer_updates_completed"
        ),
        "device": audit.get("device"),
        "optimizer": audit.get("optimizer"),
        "checkpoint_device": audit.get("checkpoint_device"),
        "checkpoint_optimizer": audit.get("checkpoint_optimizer"),
        "environment_fingerprint": audit.get("environment_fingerprint"),
        "step_semantics": audit.get("step_semantics"),
        "training_signature": audit.get("training_signature"),
        "checkpoint_format_version": audit.get(
            "checkpoint_format_version"
        ),
        "checkpoint_chain_version": audit.get(
            "checkpoint_chain_version"
        ),
        "resume_rejection_count": len(
            audit.get("resume_rejections", [])
        ),
        "resume_rejection_errors": [
            error
            for rejection in audit.get("resume_rejections", [])
            for error in rejection.get("errors", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pt-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "uploaded_data.csv",
    )
    args = parser.parse_args()
    control_device = torch.device("cpu")
    control_environment = protocol.runtime_environment_contract(
        control_device
    )
    control_environment_sha256 = protocol.environment_fingerprint(
        control_environment
    )
    mismatched_environment = dict(control_environment)
    mismatched_environment["torch_version"] = (
        f"{control_environment['torch_version']}-mismatch-control"
    )
    mismatched_environment_sha256 = protocol.environment_fingerprint(
        mismatched_environment
    )

    rows = fixture_rows()
    train_a, holdout_a, split_a = protocol.deterministic_release_stratified_split(
        rows,
        holdout_fraction=0.20,
        seed=11,
    )
    train_b, holdout_b, split_b = protocol.deterministic_release_stratified_split(
        rows,
        holdout_fraction=0.20,
        seed=11,
    )
    relocated_rows = [
        {
            **row,
            "path": (
                f"/relocated/droid-mirror/{row['dataset']}/"
                f"{Path(str(row['path'])).name}"
            ),
        }
        for row in rows
    ]
    relocated_train, relocated_holdout, relocated_split = (
        protocol.deterministic_release_stratified_split(
            relocated_rows,
            holdout_fraction=0.20,
            seed=11,
        )
    )
    pt_values, pt_audit = base.load_pt_probabilities(args.pt_source)
    (
        features,
        source_target,
        qtail_target,
        tail_scores,
        _,
        normalization,
    ) = base.make_training_matrix(
        rows,
        pt_values=pt_values,
        normalization_fit_indices=train_a,
        allocation_fit_indices=train_a,
    )
    holdout_set = {int(index) for index in holdout_a}
    mutated_rows = []
    for index, row in enumerate(rows):
        mutated = dict(row)
        mutated["instruction_hashes"] = list(row["instruction_hashes"])
        if index in holdout_set:
            mutated["instruction_hashes"].append(
                f"holdout-only-{index}-must-not-affect-training-fit"
            )
        mutated_rows.append(mutated)
    (
        mutated_features,
        _,
        mutated_qtail_target,
        mutated_tail_scores,
        _,
        mutated_normalization,
    ) = base.make_training_matrix(
        mutated_rows,
        pt_values=pt_values,
        normalization_fit_indices=train_a,
        allocation_fit_indices=train_a,
    )

    bootstrap_source = np.asarray(
        [0.16, 0.14, 0.12, 0.08, 0.16, 0.14, 0.12, 0.08],
        dtype=np.float64,
    )
    bootstrap_qtail = np.asarray(
        [0.10, 0.10, 0.16, 0.14, 0.10, 0.10, 0.16, 0.14],
        dtype=np.float64,
    )
    bootstrap_tail = np.asarray(
        [False, False, True, True, False, False, True, True]
    )
    bootstrap_strata = np.asarray(
        ["1.0.0"] * 4 + ["1.0.1"] * 4
    )
    bootstrap_a = protocol.paired_bootstrap(
        bootstrap_source,
        bootstrap_qtail,
        bootstrap_tail,
        bootstrap_strata,
        samples=250,
        seed=20260727,
    )
    bootstrap_b = protocol.paired_bootstrap(
        bootstrap_source,
        bootstrap_qtail,
        bootstrap_tail,
        bootstrap_strata,
        samples=250,
        seed=20260727,
    )
    randomization_a = protocol.paired_arm_swap_randomization(
        bootstrap_source,
        bootstrap_qtail,
        bootstrap_tail,
        samples=5_000,
        seed=protocol.FORMAL_RANDOMIZATION_SEED,
    )
    randomization_b = protocol.paired_arm_swap_randomization(
        bootstrap_source,
        bootstrap_qtail,
        bootstrap_tail,
        samples=5_000,
        seed=protocol.FORMAL_RANDOMIZATION_SEED,
    )
    randomization_negative = protocol.paired_arm_swap_randomization(
        bootstrap_source,
        bootstrap_source,
        bootstrap_tail,
        samples=5_000,
        seed=protocol.FORMAL_RANDOMIZATION_SEED,
    )
    supported_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=randomization_a["observed_gain_pp"],
        extreme_underallocation_reduction_pp=8.0,
        bootstrap=bootstrap_a,
        randomization=randomization_a,
    )
    not_supported_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=-2.0,
        extreme_underallocation_reduction_pp=-1.0,
        bootstrap={
            "ci95_low_pp": -3.0,
            "ci95_high_pp": -1.0,
        },
        randomization=randomization_negative,
    )
    inconclusive_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=1.0,
        extreme_underallocation_reduction_pp=0.5,
        bootstrap={
            "ci95_low_pp": -1.0,
            "ci95_high_pp": 3.0,
        },
        randomization=randomization_negative,
    )
    supported_ci_boundary_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=2.0,
        extreme_underallocation_reduction_pp=0.01,
        bootstrap={
            "ci95_low_pp": 2.0,
            "ci95_high_pp": 4.0,
        },
        randomization=randomization_negative,
    )
    below_threshold_ci_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=1.9,
        extreme_underallocation_reduction_pp=1.0,
        bootstrap={
            "ci95_low_pp": 1.5,
            "ci95_high_pp": 1.999,
        },
        randomization=randomization_negative,
    )
    zero_extreme_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=4.0,
        extreme_underallocation_reduction_pp=0.0,
        bootstrap={
            "ci95_low_pp": 3.0,
            "ci95_high_pp": 5.0,
        },
        randomization=randomization_negative,
    )
    threshold_straddling_control = protocol.heldout_hypothesis_gate(
        tail_share_gain_pp=3.0,
        extreme_underallocation_reduction_pp=1.0,
        bootstrap={
            "ci95_low_pp": 1.5,
            "ci95_high_pp": 3.5,
        },
        randomization=randomization_negative,
    )
    outcome_controls = {
        "supported": supported_control,
        "not_supported": not_supported_control,
        "inconclusive": inconclusive_control,
    }
    repo_root = Path(__file__).resolve().parents[1]
    progress_source = (
        repo_root / "tools" / "qtail_droid_full_progress.py"
    ).read_text(encoding="utf-8")
    stage_source = (
        repo_root / "tools" / "qtail_verify_droid_stage_markers.py"
    ).read_text(encoding="utf-8")
    page_verifier_source = (
        repo_root / "tools" / "qtail_verify_droid_page.mjs"
    ).read_text(encoding="utf-8")
    page_source = (
        repo_root / "qtail-droid-full-training.html"
    ).read_text(encoding="utf-8")
    formal_protocol_accepts_locked = not protocol.formal_protocol_mismatches(
        require_verified_mirror=True,
        seed=protocol.FORMAL_SEED,
        steps=protocol.FORMAL_STEPS_PER_STAGE,
        checkpoint_every_steps=(
            protocol.FORMAL_CHECKPOINT_EVERY_STEPS
        ),
        bootstrap_samples=protocol.FORMAL_BOOTSTRAP_SAMPLES,
        holdout_fraction=protocol.FORMAL_HOLDOUT_FRACTION,
        min_record_parse_rate=protocol.FORMAL_MIN_RECORD_PARSE_RATE,
        min_record_scan_complete_rate=(
            protocol.FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE
        ),
        pt_source_sha256=protocol.FORMAL_PT_SOURCE_SHA256,
    )
    formal_protocol_rejections = protocol.formal_protocol_mismatches(
        require_verified_mirror=False,
        seed=protocol.FORMAL_SEED,
        steps=protocol.FORMAL_STEPS_PER_STAGE,
        checkpoint_every_steps=10_000,
        bootstrap_samples=protocol.FORMAL_BOOTSTRAP_SAMPLES,
        holdout_fraction=protocol.FORMAL_HOLDOUT_FRACTION,
        min_record_parse_rate=0.95,
        min_record_scan_complete_rate=0.95,
        pt_source_sha256=protocol.FORMAL_PT_SOURCE_SHA256,
    )
    formal_protocol_rejection_cases = {
        "require_verified_mirror": (
            {"require_verified_mirror": False},
            "require_verified_mirror=false",
        ),
        "seed": (
            {"seed": protocol.FORMAL_SEED + 1},
            f"seed={protocol.FORMAL_SEED + 1}",
        ),
        "steps": (
            {"steps": protocol.FORMAL_STEPS_PER_STAGE - 1},
            f"steps={protocol.FORMAL_STEPS_PER_STAGE - 1}",
        ),
        "checkpoint_every_steps": (
            {
                "checkpoint_every_steps":
                protocol.FORMAL_CHECKPOINT_EVERY_STEPS + 1
            },
            "checkpoint_every_steps="
            f"{protocol.FORMAL_CHECKPOINT_EVERY_STEPS + 1}",
        ),
        "bootstrap_samples": (
            {
                "bootstrap_samples":
                protocol.FORMAL_BOOTSTRAP_SAMPLES - 1
            },
            f"bootstrap_samples={protocol.FORMAL_BOOTSTRAP_SAMPLES - 1}",
        ),
        "holdout_fraction": (
            {"holdout_fraction": 0.25},
            "holdout_fraction=0.25",
        ),
        "min_record_parse_rate": (
            {"min_record_parse_rate": 0.99},
            "min_record_parse_rate=0.99",
        ),
        "min_record_scan_complete_rate": (
            {"min_record_scan_complete_rate": 0.99},
            "min_record_scan_complete_rate=0.99",
        ),
        "pt_source_sha256": (
            {"pt_source_sha256": "0" * 64},
            "pt_source_sha256",
        ),
    }
    locked_protocol_arguments = {
        "require_verified_mirror": True,
        "seed": protocol.FORMAL_SEED,
        "steps": protocol.FORMAL_STEPS_PER_STAGE,
        "checkpoint_every_steps": (
            protocol.FORMAL_CHECKPOINT_EVERY_STEPS
        ),
        "bootstrap_samples": protocol.FORMAL_BOOTSTRAP_SAMPLES,
        "holdout_fraction": protocol.FORMAL_HOLDOUT_FRACTION,
        "min_record_parse_rate": protocol.FORMAL_MIN_RECORD_PARSE_RATE,
        "min_record_scan_complete_rate": (
            protocol.FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE
        ),
        "pt_source_sha256": protocol.FORMAL_PT_SOURCE_SHA256,
    }
    formal_protocol_rejections_by_field = {
        name: {
            "expected": expected,
            "observed": protocol.formal_protocol_mismatches(
                **{**locked_protocol_arguments, **mutation}
            ),
        }
        for name, (mutation, expected)
        in formal_protocol_rejection_cases.items()
    }
    complete_releases = [
        {
            "release": release,
            "metadata_status": "verified",
            "observed_tfrecord_shards": expected[
                "tfrecord_shards"
            ],
            "official_tfrecord_shards": expected[
                "tfrecord_shards"
            ],
            "observed_records_decoded": expected["records"],
            "official_expected_records": expected["records"],
            "full_shard_coverage": True,
            "full_record_count_match": True,
        }
        for release, expected in protocol.FORMAL_RELEASE_CONTRACT.items()
    ]
    incomplete_release = {
        **complete_releases[1],
        "observed_records_decoded": (
            complete_releases[1]["observed_records_decoded"] - 1
        ),
        "full_record_count_match": False,
    }
    formal_record_closure_accepts_complete = not (
        protocol.formal_record_closure_errors(complete_releases)
    )
    formal_record_closure_rejections = (
        protocol.formal_record_closure_errors(
            [complete_releases[0], incomplete_release]
        )
    )
    formal_single_release_rejections = (
        protocol.formal_record_closure_errors(
            [complete_releases[0]]
        )
    )
    formal_shard_total_rejections = {}
    for total, first_release_shards in (
        (4_095, 2_047),
        (4_097, 2_049),
        (4_100, 2_052),
    ):
        changed = [dict(row) for row in complete_releases]
        changed[0]["observed_tfrecord_shards"] = first_release_shards
        changed[0]["full_shard_coverage"] = False
        formal_shard_total_rejections[str(total)] = (
            protocol.formal_record_closure_errors(changed)
        )
    coverage_rows = [
        {"instruction_hashes": ["shared", "rare-a"]},
        {"instruction_hashes": ["shared", "rare-b"]},
        {"instruction_hashes": ["rare-a"]},
        {"instruction_hashes": ["rare-b"]},
        {"instruction_hashes": ["unseen-c"]},
    ]
    coverage_source = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
    coverage_qtail = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    coverage_control = protocol.rare_instruction_fingerprint_coverage(
        coverage_rows,
        np.asarray([0, 1]),
        np.asarray([2, 3, 4]),
        coverage_source,
        coverage_qtail,
        budgets=(2, 4),
        thresholds=(0.25, 0.50),
    )
    mutated_coverage_rows = [dict(row) for row in coverage_rows]
    mutated_coverage_rows[4] = {
        "instruction_hashes": ["unseen-c", "holdout-only-d"]
    }
    mutated_coverage_control = (
        protocol.rare_instruction_fingerprint_coverage(
            mutated_coverage_rows,
            np.asarray([0, 1]),
            np.asarray([2, 3, 4]),
            coverage_source,
            coverage_qtail,
            budgets=(2, 4),
            thresholds=(0.25, 0.50),
        )
    )
    no_eligible_coverage = (
        protocol.rare_instruction_fingerprint_coverage(
            [
                {"instruction_hashes": ["common"]},
                {"instruction_hashes": ["common"]},
                {"instruction_hashes": ["common"]},
                {"instruction_hashes": ["common"]},
            ],
            np.asarray([0, 1]),
            np.asarray([2, 3]),
            np.asarray([0.5, 0.5]),
            np.asarray([0.5, 0.5]),
            max_train_document_frequency=0,
            budgets=(2,),
            thresholds=(0.50,),
        )
    )
    empty_instruction_coverage = (
        protocol.rare_instruction_fingerprint_coverage(
            [
                {"instruction_hashes": []},
                {"instruction_hashes": []},
                {"instruction_hashes": []},
                {"instruction_hashes": []},
            ],
            np.asarray([0, 1]),
            np.asarray([2, 3]),
            np.asarray([0.5, 0.5]),
            np.asarray([0.5, 0.5]),
            budgets=(2,),
            thresholds=(0.50,),
        )
    )
    with tempfile.TemporaryDirectory(prefix="qtail-droid-resume-control-") as temp:
        control_root = Path(temp)
        control_features = np.linspace(
            0.01,
            0.80,
            num=80,
            dtype=np.float32,
        ).reshape(8, 10)
        control_target = np.arange(1, 9, dtype=np.float32)
        control_target /= control_target.sum()
        (
            full_history,
            full_predictions,
            full_model,
            full_resume,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "full",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        resumed_checkpoint_dir = (
            control_root / "resumed" / "intermediate_checkpoints"
        )
        resumed_checkpoint_dir.mkdir(parents=True)
        source_checkpoint = (
            control_root
            / "full"
            / "intermediate_checkpoints"
            / "resume_control_step_000003.pt"
        )
        source_initial_checkpoint = (
            control_root
            / "full"
            / "intermediate_checkpoints"
            / "resume_control_step_000000.pt"
        )

        def copy_resume_chain(destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            for source in (source_initial_checkpoint, source_checkpoint):
                shutil.copy2(source, destination / source.name)

        copy_resume_chain(resumed_checkpoint_dir)
        (
            resumed_history,
            resumed_predictions,
            resumed_model,
            resumed_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "resumed",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        final_checkpoint = torch.load(
            control_root
            / "resumed"
            / "intermediate_checkpoints"
            / "resume_control_step_000007.pt",
            map_location="cpu",
            weights_only=False,
        )
        rejected_checkpoint_dir = (
            control_root / "rejected" / "intermediate_checkpoints"
        )
        copy_resume_chain(rejected_checkpoint_dir)
        rejected_payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        rejected_payload["device"] = "mps"
        rejected_payload["optimizer"] = "SGD(lr=0.1)"
        torch.save(
            rejected_payload,
            rejected_checkpoint_dir / source_checkpoint.name,
        )
        (
            rejected_history,
            rejected_predictions,
            rejected_model,
            rejected_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "rejected",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        environment_rejected_checkpoint_dir = (
            control_root
            / "environment_rejected"
            / "intermediate_checkpoints"
        )
        copy_resume_chain(environment_rejected_checkpoint_dir)
        environment_rejected_payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        environment_rejected_payload["environment_contract"] = (
            mismatched_environment
        )
        environment_rejected_payload["environment_fingerprint"] = (
            mismatched_environment_sha256
        )
        torch.save(
            environment_rejected_payload,
            environment_rejected_checkpoint_dir / source_checkpoint.name,
        )
        (
            environment_rejected_history,
            environment_rejected_predictions,
            environment_rejected_model,
            environment_rejected_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "environment_rejected",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        truncated_checkpoint_dir = (
            control_root / "truncated" / "intermediate_checkpoints"
        )
        copy_resume_chain(truncated_checkpoint_dir)
        (
            truncated_checkpoint_dir / source_checkpoint.name
        ).write_bytes(b"truncated-checkpoint")
        (
            truncated_history,
            truncated_predictions,
            truncated_model,
            truncated_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "truncated",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        overstep_checkpoint_dir = (
            control_root / "overstep" / "intermediate_checkpoints"
        )
        copy_resume_chain(overstep_checkpoint_dir)
        overstep_payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        overstep_payload["step"] = 99
        overstep_payload["optimizer_updates_completed"] = 99
        torch.save(
            overstep_payload,
            overstep_checkpoint_dir / source_checkpoint.name,
        )
        (
            overstep_history,
            overstep_predictions,
            overstep_model,
            overstep_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "overstep",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        model_tamper_checkpoint_dir = (
            control_root / "model_tamper" / "intermediate_checkpoints"
        )
        copy_resume_chain(model_tamper_checkpoint_dir)
        model_tamper_payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        first_model_tensor = next(
            iter(model_tamper_payload["state_dict"].values())
        )
        first_model_tensor.view(-1)[0] += 100.0
        torch.save(
            model_tamper_payload,
            model_tamper_checkpoint_dir / source_checkpoint.name,
        )
        (
            model_tamper_history,
            model_tamper_predictions,
            model_tamper_model,
            model_tamper_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "model_tamper",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        optimizer_tamper_checkpoint_dir = (
            control_root
            / "optimizer_tamper"
            / "intermediate_checkpoints"
        )
        copy_resume_chain(optimizer_tamper_checkpoint_dir)
        optimizer_tamper_payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        optimizer_moment_mutated = False
        for parameter_state in optimizer_tamper_payload[
            "optimizer_state_dict"
        ]["state"].values():
            for state_name in ("exp_avg", "exp_avg_sq"):
                moment = parameter_state.get(state_name)
                if isinstance(moment, torch.Tensor) and moment.numel():
                    moment.view(-1)[0] += 100.0
                    optimizer_moment_mutated = True
                    break
            if optimizer_moment_mutated:
                break
        if not optimizer_moment_mutated:
            raise AssertionError("AdamW moment fixture was not populated")
        torch.save(
            optimizer_tamper_payload,
            optimizer_tamper_checkpoint_dir / source_checkpoint.name,
        )
        (
            optimizer_tamper_history,
            optimizer_tamper_predictions,
            optimizer_tamper_model,
            optimizer_tamper_audit,
        ) = protocol.train_once_audited(
            features=control_features,
            target=control_target,
            steps=7,
            seed=19,
            label="resume_control",
            device=torch.device("cpu"),
            out=control_root / "optimizer_tamper",
            checkpoint_every_steps=3,
            completed_models=[],
            environment_contract=control_environment,
        )
        state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                resumed_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        rejected_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                rejected_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        environment_rejected_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                environment_rejected_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        truncated_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                truncated_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        overstep_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                overstep_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        model_tamper_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                model_tamper_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        optimizer_tamper_state_matches = all(
            torch.allclose(
                full_model.state_dict()[name],
                optimizer_tamper_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in full_model.state_dict()
        )
        manifest_control_root = control_root / "checkpoint_manifest"
        manifest_labels = (
            "evaluation_source",
            "evaluation_qtail",
            "deployment_source",
            "deployment_qtail",
        )
        for label in manifest_labels:
            protocol.train_once_audited(
                features=control_features,
                target=control_target,
                steps=7,
                seed=19,
                label=label,
                device=torch.device("cpu"),
                out=manifest_control_root,
                checkpoint_every_steps=3,
                completed_models=[],
                environment_contract=control_environment,
            )
        _, accepted_checkpoint_manifest = (
            protocol.build_intermediate_checkpoint_manifest(
                out=manifest_control_root,
                steps=7,
                checkpoint_every_steps=3,
                seed=19,
                device=torch.device("cpu"),
                environment_sha256=control_environment_sha256,
            )
        )
        fingerprint_checkpoint = (
            manifest_control_root
            / "intermediate_checkpoints"
            / "evaluation_qtail_step_000000.pt"
        )
        state_checkpoint = (
            manifest_control_root
            / "intermediate_checkpoints"
            / "evaluation_qtail_step_000003.pt"
        )

        def manifest_tamper_result(
            checkpoint: Path,
            mutator: Any,
        ) -> tuple[bool, list[str]]:
            original_bytes = checkpoint.read_bytes()
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            mutator(payload)
            torch.save(payload, checkpoint)
            rejected = False
            normalized_errors: list[str] = []
            try:
                protocol.build_intermediate_checkpoint_manifest(
                    out=manifest_control_root,
                    steps=7,
                    checkpoint_every_steps=3,
                    seed=19,
                    device=torch.device("cpu"),
                    environment_sha256=control_environment_sha256,
                )
            except ValueError:
                rejected_manifest = json.loads(
                    (
                        manifest_control_root
                        / "droid_intermediate_checkpoint_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                normalized_errors = [
                    str(error).replace(
                        str(control_root),
                        "<control_root>",
                    )
                    for error in rejected_manifest.get("errors", [])
                ]
                rejected = rejected_manifest.get("status") == "failed"
            finally:
                checkpoint.write_bytes(original_bytes)
            return rejected, normalized_errors

        def mutate_fingerprints(payload: dict[str, Any]) -> None:
            payload["feature_sha256"] = "f" * 64
            payload["initialized_state_sha256"] = "e" * 64

        def mutate_model_tensor(payload: dict[str, Any]) -> None:
            tensor = next(iter(payload["state_dict"].values()))
            tensor.view(-1)[0] += 100.0

        def mutate_optimizer_moment(payload: dict[str, Any]) -> None:
            for parameter_state in payload["optimizer_state_dict"][
                "state"
            ].values():
                for state_name in ("exp_avg", "exp_avg_sq"):
                    moment = parameter_state.get(state_name)
                    if isinstance(moment, torch.Tensor) and moment.numel():
                        moment.view(-1)[0] += 100.0
                        return
            raise AssertionError("AdamW moment fixture was not populated")

        (
            fingerprint_mismatch_rejected,
            fingerprint_mismatch_errors,
        ) = manifest_tamper_result(
            fingerprint_checkpoint,
            mutate_fingerprints,
        )
        (
            model_state_tamper_manifest_rejected,
            model_state_tamper_manifest_errors,
        ) = manifest_tamper_result(
            state_checkpoint,
            mutate_model_tensor,
        )
        (
            optimizer_tamper_manifest_rejected,
            optimizer_tamper_manifest_errors,
        ) = manifest_tamper_result(
            state_checkpoint,
            mutate_optimizer_moment,
        )
        unexpected_checkpoint = (
            manifest_control_root
            / "intermediate_checkpoints"
            / "unexpected_step_000001.pt"
        )
        unexpected_checkpoint.write_bytes(b"unexpected-checkpoint")
        unexpected_checkpoint_rejected = False
        try:
            protocol.build_intermediate_checkpoint_manifest(
                out=manifest_control_root,
                steps=7,
                checkpoint_every_steps=3,
                seed=19,
                device=torch.device("cpu"),
                environment_sha256=control_environment_sha256,
            )
        except ValueError:
            rejected_checkpoint_manifest = json.loads(
                (
                    manifest_control_root
                    / "droid_intermediate_checkpoint_manifest.json"
                ).read_text(encoding="utf-8")
            )
            unexpected_checkpoint_rejected = bool(
                rejected_checkpoint_manifest.get("status") == "failed"
                and any(
                    "unexpected checkpoint" in str(error)
                    for error in rejected_checkpoint_manifest.get(
                        "errors", []
                    )
                )
            )
        else:
            rejected_checkpoint_manifest = {}
        checkpoint_manifest_control = {
            "accepted_status": accepted_checkpoint_manifest.get("status"),
            "accepted_expected_steps": accepted_checkpoint_manifest.get(
                "contract", {}
            ).get("expected_steps"),
            "accepted_expected_checkpoint_count": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "expected_checkpoint_count"
                )
            ),
            "accepted_actual_checkpoint_count": (
                accepted_checkpoint_manifest.get(
                    "actual_checkpoint_count"
                )
            ),
            "accepted_pairs": sorted(
                [
                    [
                        str(entry.get("model_stage")),
                        int(entry.get("step", -1)),
                    ]
                    for entry in accepted_checkpoint_manifest.get(
                        "entries", []
                    )
                ]
            ),
            "accepted_hashes_are_sha256": all(
                len(str(entry.get("sha256", ""))) == 64
                and len(str(entry.get("feature_sha256", ""))) == 64
                and len(
                    str(entry.get("initialized_state_sha256", ""))
                )
                == 64
                and len(str(entry.get("model_state_sha256", ""))) == 64
                and len(
                    str(entry.get("optimizer_state_sha256", ""))
                )
                == 64
                and (
                    entry.get("step") == 0
                    or len(
                        str(entry.get("parent_checkpoint_sha256", ""))
                    )
                    == 64
                )
                for entry in accepted_checkpoint_manifest.get(
                    "entries", []
                )
            ),
            "accepted_checkpoint_format_version": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "checkpoint_format_version"
                )
            ),
            "accepted_checkpoint_chain_version": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "checkpoint_chain_version"
                )
            ),
            "accepted_parent_chains_verified": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "parent_checkpoint_hash_chains_verified"
                )
            ),
            "accepted_paired_feature_signatures_equal": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "paired_feature_signatures_equal"
                )
            ),
            "accepted_initialized_state_signatures_equal": (
                accepted_checkpoint_manifest.get("contract", {}).get(
                    "initialized_state_signatures_equal"
                )
            ),
            "fingerprint_mismatch_rejected": (
                fingerprint_mismatch_rejected
            ),
            "fingerprint_mismatch_errors": fingerprint_mismatch_errors,
            "model_state_tamper_rejected": (
                model_state_tamper_manifest_rejected
            ),
            "model_state_tamper_errors": (
                model_state_tamper_manifest_errors
            ),
            "optimizer_moment_tamper_rejected": (
                optimizer_tamper_manifest_rejected
            ),
            "optimizer_moment_tamper_errors": (
                optimizer_tamper_manifest_errors
            ),
            "unexpected_checkpoint_rejected": (
                unexpected_checkpoint_rejected
            ),
            "rejected_status": rejected_checkpoint_manifest.get("status"),
        }
        resume_control = {
            "uninterrupted_resume_audit": stable_resume_audit(full_resume),
            "resumed_audit": stable_resume_audit(resumed_audit),
            "rejected_mismatch_audit": stable_resume_audit(
                rejected_audit
            ),
            "rejected_environment_audit": stable_resume_audit(
                environment_rejected_audit
            ),
            "full_history": full_history,
            "resumed_history": resumed_history,
            "rejected_mismatch_history": rejected_history,
            "truncated_checkpoint_audit": stable_resume_audit(
                truncated_audit
            ),
            "overstep_checkpoint_audit": stable_resume_audit(
                overstep_audit
            ),
            "model_tensor_tamper_audit": stable_resume_audit(
                model_tamper_audit
            ),
            "optimizer_moment_tamper_audit": stable_resume_audit(
                optimizer_tamper_audit
            ),
            "final_checkpoint_format_version": final_checkpoint.get(
                "format_version"
            ),
            "final_optimizer_updates_completed": final_checkpoint.get(
                "optimizer_updates_completed"
            ),
        }
    with tempfile.TemporaryDirectory(
        prefix="qtail-droid-bounded-cli-control-"
    ) as temp:
        cli_root = Path(temp)
        cli_data = cli_root / "data"
        cli_out = cli_root / "out"
        cli_markers = cli_root / "markers"
        for release in ("1.0.0", "1.0.1"):
            release_dir = cli_data / release
            release_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                shard_path = (
                    release_dir
                    / (
                        f"droid-{release}-{index}.tfrecord-"
                        f"{index:05d}-of-00002"
                    )
                )
                writer = TFRecordWriter(str(shard_path))
                writer.write(
                    {
                        "steps/is_first": ([1, 0], "int"),
                        "steps/reward": ([0.0, 1.0], "float"),
                        "steps/action": (
                            [0.1 + index, 0.2 + index],
                            "float",
                        ),
                        "steps/language_instruction": (
                            f"task-{release}-{index}".encode("utf-8"),
                            "byte",
                        ),
                    }
                )
                writer.close()
        cli_completed = subprocess.run(
            [
                sys.executable,
                str(Path(protocol.__file__).resolve()),
                "--data-dir",
                str(cli_data),
                "--out",
                str(cli_out),
                "--marker-dir",
                str(cli_markers),
                "--steps",
                "1",
                "--checkpoint-every-steps",
                "1",
                "--max-shards",
                "4",
                "--records-per-shard",
                "1",
                "--min-shards",
                "4",
                "--bootstrap-samples",
                "20",
                "--status-every-shards",
                "1",
                "--pt-source",
                str(args.pt_source),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        bounded_report_path = (
            cli_out / "droid_full_training_report.json"
        )
        bounded_report = (
            json.loads(bounded_report_path.read_text(encoding="utf-8"))
            if bounded_report_path.is_file()
            else {}
        )
        bounded_formal_marker_absent = not (
            cli_markers / "DROID_MODEL_TRAINING_COMPLETE"
        ).exists()
        helper_bounded_marker_dir = cli_root / "helper-bounded"
        helper_formal_marker_dir = cli_root / "helper-formal"
        bounded_helper_written = (
            protocol.publish_training_completion_marker(
                helper_bounded_marker_dir,
                formal_run=False,
            )
        )
        formal_helper_written = (
            protocol.publish_training_completion_marker(
                helper_formal_marker_dir,
                formal_run=True,
            )
        )
        bounded_cli_control = {
            "returncode": cli_completed.returncode,
            "stderr_tail": cli_completed.stderr[-2_000:],
            "report_status": bounded_report.get("status"),
            "formal_protocol_locked": bounded_report.get(
                "formal_protocol", {}
            ).get("locked"),
            "formal_completion_marker_absent": (
                bounded_formal_marker_absent
            ),
            "bounded_helper_written": bounded_helper_written,
            "bounded_helper_marker_exists": (
                helper_bounded_marker_dir
                / "DROID_MODEL_TRAINING_COMPLETE"
            ).exists(),
            "formal_helper_written": formal_helper_written,
            "formal_helper_marker_exists": (
                helper_formal_marker_dir
                / "DROID_MODEL_TRAINING_COMPLETE"
            ).exists(),
        }
    checks = {
        "split_indices_deterministic": bool(
            np.array_equal(train_a, train_b)
            and np.array_equal(holdout_a, holdout_b)
        ),
        "split_hash_deterministic": (
            split_a["holdout_relative_path_sha256"]
            == split_b["holdout_relative_path_sha256"]
            and split_a["holdout_relative_paths"]
            == split_b["holdout_relative_paths"]
        ),
        "holdout_membership_uses_locked_official_relative_paths": bool(
            split_a["holdout_membership_locked"] is True
            and split_a["membership_path_scope"]
            == "official_release_relative_path"
            and len(split_a["holdout_relative_paths"]) == len(holdout_a)
            and all(
                not path.startswith("/")
                and path.count("/") == 1
                for path in split_a["holdout_relative_paths"]
            )
            and np.array_equal(train_a, relocated_train)
            and np.array_equal(holdout_a, relocated_holdout)
            and split_a["holdout_relative_paths"]
            == relocated_split["holdout_relative_paths"]
            and split_a["holdout_relative_path_sha256"]
            == relocated_split["holdout_relative_path_sha256"]
        ),
        "both_releases_in_holdout": (
            {rows[int(index)]["dataset"] for index in holdout_a}
            == {"1.0.0", "1.0.1"}
        ),
        "normalization_fit_excludes_holdout": (
            normalization["fit_row_count"] == len(train_a)
        ),
        "tail_transform_fit_excludes_holdout": (
            normalization["tail_score_contract"]["transform_fit_row_count"]
            == len(train_a)
            and normalization["tail_score_contract"][
                "instruction_document_frequency_fit"
            ]
            == "normalization_fit_rows_only"
        ),
        "allocation_fit_excludes_holdout": bool(
            normalization["allocation_fit_row_count"] == len(train_a)
            and bool(np.all(source_target[holdout_a] == 0.0))
            and bool(np.all(qtail_target[holdout_a] == 0.0))
            and bool(np.isclose(source_target.sum(), 1.0))
            and bool(np.isclose(qtail_target.sum(), 1.0))
        ),
        "holdout_instruction_mutation_cannot_change_training_fit": bool(
            np.allclose(features[train_a], mutated_features[train_a])
            and np.allclose(tail_scores[train_a], mutated_tail_scores[train_a])
            and np.allclose(qtail_target[train_a], mutated_qtail_target[train_a])
            and normalization["tail_score_contract"]["effective_weights"]
            == mutated_normalization["tail_score_contract"]["effective_weights"]
        ),
        "feature_dimension_is_ten": (
            features.shape == (len(rows), 10)
            and len(base.FEATURE_NAMES) == 10
        ),
        "shard_position_not_used": (
            normalization["tail_score_contract"]["shard_position_used"] is False
        ),
        "empirical_pt_source_audited": (
            pt_audit["count"] >= len(rows)
            and len(pt_audit["sha256"]) == 64
        ),
        "positive_control_supports_hypothesis": bool(
            supported_control["outcome"] == "supported"
            and supported_control["supported"] is True
            and supported_control["passed"] is True
            and supported_control["completion_role"]
            == "outcome_only_not_experiment_execution_gate"
        ),
        "outcome_controls_cover_exact_three_states": bool(
            set(outcome_controls) == {
                "supported",
                "not_supported",
                "inconclusive",
            }
            and all(
                control["outcome"] == expected
                for expected, control in outcome_controls.items()
            )
            and not_supported_control["supported"] is False
            and not_supported_control["passed"] is False
            and inconclusive_control["supported"] is False
            and inconclusive_control["passed"] is False
            and all(
                control["completion_role"]
                == "outcome_only_not_experiment_execution_gate"
                for control in outcome_controls.values()
            )
        ),
        "minimum_effect_ci_boundaries_are_fail_closed": bool(
            supported_ci_boundary_control["outcome"] == "supported"
            and supported_ci_boundary_control["supported"] is True
            and below_threshold_ci_control["outcome"]
            == "not_supported"
            and below_threshold_ci_control["supported"] is False
            and zero_extreme_control["outcome"] == "not_supported"
            and zero_extreme_control["supported"] is False
            and threshold_straddling_control["outcome"]
            == "inconclusive"
            and threshold_straddling_control["supported"] is False
        ),
        "formal_protocol_requires_verified_mirror_and_full_parse": bool(
            formal_protocol_accepts_locked
            and "require_verified_mirror=false"
            in formal_protocol_rejections
            and "checkpoint_every_steps=10000"
            in formal_protocol_rejections
            and "min_record_parse_rate=0.95"
            in formal_protocol_rejections
            and "min_record_scan_complete_rate=0.95"
            in formal_protocol_rejections
            and all(
                item["expected"] in item["observed"]
                and len(item["observed"]) == 1
                for item
                in formal_protocol_rejections_by_field.values()
            )
        ),
        "formal_record_count_closure_is_mandatory": bool(
            formal_record_closure_accepts_complete
            and "1.0.1:observed_records=95657"
            in formal_record_closure_rejections
            and "total_records=187890"
            in formal_record_closure_rejections
        ),
        "formal_release_set_is_exact": bool(
            "1.0.1:release_missing"
            in formal_single_release_rejections
            and "total_shards=2048"
            in formal_single_release_rejections
            and "total_records=92233"
            in formal_single_release_rejections
        ),
        "formal_shard_totals_4095_4097_4100_rejected": all(
            f"total_shards={total}" in errors
            for total, errors in formal_shard_total_rejections.items()
        ),
        "release_stratified_bootstrap_deterministic": (
            bootstrap_a == bootstrap_b
            and bootstrap_a["method"] == protocol.BOOTSTRAP_METHOD
            and bootstrap_a["strata"] == list(protocol.BOOTSTRAP_STRATA)
            and bootstrap_a["strata_counts"]
            == {"1.0.0": 4, "1.0.1": 4}
        ),
        "bootstrap_fraction_is_explicitly_not_p_value": bool(
            bootstrap_a["p_gain_le_zero_is_p_value"] is False
            and bootstrap_a["p_gain_le_zero"]
            == bootstrap_a["descriptive_fraction_gain_le_zero"]
            and "not a hypothesis-test p value"
            in bootstrap_a["legacy_field_notice"]
        ),
        "paired_arm_swap_diagnostic_is_deterministic": bool(
            randomization_a == randomization_b
            and randomization_a["samples"] == 5_000
            and randomization_a[
                "conditional_p_value_is_valid_p_value"
            ]
            is False
            and randomization_a[
                "exchangeability_justified_by_experiment_design"
            ]
            is False
            and randomization_a["finite_sample_correction"]
            == "(k+1)/(B+1)"
            and "not a valid hypothesis-test p value"
            in randomization_a["legacy_field_notice"]
        ),
        "paired_arm_swap_diagnostic_never_gates_completion": bool(
            randomization_negative["observed_gain_pp"] == 0.0
            and randomization_negative["conditional_p_value"] == 1.0
            and '"outcome_is_completion_gate": False'
            in progress_source
            and 'hypothesis_gate.get("outcome") == recomputed_outcome'
            in progress_source
            and 'hypothesis_gate.get("supported") is recomputed_supported'
            in progress_source
            and 'hypothesis_gate.get("supported") is True'
            not in progress_source
            and 'gate.get("outcome") != expected_outcome'
            in stage_source
            and 'gate.get("supported") is not expected_supported'
            in stage_source
            and 'gate.get("supported") is not True'
            not in stage_source
            and "hypothesisGate.outcome === recomputedOutcome"
            in page_verifier_source
            and "hypothesisGate.supported === recomputedSupported"
            in page_verifier_source
            and "hypothesisGate.supported === true"
            not in page_verifier_source
            and "hypothesisGate.passed === true"
            not in page_verifier_source
            and (
                "supported / inconclusive / not_supported "
                "相互独立"
            )
            in page_source
        ),
        "rare_fingerprint_coverage_exact_expectation": bool(
            np.isclose(
                coverage_control["curve"][0]["source_expected_coverage"],
                ((1 - 0.2**2) + (1 - 0.9**2) + (1 - 0.9**2)) / 3,
            )
            and np.isclose(
                coverage_control["curve"][0]["qtail_expected_coverage"],
                1 - (2 / 3) ** 2,
            )
        ),
        "rare_fingerprint_qtail_positive_small_budget_control": bool(
            coverage_control["curve"][0]["gain_pp"] > 0.0
            and coverage_control["time_to_coverage"][1]["qtail_draw_reduction"]
            > 0
        ),
        "rare_fingerprint_rarity_fit_excludes_holdout": bool(
            coverage_control["rarity_fit_scope"] == "training_shards_only"
            and coverage_control["evaluation_scope"] == "holdout_shards_only"
            and coverage_control["training_document_frequency_sha256"]
            == mutated_coverage_control[
                "training_document_frequency_sha256"
            ]
            and coverage_control["rare_holdout_fingerprint_count"] == 3
            and mutated_coverage_control[
                "rare_holdout_fingerprint_count"
            ]
            == 4
        ),
        "rare_fingerprint_claim_boundary_is_explicit": bool(
            coverage_control["metric_role"]
            == "auxiliary_descriptive_metric_not_a_completion_gate"
            and "not semantic task coverage"
            in coverage_control["claim_boundary"]
        ),
        "empty_rare_fingerprint_sets_are_explicit_auxiliary_status": bool(
            no_eligible_coverage["status"]
            == "no_eligible_fingerprints"
            and empty_instruction_coverage["status"]
            == "no_eligible_fingerprints"
            and no_eligible_coverage["rare_holdout_fingerprint_count"]
            == 0
            and empty_instruction_coverage[
                "rare_holdout_fingerprint_count"
            ]
            == 0
            and no_eligible_coverage["curve"] == []
            and empty_instruction_coverage["time_to_coverage"] == []
            and no_eligible_coverage["metric_role"]
            == "auxiliary_descriptive_metric_not_a_completion_gate"
        ),
        "checkpoint_resume_matches_uninterrupted": bool(
            resumed_audit["resumed"] is True
            and resumed_audit["resumed_from_step"] == 3
            and resumed_audit["optimizer_updates_completed"] == 7
            and full_history == resumed_history
            and np.allclose(
                full_predictions,
                resumed_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and state_matches
        ),
        "optimizer_update_boundary_is_exact": bool(
            final_checkpoint.get("format_version")
            == protocol.CHECKPOINT_FORMAT_VERSION
            and final_checkpoint.get("checkpoint_chain_version")
            == protocol.CHECKPOINT_CHAIN_VERSION
            and final_checkpoint.get("step") == 7
            and final_checkpoint.get("optimizer_updates_completed") == 7
            and full_resume["optimizer_updates_completed"] == 7
            and full_history[-1]["step"] == 7
            and final_checkpoint.get("environment_fingerprint")
            == control_environment_sha256
        ),
        "mismatched_device_optimizer_checkpoint_rejected": bool(
            rejected_audit["resumed"] is False
            and rejected_audit["resumed_from_step"] == 0
            and rejected_audit["optimizer_updates_completed"] == 7
            and full_history == rejected_history
            and np.allclose(
                full_predictions,
                rejected_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and rejected_state_matches
        ),
        "mismatched_environment_checkpoint_rejected": bool(
            environment_rejected_audit["resumed"] is False
            and environment_rejected_audit["resumed_from_step"] == 0
            and environment_rejected_audit[
                "environment_fingerprint"
            ]
            == control_environment_sha256
            and full_history == environment_rejected_history
            and np.allclose(
                full_predictions,
                environment_rejected_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and environment_rejected_state_matches
        ),
        "truncated_checkpoint_rejected": bool(
            truncated_audit["resumed"] is False
            and truncated_audit["resumed_from_step"] == 0
            and full_history == truncated_history
            and np.allclose(
                full_predictions,
                truncated_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and truncated_state_matches
        ),
        "overstep_checkpoint_rejected": bool(
            overstep_audit["resumed"] is False
            and overstep_audit["resumed_from_step"] == 0
            and full_history == overstep_history
            and np.allclose(
                full_predictions,
                overstep_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and overstep_state_matches
        ),
        "model_tensor_tamper_checkpoint_rejected": bool(
            model_tamper_audit["resumed"] is False
            and model_tamper_audit["resumed_from_step"] == 0
            and any(
                "model_state_sha256" in error
                for rejection in model_tamper_audit[
                    "resume_rejections"
                ]
                for error in rejection["errors"]
            )
            and full_history == model_tamper_history
            and np.allclose(
                full_predictions,
                model_tamper_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and model_tamper_state_matches
        ),
        "optimizer_moment_tamper_checkpoint_rejected": bool(
            optimizer_tamper_audit["resumed"] is False
            and optimizer_tamper_audit["resumed_from_step"] == 0
            and any(
                "optimizer_state_sha256" in error
                for rejection in optimizer_tamper_audit[
                    "resume_rejections"
                ]
                for error in rejection["errors"]
            )
            and full_history == optimizer_tamper_history
            and np.allclose(
                full_predictions,
                optimizer_tamper_predictions,
                rtol=0.0,
                atol=0.0,
            )
            and optimizer_tamper_state_matches
        ),
        "intermediate_checkpoint_manifest_exact_grid": bool(
            checkpoint_manifest_control["accepted_status"] == "complete"
            and checkpoint_manifest_control["accepted_expected_steps"]
            == [0, 3, 6, 7]
            and checkpoint_manifest_control[
                "accepted_expected_checkpoint_count"
            ]
            == 16
            and checkpoint_manifest_control[
                "accepted_actual_checkpoint_count"
            ]
            == 16
            and len(checkpoint_manifest_control["accepted_pairs"]) == 16
            and len(
                {
                    tuple(pair)
                    for pair in checkpoint_manifest_control[
                        "accepted_pairs"
                    ]
                }
            )
            == 16
            and checkpoint_manifest_control[
                "accepted_hashes_are_sha256"
            ]
            is True
            and checkpoint_manifest_control[
                "accepted_paired_feature_signatures_equal"
            ]
            is True
            and checkpoint_manifest_control[
                "accepted_initialized_state_signatures_equal"
            ]
            is True
            and checkpoint_manifest_control[
                "accepted_checkpoint_format_version"
            ]
            == protocol.CHECKPOINT_FORMAT_VERSION
            and checkpoint_manifest_control[
                "accepted_checkpoint_chain_version"
            ]
            == protocol.CHECKPOINT_CHAIN_VERSION
            and checkpoint_manifest_control[
                "accepted_parent_chains_verified"
            ]
            is True
            and checkpoint_manifest_control[
                "fingerprint_mismatch_rejected"
            ]
            is True
            and checkpoint_manifest_control[
                "model_state_tamper_rejected"
            ]
            is True
            and checkpoint_manifest_control[
                "optimizer_moment_tamper_rejected"
            ]
            is True
        ),
        "unexpected_intermediate_checkpoint_rejected": bool(
            checkpoint_manifest_control[
                "unexpected_checkpoint_rejected"
            ]
            is True
            and checkpoint_manifest_control["rejected_status"] == "failed"
        ),
        "bounded_cli_cannot_publish_formal_completion_marker": bool(
            bounded_cli_control["returncode"] == 0
            and bounded_cli_control["report_status"] == "complete"
            and bounded_cli_control["formal_protocol_locked"] is False
            and bounded_cli_control[
                "formal_completion_marker_absent"
            ]
            is True
            and bounded_cli_control["bounded_helper_written"] is False
            and bounded_cli_control[
                "bounded_helper_marker_exists"
            ]
            is False
            and bounded_cli_control["formal_helper_written"] is True
            and bounded_cli_control["formal_helper_marker_exists"] is True
        ),
    }
    passed = all(checks.values())
    payload = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "checks": checks,
        "split_contract": split_a,
        "pt_source_audit": pt_audit,
        "tail_score_contract": normalization["tail_score_contract"],
        "positive_control": supported_control,
        "negative_control": not_supported_control,
        "outcome_controls": outcome_controls,
        "minimum_effect_boundary_controls": {
            "supported_at_ci_lower_bound": (
                supported_ci_boundary_control
            ),
            "not_supported_below_ci_upper_bound": (
                below_threshold_ci_control
            ),
            "not_supported_at_zero_extreme": zero_extreme_control,
            "inconclusive_when_ci_straddles_threshold": (
                threshold_straddling_control
            ),
        },
        "bootstrap_control": bootstrap_a,
        "randomization_positive_control": randomization_a,
        "randomization_negative_control": randomization_negative,
        "formal_protocol_rejections": formal_protocol_rejections,
        "formal_protocol_rejections_by_field": (
            formal_protocol_rejections_by_field
        ),
        "formal_record_closure_rejections": (
            formal_record_closure_rejections
        ),
        "formal_single_release_rejections": (
            formal_single_release_rejections
        ),
        "formal_shard_total_rejections": (
            formal_shard_total_rejections
        ),
        "runtime_environment": control_environment,
        "runtime_environment_fingerprint": control_environment_sha256,
        "rare_instruction_fingerprint_coverage_control": coverage_control,
        "no_eligible_rare_fingerprint_control": no_eligible_coverage,
        "empty_instruction_fingerprint_control": (
            empty_instruction_coverage
        ),
        "optimizer_step_control": resume_control,
        "intermediate_checkpoint_manifest_control": (
            checkpoint_manifest_control
        ),
        "bounded_cli_marker_control": bounded_cli_control,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(json.dumps(payload, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
