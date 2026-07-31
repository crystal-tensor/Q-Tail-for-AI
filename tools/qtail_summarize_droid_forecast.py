#!/usr/bin/env python3
"""Seal a bounded DROID forecast without promoting it to formal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = read_json(args.report)
    if (
        report.get("status") != "complete"
        or report.get("training_scope") != "bounded_test_subset"
        or report.get("formal_protocol", {}).get("locked") is not False
    ):
        raise SystemExit(
            "Forecast summary requires a complete, bounded, non-formal run."
        )
    effect = report["effect_metrics"]
    bootstrap = effect["paired_bootstrap"]
    compute = report["compute_audit"]
    trajectory = report["trajectory_evidence"]
    coverage = report["rare_instruction_fingerprint_coverage"]
    checkpoint_audit = report["intermediate_checkpoint_audit"]
    checkpoint_manifest = Path(checkpoint_audit["manifest"])
    source_tail_share = float(effect["source_pred_tail_share"])
    qtail_tail_share = float(effect["qtail_pred_tail_share"])
    relative_tail_gain = (
        (qtail_tail_share / source_tail_share - 1.0) * 100.0
        if source_tail_share > 0.0
        else None
    )
    coverage_curve = [
        {
            "draw_budget": int(item["draw_budget"]),
            "source_expected_coverage": float(
                item["source_expected_coverage"]
            ),
            "qtail_expected_coverage": float(
                item["qtail_expected_coverage"]
            ),
            "gain_pp": float(item["gain_pp"]),
        }
        for item in coverage["curve"]
    ]
    payload = {
        "generated_at": report["generated_at"],
        "status": "complete_nonformal_forecast",
        "evidence_role": (
            "predictive_engineering_evidence_not_a_formal_completion_gate"
        ),
        "claim_boundary": [
            "This forecast uses only the currently complete 908 DROID 1.0.0 shards.",
            "It excludes DROID 1.0.1 and is not a verified 4,096-shard official mirror.",
            "It uses the formal optimizer-update budget to detect protocol or model failures early.",
            "Its effect metrics cannot be promoted to the final full-data claim.",
            "Negative rare-instruction fingerprint results are preserved alongside positive tail-allocation results.",
            "Allocation-head evidence is not robot-policy tail success.",
        ],
        "input": {
            "shards": int(report["shard_count"]),
            "records_decoded": int(trajectory["records_decoded"]),
            "represented_bytes": int(report["total_bytes"]),
            "releases": report["datasets"],
            "holdout_shards": int(
                report["holdout_evaluation"]["holdout_shards"]
            ),
            "full_record_mode": trajectory["full_record_mode"],
            "record_parse_rate": float(
                trajectory["record_parse_rate"]
            ),
            "record_scan_complete_rate": float(
                trajectory["record_scan_complete_rate"]
            ),
        },
        "equal_compute": {
            "device": compute["training_device"],
            "architecture": compute["architecture"],
            "optimizer": compute["same_optimizer"],
            "seed": int(report["seed"]),
            "steps_per_stage": int(report["steps"]),
            "source_optimizer_updates": int(
                compute["source_optimizer_updates"]
            ),
            "qtail_optimizer_updates": int(
                compute["qtail_optimizer_updates"]
            ),
            "same_parameter_count": compute["same_parameter_count"],
        },
        "tail_allocation_forecast": {
            "source_tail_share": source_tail_share,
            "qtail_tail_share": qtail_tail_share,
            "gain_pp": float(effect["predicted_tail_share_gain_pp"]),
            "relative_gain_percent": relative_tail_gain,
            "extreme_underallocation_reduction_pp": float(
                effect["extreme_underallocation_reduction_pp"]
            ),
            "bootstrap_samples": int(bootstrap["samples"]),
            "ci95_pp": [
                float(bootstrap["ci95_low_pp"]),
                float(bootstrap["ci95_high_pp"]),
            ],
            "p_gain_le_zero": float(
                bootstrap["p_gain_le_zero"]
            ),
            "forecast_gate_passed": bool(
                effect["hypothesis_gate"]["passed"]
            ),
        },
        "rare_instruction_fingerprint_forecast": {
            "claim_boundary": coverage["claim_boundary"],
            "curve": coverage_curve,
            "all_observed_gains_positive": all(
                item["gain_pp"] > 0.0 for item in coverage_curve
            ),
            "observed_direction": (
                "qtail_slower_on_every_reported_budget"
                if all(item["gain_pp"] < 0.0 for item in coverage_curve)
                else "mixed_or_positive"
            ),
        },
        "checkpoint_audit": {
            "status": checkpoint_audit["status"],
            "actual_checkpoint_count": int(
                checkpoint_audit["actual_checkpoint_count"]
            ),
            "expected_steps": checkpoint_audit["contract"][
                "expected_steps"
            ],
            "all_checkpoint_hashes_recorded": checkpoint_audit[
                "all_checkpoint_hashes_recorded"
            ],
        },
        "artifacts": {
            "training_report": artifact(args.report),
            "intermediate_checkpoint_manifest": artifact(
                checkpoint_manifest
            ),
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
