#!/usr/bin/env python3
"""Prove that final-page preview cannot publish formal completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from qtail_droid_full_progress import (
    artifact_link_availability,
    atomic_write_json,
    formal_artifact_requirement_paths,
    public_final_projection_is_committed,
    resolve_pipeline_stage,
    verify_snapshot_source_parity,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preview = {
        "droid_training_complete": True,
        "final_page_qa_complete": False,
        "final_page_qa_preview_active": True,
    }
    preview_stage = resolve_pipeline_stage(preview)
    bootstrap = {
        "droid_training_complete": True,
        "final_page_qa_complete": False,
        "final_page_qa_preview_active": True,
        "final_page_qa_bootstrap_active": True,
    }
    bootstrap_stage = resolve_pipeline_stage(bootstrap)
    sealed_without_projection = {
        "droid_training_complete": True,
        "final_page_qa_sealed": True,
        "final_page_qa_complete": False,
        "final_page_qa_preview_active": False,
        "final_page_qa_bootstrap_active": False,
        "droid_public_projection_committed": False,
    }
    sealed_without_projection_stage = resolve_pipeline_stage(
        sealed_without_projection
    )
    committed = {
        "droid_training_complete": True,
        "final_page_qa_complete": True,
        "final_page_qa_preview_active": False,
    }
    committed_stage = resolve_pipeline_stage(committed)
    bootstrap_audit = {
        "status": "in_progress",
        "passed_requirements": 8,
        "total_requirements": 9,
        "requirements": [
            {
                "id": "final_page_qa",
                "passed": False,
                "evidence": {
                    "committed": False,
                    "preview_active": True,
                    "qa_state": "sealing",
                },
            }
        ],
    }
    committed_audit = {
        **bootstrap_audit,
        "status": "complete",
        "passed_requirements": 9,
        "requirements": [
            {
                "id": "final_page_qa",
                "passed": True,
                "evidence": {
                    "committed": True,
                    "preview_active": False,
                    "qa_state": "committed",
                },
            }
        ],
    }
    complete_latest = {"status": "complete", "stage": "complete"}
    artifact_contract_root = Path("/formal/qtail_droid_full")
    baseline_artifact_contract = formal_artifact_requirement_paths(
        artifact_contract_root,
        {
            "final_page_qa_effective": False,
            "final_page_qa_complete": False,
        },
    )
    effective_artifact_contract = formal_artifact_requirement_paths(
        artifact_contract_root,
        {
            "final_page_qa_effective": True,
            "final_page_qa_complete": False,
        },
    )
    complete_artifact_contract = formal_artifact_requirement_paths(
        artifact_contract_root,
        {
            "final_page_qa_effective": True,
            "final_page_qa_complete": True,
        },
    )
    with tempfile.TemporaryDirectory() as temporary:
        artifact_root = Path(temporary)
        artifact_result_root = (
            artifact_root / "results" / "qtail_droid_full"
        )
        artifact_result_root.mkdir(parents=True)
        artifact_root.joinpath("qtail-droid-full-training.html").write_text(
            "\n".join(
                [
                    '<a class="artifact" href="passed.json">passed</a>',
                    '<a class="artifact" href="failed.json">failed</a>',
                    '<a class="artifact" href="corrupt.json">corrupt</a>',
                    (
                        '<a class="artifact" '
                        'href="valid_false.json">invalid</a>'
                    ),
                    (
                        '<a class="artifact" '
                        'href="passed_false.json">not-passed</a>'
                    ),
                    (
                        '<a class="artifact" '
                        'href="results/qtail_droid_full/final_page_qa.json">'
                        "qa</a>"
                    ),
                    (
                        '<a class="artifact" '
                        'href="results/qtail_droid_full/'
                        'final_page_desktop.png">desktop</a>'
                    ),
                    (
                        '<a class="artifact" '
                        'href="results/qtail_droid_full/'
                        'final_page_mobile.png">mobile</a>'
                    ),
                ]
            ),
            encoding="utf-8",
        )
        atomic_write_json(
            artifact_root / "passed.json",
            {"status": "passed"},
        )
        atomic_write_json(
            artifact_root / "failed.json",
            {"status": "failed"},
        )
        artifact_root.joinpath("corrupt.json").write_text(
            "{not-json",
            encoding="utf-8",
        )
        atomic_write_json(
            artifact_root / "valid_false.json",
            {"status": "complete", "valid": False},
        )
        atomic_write_json(
            artifact_root / "passed_false.json",
            {"status": "complete", "passed": False},
        )
        atomic_write_json(
            artifact_result_root / "final_page_qa.json",
            {"status": "failed"},
        )
        for filename in ("final_page_desktop.png", "final_page_mobile.png"):
            artifact_result_root.joinpath(filename).write_bytes(b"png")
        failed_projection = artifact_link_availability(artifact_root)
        atomic_write_json(
            artifact_result_root / "final_page_qa.json",
            {"status": "complete"},
        )
        complete_projection = artifact_link_availability(artifact_root)
        failed_items = failed_projection["items"]
        complete_items = complete_projection["items"]
        final_hrefs = {
            "results/qtail_droid_full/final_page_qa.json",
            "results/qtail_droid_full/final_page_desktop.png",
            "results/qtail_droid_full/final_page_mobile.png",
        }
    with tempfile.TemporaryDirectory() as temporary:
        parity_root = Path(temporary)
        parity_source = parity_root / "source"
        parity_source.mkdir()
        parity_file = parity_source / "proof.txt"
        parity_file.write_text("snapshot parity\n", encoding="utf-8")
        parity_manifest = parity_root / "SHA256SUMS"
        parity_digest = hashlib.sha256(parity_file.read_bytes()).hexdigest()
        parity_manifest.write_text(
            f"{parity_digest}  ./proof.txt\n",
            encoding="utf-8",
        )
        parity_positive = verify_snapshot_source_parity(
            parity_source,
            parity_manifest,
        )
        parity_file.write_text("source drift\n", encoding="utf-8")
        parity_drift = verify_snapshot_source_parity(
            parity_source,
            parity_manifest,
        )
        parity_manifest.write_text(
            f"{parity_digest}  ../proof.txt\n",
            encoding="utf-8",
        )
        parity_escape = verify_snapshot_source_parity(
            parity_source,
            parity_manifest,
        )
    controls = [
        {
            "name": "preview_stage_is_qa_in_progress",
            "passed": preview_stage == "final_page_qa",
        },
        {
            "name": "preview_is_not_effective_completion",
            "passed": (
                preview["final_page_qa_effective"] is False
                and preview["final_page_qa_in_progress"] is True
            ),
        },
        {
            "name": "lease_bound_bootstrap_remains_sealing_not_complete",
            "passed": (
                bootstrap_stage == "final_page_qa"
                and bootstrap["final_page_qa_effective"] is False
                and bootstrap["final_page_qa_in_progress"] is True
            ),
        },
        {
            "name": "committed_marker_reaches_complete",
            "passed": (
                committed_stage == "complete"
                and committed["final_page_qa_effective"] is True
            ),
        },
        {
            "name": "sealed_final_without_public_projection_stays_eight_of_nine",
            "passed": (
                sealed_without_projection_stage == "final_page_qa"
                and sealed_without_projection["final_page_qa_effective"] is False
                and sealed_without_projection[
                    "final_page_qa_in_progress"
                ]
                is False
            ),
        },
        {
            "name": "bootstrap_projection_is_not_frozen",
            "passed": not public_final_projection_is_committed(
                complete_latest,
                bootstrap_audit,
            ),
        },
        {
            "name": "committed_projection_can_be_frozen",
            "passed": public_final_projection_is_committed(
                complete_latest,
                committed_audit,
            ),
        },
        {
            "name": "formal_pre_page_artifact_baseline_is_63",
            "passed": (
                len(baseline_artifact_contract["baseline"]) == 63
                and len(baseline_artifact_contract["required"]) == 63
            ),
        },
        {
            "name": "effective_qa_adds_only_nine_process_log_artifacts",
            "passed": (
                effective_artifact_contract["baseline"]
                == baseline_artifact_contract["baseline"]
                and len(effective_artifact_contract["process_logs"]) == 9
                and len(effective_artifact_contract["required"]) == 72
            ),
        },
        {
            "name": (
                "complete_qa_adds_five_final_artifacts_without_baseline_drift"
            ),
            "passed": (
                complete_artifact_contract["baseline"]
                == baseline_artifact_contract["baseline"]
                and len(complete_artifact_contract["final_qa"]) == 5
                and len(complete_artifact_contract["required"]) == 77
            ),
        },
        {
            "name": (
                "workspace_snapshot_parity_accepts_match_and_rejects_drift_or_escape"
            ),
            "passed": (
                parity_positive["verified"] is True
                and parity_positive["verified_file_count"] == 1
                and parity_drift["verified"] is False
                and parity_drift["error_count"] == 1
                and parity_escape["verified"] is False
                and parity_escape["error_count"] == 1
            ),
        },
        {
            "name": "passed_json_artifact_is_ready",
            "passed": failed_items["passed.json"]["available"] is True,
        },
        {
            "name": "failed_corrupt_and_false_json_semantics_are_withheld",
            "passed": (
                failed_items["failed.json"]["available"] is False
                and failed_items["failed.json"]["reason"]
                == "产物语义状态为 failed"
                and failed_items["corrupt.json"]["available"] is False
                and failed_items["corrupt.json"]["reason"]
                == "JSON 产物无法解析"
                and failed_items["valid_false.json"]["available"] is False
                and failed_items["valid_false.json"]["reason"]
                == "产物语义字段 valid=false"
                and failed_items["passed_false.json"]["available"] is False
                and failed_items["passed_false.json"]["reason"]
                == "产物语义字段 passed=false"
            ),
        },
        {
            "name": "failed_final_qa_artifact_family_is_withheld",
            "passed": all(
                failed_items[href]["available"] is False
                and failed_items[href]["reason"] == "最终 QA 尚未成功完成"
                for href in final_hrefs
            ),
        },
        {
            "name": "complete_final_qa_artifact_family_is_ready",
            "passed": all(
                complete_items[href]["available"] is True
                for href in final_hrefs
            ),
        },
    ]
    passed = all(control["passed"] for control in controls)
    payload = {
        "generated_at": now(),
        "status": "passed" if passed else "failed",
        "controls_passed": sum(control["passed"] for control in controls),
        "controls_total": len(controls),
        "controls": controls,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
