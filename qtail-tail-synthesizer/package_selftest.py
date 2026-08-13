#!/usr/bin/env python3
"""End-to-end positive and negative controls for the production package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def invoke(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *map(str, arguments)],
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    synthesize = package / "synthesize.py"
    validate = package / "validate.py"
    controls: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="qtail-synth-selftest-") as tmp:
        root = Path(tmp)
        cases = [
            (
                "canonical",
                ["task", "count", "success_rate", "difficulty"],
                [
                    ["common_pick", 1000, 0.94, 0.12],
                    ["common_place", 700, 0.90, 0.18],
                    ["occluded_pick", 45, 0.58, 0.75],
                    ["slippery_place", 18, 0.42, 0.88],
                    ["rare_recovery", 5, 0.20, 0.96],
                ],
                997,
            ),
            (
                "semantic_aliases",
                ["instruction", "trajectories", "pass_rate", "risk"],
                [
                    ["open drawer", 250, 0.88, 0.20],
                    ["recover dropped tool", 4, 0.22, 0.92],
                    ["insert deformable item", 9, 0.35, 0.84],
                    ["close drawer", 180, 0.91, 0.16],
                ],
                73,
            ),
            (
                "rarity_proxy_fallback",
                ["scenario", "frequency"],
                [
                    ["routine", 500],
                    ["rare_contact", 7],
                    ["extreme_recovery", 1],
                    ["medium", 90],
                ],
                41,
            ),
        ]
        allocations: dict[str, Path] = {}
        for name, fields, rows, budget in cases:
            source = root / f"{name}.csv"
            result = root / name
            write_csv(source, fields, rows)
            process = invoke(
                synthesize,
                "--model",
                args.model,
                "--source",
                source,
                "--out",
                result,
                "--budget",
                budget,
                "--materialize",
            )
            check = invoke(validate, "--result", result, "--budget", budget)
            report = json.loads((result / "synthesis_report.json").read_text())
            passed = bool(
                process.returncode == 0
                and check.returncode == 0
                and report.get("status") == "complete"
                and int(report.get("materialized_rows", -1)) == budget
                and abs(float(report.get("allocation_sum", 0.0)) - 1.0) < 1e-12
                and float(report.get("tail_share_gain_pp", 0.0)) > 0.0
            )
            allocations[name] = result / "qtail_synthetic_allocation.csv"
            controls.append(
                {
                    "name": name,
                    "passed": passed,
                    "budget": budget,
                    "tail_share_gain_pp": report.get("tail_share_gain_pp"),
                    "detected_columns": report.get("source_meta", {}).get("detected"),
                }
            )

        repeat = root / "canonical_repeat"
        process = invoke(
            synthesize,
            "--model",
            args.model,
            "--source",
            root / "canonical.csv",
            "--out",
            repeat,
            "--budget",
            997,
            "--materialize",
        )
        repeated_allocation = repeat / "qtail_synthetic_allocation.csv"
        controls.append(
            {
                "name": "deterministic_allocation_repeat",
                "passed": process.returncode == 0
                and sha256(allocations["canonical"]) == sha256(repeated_allocation),
                "first_sha256": sha256(allocations["canonical"]),
                "repeat_sha256": sha256(repeated_allocation),
            }
        )

        empty = root / "empty.csv"
        empty.write_text("task,count\n", encoding="utf-8")
        empty_process = invoke(
            synthesize,
            "--model",
            args.model,
            "--source",
            empty,
            "--out",
            root / "empty_result",
            "--budget",
            10,
        )
        controls.append(
            {
                "name": "empty_source_rejected",
                "passed": empty_process.returncode != 0,
                "returncode": empty_process.returncode,
            }
        )

        tampered_model = root / "tampered_model.pt"
        bundle = torch.load(args.model, map_location="cpu", weights_only=False)
        bundle["feature_names"] = ["tampered"]
        torch.save(bundle, tampered_model)
        tampered_process = invoke(
            synthesize,
            "--model",
            tampered_model,
            "--source",
            root / "canonical.csv",
            "--out",
            root / "tampered_result",
            "--budget",
            10,
        )
        controls.append(
            {
                "name": "tampered_model_contract_rejected",
                "passed": tampered_process.returncode != 0
                and "contract mismatch" in tampered_process.stderr,
                "returncode": tampered_process.returncode,
            }
        )

    passed = sum(bool(item["passed"]) for item in controls)
    report = {
        "format_version": "qtail_portable_synthesizer_package_selftest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed == len(controls) else "failed",
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "controls_passed": passed,
        "controls_total": len(controls),
        "controls": controls,
        "claim_boundary": (
            "These controls validate CSV schema adaptation, allocation determinism, "
            "budget conservation, and model-contract rejection. They do not validate "
            "downstream sensor or trajectory rendering."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
