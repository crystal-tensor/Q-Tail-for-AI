#!/usr/bin/env python3
"""Verify that formal DROID gates precede the first optimizer-backed stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAL_PROTOCOL = "formal_errors = formal_protocol_mismatches("
MIRROR_AUDIT = "input_audit = verified_mirror_audit("
ROW_BUILD = "rows, selected_cache_paths = build_rows_resumable("
RECORD_CLOSURE = (
    "record_closure_errors = formal_record_closure_errors("
)
PARSE_GATE = "if parse_rate < args.min_record_parse_rate:"
SCAN_GATE = (
    "if args.records_per_shard == 0 and scan_complete_rate "
    "< args.min_record_scan_complete_rate:"
)
SPLIT = (
    "deterministic_release_stratified_split(\n"
    "            rows,"
)
HOLDOUT_GATE = (
    "if formal_run:\n"
    "        holdout_by_release = {"
)
TRAINING_MARKER = (
    '(args.marker_dir / "DROID_MODEL_TRAINING_STARTED").touch()'
)
FIRST_TRAIN = (
    "evaluation_source_hist, _, evaluation_source_model, "
    "evaluation_source_resume = train_once_audited("
)
OPTIMIZER_STEP = "optimizer.step()"
FINAL_STEP_BREAK = (
    "if step == steps:\n"
    "            break"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def swap_tokens(source: str, first: str, second: str) -> str:
    if source.count(first) != 1 or source.count(second) != 1:
        raise ValueError("mutation tokens must each occur exactly once")
    placeholder = "__QTAIL_TRAINING_GATE_ORDER_SWAP__"
    if placeholder in source:
        raise ValueError("mutation placeholder collides with source")
    return source.replace(first, placeholder, 1).replace(
        second, first, 1
    ).replace(placeholder, second, 1)


def ordering_errors(source: str) -> list[str]:
    errors: list[str] = []
    main_start = source.find("def main() -> None:")
    if main_start < 0:
        return ["main_function_missing"]
    main_source = source[main_start:]

    markers = [
        ("formal_protocol", FORMAL_PROTOCOL),
        ("verified_mirror", MIRROR_AUDIT),
        ("all_record_rows", ROW_BUILD),
        ("record_closure", RECORD_CLOSURE),
        ("parse_gate", PARSE_GATE),
        ("scan_gate", SCAN_GATE),
        ("holdout_split", SPLIT),
        ("holdout_gate", HOLDOUT_GATE),
        ("training_marker", TRAINING_MARKER),
        ("first_training_stage", FIRST_TRAIN),
    ]
    positions: dict[str, int] = {}
    for name, token in markers:
        count = main_source.count(token)
        if count != 1:
            errors.append(f"{name}:occurrences={count}")
            continue
        positions[name] = main_source.index(token)

    expected_order = [name for name, _ in markers]
    if len(positions) == len(expected_order):
        for earlier, later in zip(expected_order, expected_order[1:]):
            if positions[earlier] >= positions[later]:
                errors.append(f"{earlier}:not_before:{later}")

    train_start = source.find("def train_once_audited(")
    train_end = source.find(
        "\ndef deterministic_release_stratified_split(",
        train_start,
    )
    if train_start < 0 or train_end < 0:
        errors.append("train_once_audited_scope_missing")
    else:
        train_source = source[train_start:train_end]
        if train_source.count(OPTIMIZER_STEP) != 1:
            errors.append(
                "optimizer_step:occurrences="
                f"{train_source.count(OPTIMIZER_STEP)}"
            )
        elif train_source.find(FINAL_STEP_BREAK) < 0:
            errors.append("final_step_break_missing")
        elif train_source.index(FINAL_STEP_BREAK) >= train_source.index(
            OPTIMIZER_STEP
        ):
            errors.append("final_step_break_not_before_optimizer_step")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.trainer.read_text(encoding="utf-8")
    controls: list[dict[str, Any]] = []

    positive_errors = ordering_errors(source)
    controls.append(
        {
            "name": "positive_current_trainer_accepted",
            "expected": "accepted",
            "observed_errors": positive_errors,
            "passed": not positive_errors,
        }
    )

    mutations = [
        (
            "mirror_after_row_build_rejected",
            MIRROR_AUDIT,
            ROW_BUILD,
        ),
        (
            "record_closure_after_parse_gate_rejected",
            RECORD_CLOSURE,
            PARSE_GATE,
        ),
        (
            "parse_after_scan_gate_rejected",
            PARSE_GATE,
            SCAN_GATE,
        ),
        (
            "scan_after_holdout_split_rejected",
            SCAN_GATE,
            SPLIT,
        ),
        (
            "holdout_gate_after_training_marker_rejected",
            HOLDOUT_GATE,
            TRAINING_MARKER,
        ),
        (
            "training_marker_after_first_stage_rejected",
            TRAINING_MARKER,
            FIRST_TRAIN,
        ),
    ]
    for name, first, second in mutations:
        mutated = swap_tokens(source, first, second)
        errors = ordering_errors(mutated)
        controls.append(
            {
                "name": name,
                "expected": "rejected",
                "observed_errors": errors,
                "passed": bool(errors),
            }
        )

    mutated_optimizer = source.replace(
        FINAL_STEP_BREAK,
        "__QTAIL_FINAL_STEP_BREAK__",
        1,
    ).replace(
        OPTIMIZER_STEP,
        FINAL_STEP_BREAK,
        1,
    ).replace(
        "__QTAIL_FINAL_STEP_BREAK__",
        OPTIMIZER_STEP,
        1,
    )
    optimizer_errors = ordering_errors(mutated_optimizer)
    controls.append(
        {
            "name": "optimizer_before_final_step_break_rejected",
            "expected": "rejected",
            "observed_errors": optimizer_errors,
            "passed": bool(optimizer_errors),
        }
    )

    passed = sum(bool(control["passed"]) for control in controls)
    payload = {
        "generated_at": now(),
        "version": "qtail_droid_training_gate_order_selftest_v1",
        "status": "passed" if passed == len(controls) else "failed",
        "trainer": str(args.trainer),
        "trainer_sha256": sha256(args.trainer),
        "contract": (
            "Formal protocol, verified mirror, all-record extraction, exact "
            "record closure, parse/scan coverage, and release-stratified "
            "holdout gates must all pass before the training-start marker and "
            "the first optimizer-backed stage."
        ),
        "controls_passed": passed,
        "controls_total": len(controls),
        "controls": controls,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0 if payload["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
