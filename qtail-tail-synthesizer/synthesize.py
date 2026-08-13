#!/usr/bin/env python3
"""Apply a trained Q-Tail model to a new source CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train import FEATURE_NAMES, FORMAT_VERSION, TailSynthesisHead


TASK_KEYS = ("task", "task_id", "skill", "instruction", "scenario", "name")
COUNT_KEYS = ("count", "episodes", "trajectories", "samples", "frequency", "n")
SUCCESS_KEYS = ("success", "success_rate", "reward", "score", "pass_rate")
DIFFICULTY_KEYS = ("difficulty", "risk", "failure_rate", "tail_score", "rarity")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.strip().lower(): column for column in columns}
    return next((lookup[key] for key in candidates if key in lookup), None)


def number(value: object, default: float) -> float:
    try:
        text = str(value).strip().replace("%", "")
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return values / values.sum()


def normalize_01(values: np.ndarray) -> np.ndarray:
    low, high = np.min(values), np.max(values)
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def read_source(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        raw = list(reader)
    if not raw:
        raise ValueError("source CSV is empty")
    task_col = first(columns, TASK_KEYS) or columns[0]
    count_col = first(columns, COUNT_KEYS)
    success_col = first(columns, SUCCESS_KEYS)
    difficulty_col = first(columns, DIFFICULTY_KEYS)
    grouped: dict[str, dict] = {}
    for index, row in enumerate(raw):
        task = str(row.get(task_col, "")).strip() or f"row_{index}"
        count = max(number(row.get(count_col), 1.0) if count_col else 1.0, 1e-9)
        success = number(row.get(success_col), math.nan) if success_col else math.nan
        difficulty = number(row.get(difficulty_col), math.nan) if difficulty_col else math.nan
        item = grouped.setdefault(
            task,
            {"task_id": task, "count": 0.0, "success_sum": 0.0, "success_n": 0, "difficulty_sum": 0.0, "difficulty_n": 0, "source_row": row},
        )
        item["count"] += count
        if math.isfinite(success):
            success = success / 100.0 if success > 1 else success
            item["success_sum"] += min(max(success, 0.0), 1.0)
            item["success_n"] += 1
        if math.isfinite(difficulty):
            difficulty = difficulty / 10.0 if 1 < difficulty <= 10 else difficulty / 100.0 if difficulty > 10 else difficulty
            item["difficulty_sum"] += min(max(difficulty, 0.0), 1.0)
            item["difficulty_n"] += 1
    return list(grouped.values()), {
        "columns": columns,
        "rows": len(raw),
        "tasks": len(grouped),
        "detected": {"task": task_col, "count": count_col, "success": success_col, "difficulty": difficulty_col},
    }


def profiles(items: list[dict]) -> tuple[np.ndarray, list[dict]]:
    counts = np.asarray([item["count"] for item in items], dtype=np.float64)
    shares = normalize(counts)
    rarity = normalize_01(1.0 / np.sqrt(shares + 1e-12))
    result = []
    features = []
    for index, item in enumerate(items):
        success = item["success_sum"] / item["success_n"] if item["success_n"] else max(0.1, 0.9 - 0.55 * rarity[index])
        difficulty = item["difficulty_sum"] / item["difficulty_n"] if item["difficulty_n"] else rarity[index]
        name_complexity = min(len(item["task_id"].encode("utf-8")) / 96.0, 1.0)
        failure = 1.0 - success
        tail_score = 0.40 * rarity[index] + 0.30 * difficulty + 0.20 * failure + 0.10 * name_complexity
        features.append([math.log1p(counts[index]), shares[index], rarity[index], failure, difficulty, name_complexity, difficulty, success])
        result.append({**item, "source_share": float(shares[index]), "rarity": float(rarity[index]), "success_rate": float(success), "difficulty": float(difficulty), "tail_score": float(tail_score)})
    return np.asarray(features, dtype=np.float32), result


def largest_remainder(weights: np.ndarray, budget: int) -> np.ndarray:
    raw = weights * budget
    counts = np.floor(raw).astype(np.int64)
    remaining = budget - int(counts.sum())
    if remaining:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    return counts


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(model_path: Path, source_path: Path, out: Path, budget: int, materialize: bool) -> dict:
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model bundle contract mismatch")
    items, source_meta = read_source(source_path)
    feature_values, task_profiles = profiles(items)
    mean = np.asarray(bundle["normalization_mean"], dtype=np.float32)
    std = np.asarray(bundle["normalization_std"], dtype=np.float32)
    x = torch.tensor((feature_values - mean) / std, dtype=torch.float32)
    predictions = []
    for state in bundle["ensemble_state_dicts"]:
        model = TailSynthesisHead()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions.append(model(x).numpy())
    uplift = np.mean(predictions, axis=0)
    source_share = np.asarray([profile["source_share"] for profile in task_profiles])
    learned = np.exp(np.log(source_share + 1e-12) + np.clip(uplift, -8.0, 8.0))
    learned = normalize(learned)
    weights = normalize(0.85 * learned + 0.15 * source_share)
    tail_values = np.asarray([profile["tail_score"] for profile in task_profiles])
    tail_mask = tail_values >= np.quantile(tail_values, 0.70)
    source_tail_share = float(source_share[tail_mask].sum())
    learned_tail_gain = min(max(float(bundle.get("learned_tail_gain", 0.10)), 0.05), 0.35)
    calibrated_tail_share = min(0.85, source_tail_share + learned_tail_gain)
    raw_tail_share = float(weights[tail_mask].sum())
    calibration_mix = 0.0
    if raw_tail_share < calibrated_tail_share:
        tail_only = np.zeros_like(weights)
        tail_only[tail_mask] = normalize(weights[tail_mask])
        calibration_mix = (calibrated_tail_share - raw_tail_share) / max(
            1.0 - raw_tail_share, 1e-12
        )
        weights = normalize((1.0 - calibration_mix) * weights + calibration_mix * tail_only)
    counts = largest_remainder(weights, budget)
    allocation = []
    for profile, weight, count in zip(task_profiles, weights, counts):
        allocation.append({
            "task_id": profile["task_id"],
            "source_count": profile["count"],
            "source_share": profile["source_share"],
            "synthetic_count": int(count),
            "synthetic_share": float(weight),
            "tail_score": profile["tail_score"],
            "rarity": profile["rarity"],
            "success_rate_reference": profile["success_rate"],
            "difficulty": profile["difficulty"],
            "allocation_gain_pp": float((weight - profile["source_share"]) * 100),
        })
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "qtail_synthetic_allocation.csv", allocation)
    materialized_rows = []
    if materialize:
        serial = 0
        for profile, count in zip(task_profiles, counts):
            for replica in range(int(count)):
                serial += 1
                materialized_rows.append({
                    "qtail_synthetic_id": f"qtail_{serial:09d}",
                    "task_id": profile["task_id"],
                    "replica": replica + 1,
                    "tail_score": profile["tail_score"],
                    "source": "qtail_trained_resample",
                    **{f"source_{key}": value for key, value in profile["source_row"].items()},
                })
        write_csv(out / "qtail_synthetic_data.csv", materialized_rows)
    report = {
        "format_version": FORMAT_VERSION,
        "generated_at": now(),
        "status": "complete",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_meta": source_meta,
        "synthetic_budget": budget,
        "allocation_sum": float(weights.sum()),
        "materialized_rows": len(materialized_rows),
        "source_tail_share": source_tail_share,
        "synthetic_tail_share": float(weights[tail_mask].sum()),
        "tail_share_gain_pp": float((weights[tail_mask].sum() - source_tail_share) * 100),
        "learned_tail_gain_target_pp": learned_tail_gain * 100.0,
        "calibration_mix": float(calibration_mix),
        "artifacts": {
            "allocation": str(out / "qtail_synthetic_allocation.csv"),
            "synthetic_data": str(out / "qtail_synthetic_data.csv") if materialize else None,
        },
        "claim_boundary": "Rows are trained long-tail allocations and deterministic resamples. Sensor/image/trajectory rendering requires a downstream domain generator.",
    }
    temporary = out / ".synthesis_report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, out / "synthesis_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    report = run(args.model, args.source, args.out, args.budget, args.materialize)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
