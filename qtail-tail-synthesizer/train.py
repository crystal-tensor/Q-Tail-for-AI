#!/usr/bin/env python3
"""Train a portable Q-Tail data-allocation model from full DROID features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = [
    "log_source_count",
    "source_share",
    "rarity",
    "failure_proxy",
    "difficulty",
    "task_name_complexity",
    "action_complexity_proxy",
    "terminal_success_proxy",
]
FORMAT_VERSION = "qtail_portable_synthesizer_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(values, [0.01, 0.99])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4096:
        raise ValueError(f"expected 4096 full-DROID rows, found {len(rows)}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    source = np.maximum(col("source_target"), 1e-12)
    source /= source.sum()
    target = np.maximum(col("qtail_target"), 1e-12)
    target /= target.sum()
    counts = np.maximum(col("records_decoded"), 1.0)
    rarity_raw = 1.0 / np.sqrt(source)
    rarity = normalize_01(rarity_raw)
    reward_max = normalize_01(col("mean_reward_max"))
    difficulty = normalize_01(col("deployment_tail_score"))
    instruction = normalize_01(col("mean_instruction_units"))
    action = normalize_01(col("mean_action_std"))
    terminal = normalize_01(col("mean_terminal_rate"))
    features = np.column_stack(
        [
            np.log1p(counts),
            source,
            rarity,
            1.0 - reward_max,
            difficulty,
            instruction,
            action,
            terminal,
        ]
    ).astype(np.float32)
    uplift = np.log(target / source).astype(np.float32)
    return features, uplift, target.astype(np.float32), rows


class TailSynthesisHead(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.SiLU(),
            nn.Linear(width, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_indices(n: int, seed: int = 20260807) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(round(n * 0.8))
    return np.sort(order[:cut]), np.sort(order[cut:])


def train_worker(data: Path, out: Path, seed: int, steps: int) -> None:
    features, uplift, target, rows = load_rows(data)
    train_idx, val_idx = split_indices(len(features))
    mean = features[train_idx].mean(axis=0)
    std = np.maximum(features[train_idx].std(axis=0), 1e-6)
    x = torch.tensor((features - mean) / std, dtype=torch.float32)
    y = torch.tensor(uplift, dtype=torch.float32)
    torch.manual_seed(seed)
    model = TailSynthesisHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[train_tensor])
        loss = torch.mean((prediction - y[train_tensor]) ** 2)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            with torch.no_grad():
                val_loss = torch.mean((model(x[val_idx]) - y[val_idx]) ** 2)
            history.append(
                {
                    "step": step,
                    "train_mse": float(loss.detach()),
                    "validation_mse": float(val_loss),
                }
            )

    with torch.no_grad():
        predicted_uplift = model(x).numpy()
    source = np.asarray([float(row["source_target"]) for row in rows])
    source = np.maximum(source, 1e-12) / np.maximum(source, 1e-12).sum()
    allocation = np.exp(np.log(source) + predicted_uplift)
    allocation /= allocation.sum()
    target64 = target.astype(np.float64)
    kl = float(np.sum(target64 * np.log((target64 + 1e-12) / (allocation + 1e-12))))
    tail_scores = np.asarray([float(row["deployment_tail_score"]) for row in rows])
    tail_mask = tail_scores >= np.quantile(tail_scores, 0.70)
    metrics = {
        "format_version": FORMAT_VERSION,
        "generated_at": now(),
        "seed": seed,
        "steps": steps,
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "validation_mse": history[-1]["validation_mse"],
        "allocation_kl": kl,
        "source_tail_share": float(source[tail_mask].sum()),
        "predicted_tail_share": float(allocation[tail_mask].sum()),
        "target_tail_share": float(target64[tail_mask].sum()),
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "seed": seed,
            "steps": steps,
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        out / "candidate.pt",
    )
    atomic_json(out / "metrics.json", metrics)


def build_bundle(data: Path, out: Path, seeds: list[int], steps: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    workers = []
    for seed in seeds:
        candidate = out / "candidates" / f"seed_{seed}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--data",
            str(data),
            "--out",
            str(candidate),
            "--steps",
            str(steps),
            "--worker-seed",
            str(seed),
        ]
        workers.append((seed, command, subprocess.Popen(command)))
    failures = []
    for seed, command, process in workers:
        code = process.wait()
        if code != 0:
            failures.append({"seed": seed, "returncode": code, "command": command})
    if failures:
        raise RuntimeError(f"candidate workers failed: {failures}")

    candidates = []
    for seed in seeds:
        root = out / "candidates" / f"seed_{seed}"
        checkpoint = torch.load(root / "candidate.pt", map_location="cpu", weights_only=False)
        metrics = json.loads((root / "metrics.json").read_text())
        candidates.append((metrics["validation_mse"], seed, checkpoint, metrics))
    candidates.sort(key=lambda item: item[0])
    _, best_seed, best, best_metrics = candidates[0]
    ensemble_states = [item[2]["state_dict"] for item in candidates]
    bundle_path = out / "production_model.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_class": "TailSynthesisHead",
            "feature_names": FEATURE_NAMES,
            "normalization_mean": best["normalization_mean"],
            "normalization_std": best["normalization_std"],
            "best_seed": best_seed,
            "ensemble_seeds": [item[1] for item in candidates],
            "ensemble_state_dicts": ensemble_states,
            "training_source": str(data),
            "training_source_sha256": sha256(data),
            "training_rows": 4096,
            "training_steps_per_candidate": steps,
            "learned_tail_gain": max(
                0.0,
                float(best_metrics["target_tail_share"])
                - float(best_metrics["source_tail_share"]),
            ),
            "claim_boundary": (
                "This model learns DROID-derived long-tail allocation uplift. It generates "
                "allocation targets and resampled scenario rows, not new robot sensor frames."
            ),
        },
        bundle_path,
    )
    report = {
        "format_version": FORMAT_VERSION,
        "generated_at": now(),
        "status": "complete",
        "model": str(bundle_path),
        "model_sha256": sha256(bundle_path),
        "best_seed": best_seed,
        "best_metrics": best_metrics,
        "candidate_count": len(candidates),
        "candidates": [item[3] for item in candidates],
        "training_source": str(data),
        "training_source_sha256": sha256(data),
        "training_rows": 4096,
        "parallel_python_workers": len(seeds),
        "learned_tail_gain_pp": 100.0
        * max(
            0.0,
            float(best_metrics["target_tail_share"])
            - float(best_metrics["source_tail_share"]),
        ),
    }
    atomic_json(out / "training_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", default="11,29,47,83")
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    if args.worker_seed is not None:
        train_worker(args.data, args.out, args.worker_seed, args.steps)
    else:
        seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
        build_bundle(args.data, args.out, seeds, args.steps)


if __name__ == "__main__":
    main()
