#!/usr/bin/env python3
"""Run the Open X trainer with an audited checksum-bound feature cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import qtail_train_openx_demo as trainer
except ImportError:
    import qtail_train_openx_demo as trainer


FORMAT_VERSION = "qtail_openx_feature_cache_v1"
RESUME_FORMAT_VERSION = "qtail_openx_allocation_resume_v1"
THIS_FILE = Path(__file__).resolve()
TRAINER_FILE = Path(trainer.__file__).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cache_path(cache_dir: Path, relative_path: str) -> Path:
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    return cache_dir / key[:2] / f"{key}.json"


def argument_value(name: str, default: str = "") -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resumable_train_head(
    features: np.ndarray,
    target: np.ndarray,
    steps: int,
    seed: int,
    *,
    phase: str,
    resume_dir: Path,
    progress_path: Path,
    stop_after_step: int | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, trainer.AllocationHead]:
    feature_sha = array_sha256(features)
    target_sha = array_sha256(target)
    identity = {
        "format_version": RESUME_FORMAT_VERSION,
        "phase": phase,
        "feature_sha256": feature_sha,
        "target_sha256": target_sha,
        "steps_target": int(steps),
        "seed": int(seed),
        "architecture": "AllocationHead(10-32-16-1)",
        "optimizer": "AdamW(lr=0.002,weight_decay=0.0001)",
        "cached_trainer_sha256": file_sha256(THIS_FILE),
        "base_trainer_sha256": file_sha256(TRAINER_FILE),
    }
    checkpoint_path = resume_dir / f"{phase}.pt"
    torch.manual_seed(seed)
    x = torch.tensor(features)
    y = torch.tensor(target)
    model = trainer.AllocationHead(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=1e-4
    )
    history: list[dict[str, float]] = []
    start_step = 0
    resumed = False
    checkpoint = load_torch_checkpoint(checkpoint_path)
    if checkpoint.get("identity") == identity:
        completed_steps = int(checkpoint.get("completed_steps", -1))
        if 0 <= completed_steps <= steps:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                history = list(checkpoint.get("history", []))
                start_step = completed_steps
                resumed = completed_steps > 0
            except (KeyError, RuntimeError, ValueError):
                start_step = 0
                history = []
                resumed = False

    interval = max(1, steps // 20)
    checkpoint_interval = max(1, min(1000, steps))
    progress_interval = max(1, min(100, steps))
    started = time.monotonic()

    def checkpoint_payload(completed_steps: int) -> dict[str, Any]:
        return {
            "identity": identity,
            "completed_steps": completed_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }

    for step in range(start_step, steps + 1):
        pred = model(x)
        loss = torch.sum(
            y * (torch.log(y + 1e-8) - torch.log(pred + 1e-8))
        )
        if step % interval == 0 and not any(
            int(item.get("step", -1)) == step for item in history
        ):
            history.append({"step": step, "kl": float(loss.detach().cpu())})
        if step == start_step or step == steps or step % progress_interval == 0:
            atomic_json(
                progress_path,
                {
                    "format_version": "qtail_openx_optimizer_progress_v1",
                    "generated_at": now(),
                    "status": "running" if step < steps else "phase_complete",
                    "phase": phase,
                    "step": step,
                    "steps_target": steps,
                    "overall_completed_updates": (
                        step if phase == "source" else steps + step
                    ),
                    "overall_target_updates": steps * 2,
                    "resumed": resumed,
                    "checkpoint_interval": checkpoint_interval,
                    "checkpoint_path": str(checkpoint_path),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "kl": float(loss.detach().cpu()),
                    "feature_sha256": feature_sha,
                    "target_sha256": target_sha,
                },
            )
        if step == steps:
            atomic_torch_save(checkpoint_path, checkpoint_payload(steps))
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed_steps = step + 1
        should_checkpoint = completed_steps % checkpoint_interval == 0
        should_interrupt = (
            stop_after_step is not None and completed_steps == stop_after_step
        )
        if should_checkpoint or should_interrupt:
            atomic_torch_save(
                checkpoint_path, checkpoint_payload(completed_steps)
            )
        if should_interrupt:
            raise RuntimeError(
                f"controlled interruption after {completed_steps} optimizer updates"
            )
    return history, model(x).detach().cpu().numpy(), model


def main() -> None:
    cache_dir = Path(os.environ.get("QTAIL_OPENX_FEATURE_CACHE_DIR", ""))
    ledger_path = Path(os.environ.get("QTAIL_OPENX_CHECKSUM_LEDGER", ""))
    out_dir = Path(argument_value("--out", "results/openx_1t_expansion/training"))
    ledger = load_json(ledger_path, {}) if str(ledger_path) else {}
    ledger_objects = ledger.get("objects", {}) if isinstance(ledger, dict) else {}
    if not isinstance(ledger_objects, dict):
        ledger_objects = {}

    usage: dict[str, Any] = {
        "cached_rows": 0,
        "fresh_rows": 0,
        "rejected_cache_rows": 0,
        "rejection_reasons": {},
    }
    resume_dir = out_dir / "resume_checkpoints"
    progress_path = out_dir / "optimizer_progress.json"
    call_index = 0

    def resumable_train_once(
        features: np.ndarray, target: np.ndarray, steps: int, seed: int
    ) -> tuple[list[dict[str, float]], np.ndarray, trainer.AllocationHead]:
        nonlocal call_index
        call_index += 1
        phase = "source" if call_index == 1 else "qtail" if call_index == 2 else f"head_{call_index}"
        return resumable_train_head(
            features,
            target,
            steps,
            seed,
            phase=phase,
            resume_dir=resume_dir,
            progress_path=progress_path,
        )

    def reject(reason: str) -> None:
        usage["rejected_cache_rows"] += 1
        reasons = usage["rejection_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def cached_build_rows(
        data_dir: Path, records_per_shard: int, max_shards: int = 0
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in trainer.find_shards(data_dir, max_shards=max_shards):
            size = path.stat().st_size
            if size <= 0:
                continue
            relative_path = path.relative_to(data_dir).as_posix()
            entry = ledger_objects.get(relative_path)
            payload = load_json(cache_path(cache_dir, relative_path), {})
            reason = ""
            if not isinstance(entry, dict):
                reason = "not_in_expansion_ledger"
            elif payload.get("format_version") != FORMAT_VERSION:
                reason = "missing_or_wrong_format"
            elif payload.get("feature_extractor_version") != trainer.FEATURE_EXTRACTOR_VERSION:
                reason = "feature_version_mismatch"
            elif int(payload.get("records_per_shard", -1)) != records_per_shard:
                reason = "record_cap_mismatch"
            elif payload.get("relative_path") != relative_path:
                reason = "path_mismatch"
            elif int(payload.get("bytes", -1)) != size:
                reason = "size_mismatch"
            elif int(payload.get("mtime_ns", -1)) != path.stat().st_mtime_ns:
                reason = "mtime_mismatch"
            elif payload.get("official_md5_base64") != entry.get("official_md5_base64"):
                reason = "official_md5_mismatch"
            elif entry.get("official_md5_base64") != entry.get("local_md5_base64"):
                reason = "ledger_not_verified"
            elif not isinstance(payload.get("row"), dict):
                reason = "missing_row"

            if not reason:
                row = dict(payload["row"])
                row["path"] = str(path)
                row["bytes"] = size
                row["log_bytes"] = math.log1p(size)
                rows.append(row)
                usage["cached_rows"] += 1
                continue

            reject(reason)
            shard_idx, shard_total = trainer.shard_coordinates(path.name)
            rows.append(
                {
                    "dataset": trainer.dataset_name(path, data_dir),
                    "path": str(path),
                    "bytes": size,
                    "log_bytes": math.log1p(size),
                    "shard_idx": shard_idx,
                    "shard_total": shard_total,
                    **trainer.aggregate_records(path, records_per_shard),
                }
            )
            usage["fresh_rows"] += 1

        atomic_json(
            out_dir / "feature_cache_usage.json",
            {
                "format_version": "qtail_openx_feature_cache_usage_v1",
                "generated_at": now(),
                "cache_dir": str(cache_dir),
                "ledger": str(ledger_path),
                "ledger_objects": len(ledger_objects),
                "feature_extractor_version": trainer.FEATURE_EXTRACTOR_VERSION,
                "records_per_shard": records_per_shard,
                "rows": len(rows),
                **usage,
            },
        )
        return rows

    trainer.build_rows = cached_build_rows
    trainer.train_once = resumable_train_once
    trainer.main()


if __name__ == "__main__":
    main()
