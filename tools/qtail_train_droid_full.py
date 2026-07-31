#!/usr/bin/env python3
"""Run resumable, all-record Q-Tail allocation training on a full DROID mirror.

The data phase streams every decodable record from every complete TFRecord
shard and caches one audited feature row per shard. The model phase then trains
source and Q-Tail allocation heads with identical architecture, optimizer,
device, seed, and step budget. This is allocation-model evidence, not a claim
of end-to-end robot-policy success.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import os
import pickle
import platform
import shutil
import struct
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import qtail_train_openx_demo as base
from qtail_verify_droid_download_marker import (
    MARKER_VERSION as DOWNLOAD_MARKER_VERSION,
)
from qtail_verify_droid_download_marker import build_binding


BOOTSTRAP_METHOD = (
    "paired_release_stratified_shard_bootstrap_"
    "within_draw_renormalization"
)
BOOTSTRAP_STRATA = ("1.0.0", "1.0.1")
OPTIMIZER_SIGNATURE = "AdamW(lr=0.002, weight_decay=0.0001)"
CHECKPOINT_FORMAT_VERSION = 5
CHECKPOINT_CHAIN_VERSION = "sha256_parent_v1"
OPTIMIZER_UPDATE_SEMANTICS = (
    "Checkpoint step k stores the state after exactly k optimizer updates; "
    "each stage ends at k=steps."
)
FORMAL_SEED = 11
FORMAL_HOLDOUT_FRACTION = 0.20
FORMAL_HOLDOUT_SHARDS_PER_RELEASE = 410
FORMAL_HOLDOUT_RELATIVE_PATH_SHA256 = (
    "16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767"
)
FORMAL_STEPS_PER_STAGE = 20_000
FORMAL_CHECKPOINT_EVERY_STEPS = 5_000
FORMAL_BOOTSTRAP_SAMPLES = 5_000
FORMAL_BOOTSTRAP_SEED = 20260727
FORMAL_RANDOMIZATION_SAMPLES = 5_000
FORMAL_RANDOMIZATION_SEED = 20260728
FORMAL_MIN_RECORD_PARSE_RATE = 1.0
FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE = 1.0
FORMAL_PT_SOURCE_SHA256 = (
    "59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f"
)
FORMAL_EXPECTED_OBJECTS = 4_102
FORMAL_EXPECTED_TFRECORDS = 4_096
FORMAL_EXPECTED_BYTES = 3_700_745_265_151
FORMAL_EXPECTED_RECORDS = 187_891
FORMAL_RELEASE_CONTRACT = {
    "1.0.0": {"tfrecord_shards": 2_048, "records": 92_233},
    "1.0.1": {"tfrecord_shards": 2_048, "records": 95_658},
}
RARE_INSTRUCTION_MAX_TRAIN_DF = 1
RARE_COVERAGE_BUDGETS = (10, 25, 50, 100, 200, 400, 800)
RARE_COVERAGE_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
RARE_COVERAGE_MAX_SEARCH_DRAWS = 1_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_mount(path: Path | None) -> None:
    if path is not None and not os.path.ismount(path):
        raise RuntimeError(f"Required mount is unavailable: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise SystemExit(
            f"DROID trainer writer lock is already held: {path} owner={owner}"
        )
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired_at": now(),
                "script": str(Path(__file__).resolve()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def official_md5_byte_audit(
    *,
    data_dir: Path,
    checksum_manifest_path: Path,
    status_path: Path | None = None,
) -> dict[str, Any]:
    checksum_manifest = json.loads(
        checksum_manifest_path.read_text(encoding="utf-8")
    )
    objects = checksum_manifest.get("objects", [])
    errors: list[dict[str, Any]] = []
    hashed_objects = 0
    hashed_bytes = 0
    started_at = now()
    started_monotonic = time.monotonic()

    def publish_status(
        *,
        status: str,
        current_relative_path: str | None,
    ) -> None:
        if status_path is None:
            return
        elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
        atomic_write_json(
            status_path,
            {
                "generated_at": now(),
                "started_at": started_at,
                "status": status,
                "method": (
                    "full_local_byte_md5_rehash_against_official_manifest"
                ),
                "processed_objects": hashed_objects,
                "total_objects": len(objects),
                "processed_bytes": hashed_bytes,
                "total_bytes": FORMAL_EXPECTED_BYTES,
                "progress_percent": (
                    hashed_bytes / FORMAL_EXPECTED_BYTES * 100.0
                    if FORMAL_EXPECTED_BYTES
                    else 0.0
                ),
                "elapsed_seconds": elapsed_seconds,
                "throughput_bytes_per_second": (
                    hashed_bytes / elapsed_seconds
                    if elapsed_seconds > 0
                    else 0.0
                ),
                "current_relative_path": current_relative_path,
                "error_count": len(errors),
                "error_sample": errors[:20],
                "formal_training_started": False,
            },
        )

    publish_status(status="rehashing", current_relative_path=None)
    for item in sorted(
        objects,
        key=lambda value: str(value.get("relative_path", "")),
    ):
        relative = str(item.get("relative_path", ""))
        expected_bytes = int(item.get("bytes", -1))
        expected_md5 = str(item.get("md5_base64", ""))
        path = data_dir / relative
        try:
            stat = path.stat()
        except OSError as error:
            errors.append(
                {
                    "relative_path": relative,
                    "error": f"unreadable:{type(error).__name__}",
                }
            )
            continue
        if not path.is_file() or stat.st_size != expected_bytes:
            errors.append(
                {
                    "relative_path": relative,
                    "error": "missing_or_size_mismatch",
                    "expected_bytes": expected_bytes,
                    "actual_bytes": stat.st_size,
                }
            )
            continue
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(16 * 1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        actual_md5 = base64.b64encode(digest.digest()).decode("ascii")
        hashed_objects += 1
        hashed_bytes += stat.st_size
        if actual_md5 != expected_md5:
            errors.append(
                {
                    "relative_path": relative,
                    "error": "local_byte_md5_mismatch",
                    "expected_md5_base64": expected_md5,
                    "actual_md5_base64": actual_md5,
                }
            )
        if hashed_objects % 10 == 0 or hashed_objects == len(objects):
            publish_status(
                status="rehashing",
                current_relative_path=relative,
            )
    verified = bool(
        checksum_manifest.get("status") in {"verified", "complete"}
        and len(objects) == FORMAL_EXPECTED_OBJECTS
        and hashed_objects == FORMAL_EXPECTED_OBJECTS
        and hashed_bytes == FORMAL_EXPECTED_BYTES
        and not errors
    )
    result = {
        "verified": verified,
        "method": "full_local_byte_md5_rehash_against_official_manifest",
        "hashed_objects": hashed_objects,
        "expected_objects": FORMAL_EXPECTED_OBJECTS,
        "hashed_bytes": hashed_bytes,
        "expected_bytes": FORMAL_EXPECTED_BYTES,
        "error_count": len(errors),
        "error_sample": errors[:20],
    }
    publish_status(
        status="complete" if verified else "failed",
        current_relative_path=None,
    )
    return result


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def hardware_model() -> str:
    candidates = (
        Path("/sys/devices/virtual/dmi/id/product_name"),
        Path("/sys/firmware/devicetree/base/model"),
    )
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip().strip("\x00")
        except OSError:
            continue
        if value:
            return value
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.model"],
                check=False,
                capture_output=True,
                text=True,
            )
            value = completed.stdout.strip()
            if completed.returncode == 0 and value:
                return value
        except OSError:
            pass
    return platform.machine() or "unknown"


def runtime_environment_contract(device: torch.device) -> dict[str, Any]:
    return {
        "version": "qtail_runtime_environment_v1",
        "hardware_model": hardware_model(),
        "cpu_architecture": platform.machine() or "unknown",
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "device": str(device),
    }


def environment_fingerprint(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def training_signature(
    features: np.ndarray,
    target: np.ndarray,
    *,
    label: str,
    steps: int,
    seed: int,
    environment_sha256: str,
) -> str:
    digest = hashlib.sha256()
    architecture = f"AllocationHead({features.shape[1]}-32-16-1)"
    for value in (
        label,
        str(steps),
        str(seed),
        architecture,
        "AdamW-0.002-0.0001",
        environment_sha256,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for array in (features, target):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_dict_fingerprint(
    state_dict: dict[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_fingerprint(value: Any) -> str:
    """Hash nested optimizer state without relying on pickle byte stability."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            array = tensor.numpy()
            digest.update(b"tensor\0")
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
            return
        if isinstance(item, dict):
            digest.update(b"dict\0")
            ordered = sorted(
                item.items(),
                key=lambda pair: (
                    type(pair[0]).__module__,
                    type(pair[0]).__qualname__,
                    repr(pair[0]),
                ),
            )
            digest.update(str(len(ordered)).encode("ascii"))
            digest.update(b"\0")
            for key, child in ordered:
                update(key)
                update(child)
            return
        if isinstance(item, list):
            digest.update(b"list\0")
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for child in item:
                update(child)
            return
        if isinstance(item, tuple):
            digest.update(b"tuple\0")
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for child in item:
                update(child)
            return
        if item is None:
            digest.update(b"none\0")
            return
        if isinstance(item, bool):
            digest.update(b"bool\0")
            digest.update(b"1" if item else b"0")
            return
        if isinstance(item, int):
            digest.update(b"int\0")
            digest.update(str(item).encode("ascii"))
            digest.update(b"\0")
            return
        if isinstance(item, float):
            digest.update(b"float\0")
            digest.update(struct.pack(">d", item))
            return
        if isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"str\0")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            return
        if isinstance(item, (bytes, bytearray)):
            encoded = bytes(item)
            digest.update(b"bytes\0")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            return
        raise TypeError(
            "Unsupported checkpoint fingerprint value: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    update(value)
    return digest.hexdigest()


def checkpoint_expected_steps(
    steps: int,
    checkpoint_every_steps: int,
) -> list[int]:
    interval = max(1, checkpoint_every_steps)
    return sorted({*range(0, steps + 1, interval), steps})


def checkpoint_path_for(
    checkpoint_dir: Path,
    label: str,
    step: int,
) -> Path:
    return checkpoint_dir / f"{label}_step_{step:06d}.pt"


def checkpoint_content_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        errors.append("format_version")
    if payload.get("checkpoint_chain_version") != CHECKPOINT_CHAIN_VERSION:
        errors.append("checkpoint_chain_version")
    state_dict = payload.get("state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, dict):
        errors.append("state_dict")
    else:
        try:
            observed = state_dict_fingerprint(state_dict)
        except (TypeError, ValueError, RuntimeError) as error:
            errors.append(f"model_state_unhashable:{type(error).__name__}")
        else:
            if observed != payload.get("model_state_sha256"):
                errors.append("model_state_sha256")
    if not isinstance(optimizer_state, dict):
        errors.append("optimizer_state_dict")
    else:
        try:
            observed = tree_fingerprint(optimizer_state)
        except (TypeError, ValueError, RuntimeError) as error:
            errors.append(f"optimizer_state_unhashable:{type(error).__name__}")
        else:
            if observed != payload.get("optimizer_state_sha256"):
                errors.append("optimizer_state_sha256")
    environment_contract = payload.get("environment_contract")
    if not isinstance(environment_contract, dict):
        errors.append("environment_contract")
    elif environment_fingerprint(environment_contract) != payload.get(
        "environment_fingerprint"
    ):
        errors.append("environment_contract_fingerprint")
    return errors


def boundary_sha256(path: Path, window_bytes: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(16, "big", signed=False))
    with path.open("rb") as handle:
        digest.update(handle.read(window_bytes))
        if size > window_bytes:
            handle.seek(max(0, size - window_bytes))
            digest.update(handle.read(window_bytes))
    return digest.hexdigest()


def shard_identity(path: Path, data_dir: Path, records_per_shard: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": str(path.relative_to(data_dir)),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "boundary_sha256": boundary_sha256(path),
        "records_per_shard": records_per_shard,
        "feature_extractor_version": base.FEATURE_EXTRACTOR_VERSION,
    }


def cache_path(cache_dir: Path, identity: dict[str, Any]) -> Path:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"


def build_row(
    path: Path,
    data_dir: Path,
    records_per_shard: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    size = path.stat().st_size
    shard_idx, shard_total = base.shard_coordinates(path.name)
    return {
        "dataset": base.dataset_name(path, data_dir),
        "path": str(path),
        "bytes": size,
        "log_bytes": float(np.log1p(size)),
        "shard_idx": shard_idx,
        "shard_total": shard_total,
        "boundary_sha256": identity["boundary_sha256"],
        **base.aggregate_records(path, records_per_shard),
    }


def build_rows_resumable(
    *,
    data_dir: Path,
    shards: list[Path],
    out: Path,
    records_per_shard: int,
    status_every_shards: int,
    status_label: str = "extracting_all_records",
    required_mount: Path | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    cache_dir = out / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = out / "droid_feature_extraction_status.json"
    partial_rows_path = out / "droid_shard_features.partial.csv"
    total_bytes = sum(path.stat().st_size for path in shards)
    rows: list[dict[str, Any]] = []
    selected_cache_paths: list[Path] = []
    cache_hits = 0

    for index, path in enumerate(shards, start=1):
        require_mount(required_mount)
        identity = shard_identity(path, data_dir, records_per_shard)
        cached_path = cache_path(cache_dir, identity)
        row: dict[str, Any] | None = None
        if cached_path.exists():
            try:
                cached = json.loads(cached_path.read_text(encoding="utf-8"))
                if cached.get("identity") == identity and isinstance(cached.get("row"), dict):
                    row = cached["row"]
                    cache_hits += 1
            except (OSError, json.JSONDecodeError):
                row = None
        if row is None:
            row = build_row(path, data_dir, records_per_shard, identity)
            require_mount(required_mount)
            atomic_write_json(
                cached_path,
                {"generated_at": now(), "identity": identity, "row": row},
            )
        rows.append(row)
        selected_cache_paths.append(cached_path)

        should_publish = index == len(shards) or index % max(1, status_every_shards) == 0
        if should_publish:
            represented_bytes = sum(int(item["bytes"]) for item in rows)
            decoded = sum(int(item["records_decoded"]) for item in rows)
            parsed = sum(int(item["record_parse_ok"]) for item in rows)
            scan_complete = sum(int(item.get("record_scan_complete", 0)) for item in rows)
            errors = sum(bool(item.get("record_parse_error")) for item in rows)
            atomic_write_json(
                status_path,
                {
                    "generated_at": now(),
                    "status": status_label,
                    "full_record_mode": records_per_shard == 0,
                    "processed_shards": index,
                    "total_shards": len(shards),
                    "progress_percent": index / len(shards) * 100.0,
                    "represented_bytes": represented_bytes,
                    "total_tfrecord_bytes": total_bytes,
                    "represented_percent": represented_bytes / max(total_bytes, 1) * 100.0,
                    "records_decoded": decoded,
                    "parsed_shards": parsed,
                    "record_scan_complete_shards": scan_complete,
                    "parse_errors": errors,
                    "cache_hits": cache_hits,
                    "current_shard": str(path.relative_to(data_dir)),
                    "partial_rows": str(partial_rows_path),
                },
            )
            if index == len(shards) or index % max(100, status_every_shards) == 0:
                write_csv(partial_rows_path, rows)

    write_csv(out / "droid_shard_features.csv", rows)
    return rows, selected_cache_paths


def load_valid_resume_chain(
    *,
    checkpoint_dir: Path,
    label: str,
    candidate_step: int,
    expected_steps: list[int],
    steps: int,
    seed: int,
    device: torch.device,
    environment_sha256: str,
    signature: str,
    feature_sha256: str,
    initialized_state_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if candidate_step not in expected_steps:
        return None, [f"unexpected_candidate_step:{candidate_step}"]
    previous_path: Path | None = None
    previous_step: int | None = None
    previous_sha256: str | None = None
    candidate_payload: dict[str, Any] | None = None
    for step in [value for value in expected_steps if value <= candidate_step]:
        path = checkpoint_path_for(checkpoint_dir, label, step)
        if not path.is_file():
            errors.append(f"missing_ancestor:{path.name}")
            break
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            KeyError,
            ValueError,
            TypeError,
        ) as error:
            errors.append(
                f"unreadable:{path.name}:{type(error).__name__}"
            )
            break
        if not isinstance(payload, dict):
            errors.append(f"payload_not_dict:{path.name}")
            break
        content_errors = checkpoint_content_errors(payload)
        expected_parent_name = previous_path.name if previous_path else None
        metadata_valid = (
            payload.get("model") == label
            and int(payload.get("step", -1)) == step
            and int(payload.get("optimizer_updates_completed", -1)) == step
            and int(payload.get("steps", -1)) == steps
            and int(payload.get("seed", -1)) == seed
            and payload.get("device") == str(device)
            and payload.get("optimizer") == OPTIMIZER_SIGNATURE
            and payload.get("environment_fingerprint")
            == environment_sha256
            and payload.get("training_signature") == signature
            and payload.get("feature_sha256") == feature_sha256
            and payload.get("initialized_state_sha256")
            == initialized_state_sha256
            and payload.get("parent_checkpoint_name")
            == expected_parent_name
            and payload.get("parent_checkpoint_step") == previous_step
            and payload.get("parent_checkpoint_sha256")
            == previous_sha256
            and (
                step != 0
                or payload.get("model_state_sha256")
                == initialized_state_sha256
            )
        )
        if content_errors or not metadata_valid:
            detail = ",".join(content_errors) or "metadata_or_parent"
            errors.append(f"invalid:{path.name}:{detail}")
            break
        previous_path = path
        previous_step = step
        previous_sha256 = file_sha256(path)
        candidate_payload = payload
    if errors or previous_step != candidate_step:
        return None, errors or ["candidate_chain_incomplete"]
    return candidate_payload, []


def train_once_audited(
    *,
    features: np.ndarray,
    target: np.ndarray,
    steps: int,
    seed: int,
    label: str,
    device: torch.device,
    out: Path,
    checkpoint_every_steps: int,
    completed_models: list[str],
    environment_contract: dict[str, Any],
) -> tuple[
    list[dict[str, float]],
    np.ndarray,
    base.AllocationHead,
    dict[str, Any],
]:
    torch.manual_seed(seed)
    x = torch.tensor(features, device=device)
    y = torch.tensor(target, device=device)
    model = base.AllocationHead(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    feature_sha256 = array_fingerprint(features)
    initialized_state_sha256 = state_dict_fingerprint(model.state_dict())
    history: list[dict[str, float]] = []
    checkpoint_dir = out / "intermediate_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    status_path = out / "droid_model_training_status.json"
    history_interval = max(1, steps // 20)
    checkpoint_interval = max(1, checkpoint_every_steps)
    expected_checkpoint_steps = checkpoint_expected_steps(
        steps,
        checkpoint_every_steps,
    )
    signature = training_signature(
        features,
        target,
        label=label,
        steps=steps,
        seed=seed,
        environment_sha256=environment_fingerprint(environment_contract),
    )
    environment_sha256 = environment_fingerprint(environment_contract)
    start_step = 0
    resumed_from_checkpoint: str | None = None
    resumed_checkpoint_device: str | None = None
    resumed_checkpoint_optimizer: str | None = None
    resumed_checkpoint_environment_fingerprint: str | None = None
    resume_rejections: list[dict[str, Any]] = []
    last_checkpoint_path: Path | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_sha256: str | None = None
    resume_candidates = [
        (step, checkpoint_path_for(checkpoint_dir, label, step))
        for step in reversed(expected_checkpoint_steps)
        if step > 0
        and checkpoint_path_for(checkpoint_dir, label, step).is_file()
    ]
    for candidate_step, candidate in resume_candidates:
        try:
            payload, chain_errors = load_valid_resume_chain(
                checkpoint_dir=checkpoint_dir,
                label=label,
                candidate_step=candidate_step,
                expected_steps=expected_checkpoint_steps,
                steps=steps,
                seed=seed,
                device=device,
                environment_sha256=environment_sha256,
                signature=signature,
                feature_sha256=feature_sha256,
                initialized_state_sha256=initialized_state_sha256,
            )
            if payload is None:
                resume_rejections.append(
                    {
                        "checkpoint": str(candidate),
                        "errors": chain_errors,
                    }
                )
                continue
            model.load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            history = list(payload.get("history", []))
            start_step = int(payload["step"])
            resumed_from_checkpoint = str(candidate)
            resumed_checkpoint_device = str(payload["device"])
            resumed_checkpoint_optimizer = str(payload["optimizer"])
            resumed_checkpoint_environment_fingerprint = str(
                payload["environment_fingerprint"]
            )
            last_checkpoint_path = candidate
            last_checkpoint_step = candidate_step
            last_checkpoint_sha256 = file_sha256(candidate)
            break
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            KeyError,
            ValueError,
            TypeError,
        ) as error:
            resume_rejections.append(
                {
                    "checkpoint": str(candidate),
                    "errors": [
                        f"state_restore:{type(error).__name__}:{error}"
                    ],
                }
            )
            continue

    for step in range(start_step, steps + 1):
        pred = model(x)
        loss = torch.sum(y * (torch.log(y + 1e-8) - torch.log(pred + 1e-8)))
        publish_history = step % history_interval == 0 or step == steps
        publish_checkpoint = step % checkpoint_interval == 0 or step == steps
        checkpoint = None
        if publish_history and (not history or int(history[-1]["step"]) != step):
            history.append({"step": step, "kl": float(loss.detach().cpu())})
        if publish_checkpoint:
            checkpoint = checkpoint_path_for(checkpoint_dir, label, step)
            reuse_resumed_checkpoint = bool(
                resumed_from_checkpoint is not None
                and step == start_step
                and checkpoint == last_checkpoint_path
            )
            if not reuse_resumed_checkpoint:
                model_state = cpu_state_dict(model)
                optimizer_state = cpu_tree(optimizer.state_dict())
                atomic_torch_save(
                    {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "checkpoint_chain_version": (
                            CHECKPOINT_CHAIN_VERSION
                        ),
                        "model": label,
                        "step": step,
                        "optimizer_updates_completed": step,
                        "step_semantics": (
                            "State after exactly step optimizer updates; "
                            "step 0 is the initialized model."
                        ),
                        "steps": steps,
                        "seed": seed,
                        "device": str(device),
                        "environment_contract": environment_contract,
                        "environment_fingerprint": environment_sha256,
                        "training_signature": signature,
                        "feature_sha256": feature_sha256,
                        "initialized_state_sha256": (
                            initialized_state_sha256
                        ),
                        "model_state_sha256": state_dict_fingerprint(
                            model_state
                        ),
                        "optimizer_state_sha256": tree_fingerprint(
                            optimizer_state
                        ),
                        "parent_checkpoint_name": (
                            last_checkpoint_path.name
                            if last_checkpoint_path is not None
                            else None
                        ),
                        "parent_checkpoint_step": last_checkpoint_step,
                        "parent_checkpoint_sha256": last_checkpoint_sha256,
                        "state_dict": model_state,
                        "optimizer_state_dict": optimizer_state,
                        "optimizer": OPTIMIZER_SIGNATURE,
                        "history": history,
                    },
                    checkpoint,
                )
                last_checkpoint_path = checkpoint
                last_checkpoint_step = step
                last_checkpoint_sha256 = file_sha256(checkpoint)
        if publish_history or publish_checkpoint:
            atomic_write_json(
                status_path,
                {
                    "generated_at": now(),
                    "status": "training_models",
                    "active_model": label,
                    "completed_models": completed_models,
                    "step": step,
                    "optimizer_updates_completed": step,
                    "total_steps": steps,
                    "progress_percent": step / max(steps, 1) * 100.0,
                    "current_kl": float(loss.detach().cpu()),
                    "device": str(device),
                    "latest_checkpoint": str(checkpoint) if checkpoint else None,
                    "resumed_from_checkpoint": resumed_from_checkpoint,
                    "resumed_from_step": start_step,
                    "training_signature": signature,
                    "history": history,
                },
            )
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = model(x).detach().cpu().numpy()
    model = model.cpu()
    resume_audit = {
        "resumed": resumed_from_checkpoint is not None,
        "checkpoint": resumed_from_checkpoint,
        "resumed_from_step": start_step,
        "target_step": steps,
        "optimizer_updates_completed": steps,
        "device": str(device),
        "optimizer": OPTIMIZER_SIGNATURE,
        "checkpoint_device": resumed_checkpoint_device,
        "checkpoint_optimizer": resumed_checkpoint_optimizer,
        "checkpoint_environment_fingerprint": (
            resumed_checkpoint_environment_fingerprint
        ),
        "environment_fingerprint": environment_sha256,
        "feature_sha256": feature_sha256,
        "initialized_state_sha256": initialized_state_sha256,
        "step_semantics": (
            "Checkpoint step k is the state after exactly k optimizer updates."
        ),
        "training_signature": signature,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_chain_version": CHECKPOINT_CHAIN_VERSION,
        "resume_rejections": resume_rejections,
    }
    return history, predictions, model, resume_audit


def official_relative_shard_path(row: dict[str, Any]) -> str:
    release = str(row["dataset"])
    if (
        not release
        or release in {".", ".."}
        or "/" in release
        or "\\" in release
    ):
        raise ValueError(f"invalid official DROID release name: {release!r}")
    shard_name = Path(str(row["path"])).name
    if not shard_name:
        raise ValueError("DROID shard path has no filename")
    return f"{release}/{shard_name}"


def deterministic_release_stratified_split(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    relative_paths = [official_relative_shard_path(row) for row in rows]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("official DROID relative shard paths must be unique")
    by_release: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_release.setdefault(str(row["dataset"]), []).append(index)

    holdout: list[int] = []
    per_release: list[dict[str, Any]] = []
    for release, indices in sorted(by_release.items()):
        if len(indices) < 2:
            raise ValueError(f"release {release} has fewer than two shards")
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}\0{relative_paths[index]}".encode("utf-8")
            ).digest(),
        )
        holdout_count = min(
            len(indices) - 1,
            max(1, int(round(len(indices) * holdout_fraction))),
        )
        selected = ranked[:holdout_count]
        holdout.extend(selected)
        per_release.append(
            {
                "release": release,
                "total_shards": len(indices),
                "training_shards": len(indices) - holdout_count,
                "holdout_shards": holdout_count,
            }
        )

    holdout_indices = np.asarray(sorted(holdout), dtype=np.int64)
    holdout_set = set(int(value) for value in holdout_indices)
    training_indices = np.asarray(
        [index for index in range(len(rows)) if index not in holdout_set],
        dtype=np.int64,
    )
    holdout_relative_paths = sorted(
        relative_paths[index] for index in holdout_indices
    )
    holdout_relative_path_sha256 = hashlib.sha256(
        "\n".join(holdout_relative_paths).encode("utf-8")
    ).hexdigest()
    contract = {
        "version": "release_stratified_official_relative_path_hash_v2",
        "seed": seed,
        "requested_holdout_fraction": holdout_fraction,
        "training_shards": int(len(training_indices)),
        "holdout_shards": int(len(holdout_indices)),
        "actual_holdout_fraction": float(len(holdout_indices) / len(rows)),
        "per_release": per_release,
        "membership_path_scope": "official_release_relative_path",
        "holdout_membership_locked": True,
        "holdout_relative_paths": holdout_relative_paths,
        "holdout_relative_path_sha256": holdout_relative_path_sha256,
    }
    return training_indices, holdout_indices, contract


def predict_allocation(
    model: base.AllocationHead,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(torch.tensor(features, dtype=torch.float32))
            .detach()
            .cpu()
            .numpy()
        )


def rare_instruction_fingerprint_coverage(
    rows: list[dict[str, Any]],
    training_indices: np.ndarray,
    holdout_indices: np.ndarray,
    source_pred: np.ndarray,
    qtail_pred: np.ndarray,
    *,
    max_train_document_frequency: int = RARE_INSTRUCTION_MAX_TRAIN_DF,
    budgets: tuple[int, ...] = RARE_COVERAGE_BUDGETS,
    thresholds: tuple[float, ...] = RARE_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    """Compute exact held-out fingerprint discovery under repeated shard draws."""
    training_indices = np.asarray(training_indices, dtype=np.int64)
    holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
    source = np.asarray(source_pred, dtype=np.float64)
    qtail = np.asarray(qtail_pred, dtype=np.float64)
    if max_train_document_frequency < 0:
        raise ValueError("max_train_document_frequency must be non-negative")
    if (
        len(training_indices) < 1
        or len(holdout_indices) < 1
        or len(source) != len(holdout_indices)
        or len(qtail) != len(holdout_indices)
    ):
        raise ValueError(
            "coverage inputs require non-empty train/holdout indices and "
            "one prediction per holdout shard"
        )
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(qtail))
        or np.any(source < 0.0)
        or np.any(qtail < 0.0)
        or float(source.sum()) <= 0.0
        or float(qtail.sum()) <= 0.0
    ):
        raise ValueError("coverage predictions must be finite non-negative weights")
    normalized_budgets = tuple(sorted(set(int(value) for value in budgets)))
    normalized_thresholds = tuple(sorted(set(float(value) for value in thresholds)))
    if not normalized_budgets or any(value < 1 for value in normalized_budgets):
        raise ValueError("coverage budgets must be positive integers")
    if not normalized_thresholds or any(
        value <= 0.0 or value >= 1.0 for value in normalized_thresholds
    ):
        raise ValueError("coverage thresholds must be between zero and one")

    source /= source.sum()
    qtail /= qtail.sum()
    training_df: Counter[str] = Counter()
    for index in training_indices:
        training_df.update(
            set(str(value) for value in rows[int(index)].get("instruction_hashes", []))
        )
    training_df_payload = "\n".join(
        f"{fingerprint}\t{count}"
        for fingerprint, count in sorted(training_df.items())
    )
    common_payload = {
        "version": "heldout_instruction_fingerprint_coverage_v1",
        "metric_role": "auxiliary_descriptive_metric_not_a_completion_gate",
        "fingerprint_definition": (
            "SHA-256 of sampled raw instruction bytes emitted by "
            f"{base.FEATURE_EXTRACTOR_VERSION}"
        ),
        "claim_boundary": (
            "This is a byte-level instruction-fingerprint discovery proxy. "
            "It is not semantic task coverage, tail success, or robot-policy success."
        ),
        "rarity_fit_scope": "training_shards_only",
        "evaluation_scope": "holdout_shards_only",
        "sampling_model": (
            "independent_with_replacement_shard_draws_from_each_heldout_"
            "allocation_distribution"
        ),
        "expectation_formula": "mean_fingerprint(1 - (1 - p_fingerprint)^draws)",
        "training_shards": int(len(training_indices)),
        "holdout_shards": int(len(holdout_indices)),
        "max_training_shard_document_frequency": max_train_document_frequency,
        "training_document_frequency_sha256": hashlib.sha256(
            training_df_payload.encode("utf-8")
        ).hexdigest(),
    }

    holdout_membership: dict[str, list[int]] = {}
    for position, index in enumerate(holdout_indices):
        for fingerprint in set(
            str(value)
            for value in rows[int(index)].get("instruction_hashes", [])
        ):
            if training_df.get(fingerprint, 0) <= max_train_document_frequency:
                holdout_membership.setdefault(fingerprint, []).append(position)
    rare_fingerprints = sorted(holdout_membership)
    if not rare_fingerprints:
        return {
            **common_payload,
            "status": "no_eligible_fingerprints",
            "status_reason": (
                "No held-out instruction fingerprint met the configured "
                "training document-frequency threshold."
            ),
            "rare_holdout_fingerprint_count": 0,
            "unseen_in_training_fingerprint_count": 0,
            "curve": [],
            "time_to_coverage": [],
        }

    source_probabilities = np.asarray(
        [
            float(source[holdout_membership[fingerprint]].sum())
            for fingerprint in rare_fingerprints
        ],
        dtype=np.float64,
    )
    qtail_probabilities = np.asarray(
        [
            float(qtail[holdout_membership[fingerprint]].sum())
            for fingerprint in rare_fingerprints
        ],
        dtype=np.float64,
    )

    def expected_coverage(probabilities: np.ndarray, draw_budget: int) -> float:
        return float(
            np.mean(1.0 - np.power(1.0 - probabilities, draw_budget))
        )

    def draws_to_threshold(
        probabilities: np.ndarray,
        threshold: float,
    ) -> int | None:
        if expected_coverage(
            probabilities,
            RARE_COVERAGE_MAX_SEARCH_DRAWS,
        ) < threshold:
            return None
        low = 1
        high = RARE_COVERAGE_MAX_SEARCH_DRAWS
        while low < high:
            middle = (low + high) // 2
            if expected_coverage(probabilities, middle) >= threshold:
                high = middle
            else:
                low = middle + 1
        return low

    curve = []
    for draw_budget in normalized_budgets:
        source_coverage = expected_coverage(source_probabilities, draw_budget)
        qtail_coverage = expected_coverage(qtail_probabilities, draw_budget)
        curve.append(
            {
                "draw_budget": draw_budget,
                "source_expected_coverage": source_coverage,
                "qtail_expected_coverage": qtail_coverage,
                "gain_pp": (qtail_coverage - source_coverage) * 100.0,
            }
        )

    time_to_coverage = []
    for threshold in normalized_thresholds:
        source_draws = draws_to_threshold(source_probabilities, threshold)
        qtail_draws = draws_to_threshold(qtail_probabilities, threshold)
        time_to_coverage.append(
            {
                "coverage_threshold": threshold,
                "source_draws": source_draws,
                "qtail_draws": qtail_draws,
                "qtail_draw_reduction": (
                    source_draws - qtail_draws
                    if source_draws is not None and qtail_draws is not None
                    else None
                ),
            }
        )

    return {
        **common_payload,
        "status": "complete",
        "rare_holdout_fingerprint_count": len(rare_fingerprints),
        "unseen_in_training_fingerprint_count": sum(
            training_df.get(fingerprint, 0) == 0
            for fingerprint in rare_fingerprints
        ),
        "curve": curve,
        "time_to_coverage": time_to_coverage,
    }


def paired_bootstrap(
    source_pred: np.ndarray,
    qtail_pred: np.ndarray,
    tail_mask: np.ndarray,
    strata: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(source_pred)
    strata = np.asarray(strata).astype(str)
    if (
        n < 1
        or len(qtail_pred) != n
        or len(tail_mask) != n
        or len(strata) != n
    ):
        raise ValueError(
            "paired bootstrap arrays must have the same non-zero length"
        )
    strata_indices = {
        label: np.flatnonzero(strata == label)
        for label in sorted(set(strata.tolist()))
    }
    if not strata_indices or any(
        len(indices) == 0 for indices in strata_indices.values()
    ):
        raise ValueError("paired bootstrap strata must be non-empty")
    draws = np.empty(samples, dtype=np.float64)
    for idx in range(samples):
        chosen = np.concatenate(
            [
                indices[
                    rng.integers(0, len(indices), size=len(indices))
                ]
                for indices in strata_indices.values()
            ]
        )
        sampled_source = source_pred[chosen].astype(np.float64)
        sampled_qtail = qtail_pred[chosen].astype(np.float64)
        sampled_tail = tail_mask[chosen]
        sampled_source /= max(float(sampled_source.sum()), 1e-15)
        sampled_qtail /= max(float(sampled_qtail.sum()), 1e-15)
        draws[idx] = float(
            (
                sampled_qtail[sampled_tail].sum()
                - sampled_source[sampled_tail].sum()
            )
            * 100.0
        )
    descriptive_fraction = float(np.mean(draws <= 0.0))
    return {
        "samples": samples,
        "method": BOOTSTRAP_METHOD,
        "inference_role": (
            "conditional_percentile_interval_and_descriptive_fraction_only"
        ),
        "strata": list(strata_indices),
        "strata_counts": {
            label: int(len(indices))
            for label, indices in strata_indices.items()
        },
        "mean_gain_pp": float(np.mean(draws)),
        "ci95_low_pp": float(np.quantile(draws, 0.025)),
        "ci95_high_pp": float(np.quantile(draws, 0.975)),
        "descriptive_fraction_gain_le_zero": descriptive_fraction,
        "p_gain_le_zero": descriptive_fraction,
        "p_gain_le_zero_is_p_value": False,
        "legacy_field_notice": (
            "p_gain_le_zero is retained for artifact compatibility and is "
            "only the descriptive fraction of bootstrap replicates at or "
            "below zero; it is not a hypothesis-test p value."
        ),
    }


def paired_arm_swap_randomization(
    source_pred: np.ndarray,
    qtail_pred: np.ndarray,
    tail_mask: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    source = np.asarray(source_pred, dtype=np.float64)
    qtail = np.asarray(qtail_pred, dtype=np.float64)
    tail = np.asarray(tail_mask, dtype=bool)
    if (
        samples < 1
        or len(source) < 2
        or len(qtail) != len(source)
        or len(tail) != len(source)
        or not np.any(tail)
        or np.all(tail)
    ):
        raise ValueError(
            "randomization requires positive samples, paired arrays, and "
            "non-empty tail/non-tail groups"
        )
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(qtail))
        or np.any(source < 0.0)
        or np.any(qtail < 0.0)
        or float(source.sum()) <= 0.0
        or float(qtail.sum()) <= 0.0
    ):
        raise ValueError(
            "randomization predictions must be finite non-negative weights"
        )
    source = source / source.sum()
    qtail = qtail / qtail.sum()
    observed_gain_pp = float(
        (qtail[tail].sum() - source[tail].sum()) * 100.0
    )
    rng = np.random.default_rng(seed)
    null_gains = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        swap = rng.integers(0, 2, size=len(source), dtype=np.int8).astype(bool)
        randomized_source = np.where(swap, qtail, source)
        randomized_qtail = np.where(swap, source, qtail)
        randomized_source /= randomized_source.sum()
        randomized_qtail /= randomized_qtail.sum()
        null_gains[index] = float(
            (
                randomized_qtail[tail].sum()
                - randomized_source[tail].sum()
            )
            * 100.0
        )
    exceedances = int(
        np.count_nonzero(null_gains >= observed_gain_pp - 1e-12)
    )
    diagnostic_exceedance_fraction = float(
        (exceedances + 1) / (samples + 1)
    )
    return {
        "version": "paired_shard_arm_swap_diagnostic_v2",
        "samples": samples,
        "seed": seed,
        "unit": "non_independent_heldout_shard_weight",
        "method": (
            "independent_within_shard_source_qtail_arm_swap_with_"
            "within_permutation_arm_renormalization"
        ),
        "null_hypothesis": (
            "Source and Q-Tail labels are exchangeable within each fixed "
            "held-out shard pair."
        ),
        "alternative": "qtail_tail_share_gain_is_positive",
        "inference_scope": (
            "conditional_on_fixed_split_fixed_tail_taxonomy_and_fixed_"
            "trained_model_predictions"
        ),
        "exchangeability_justified_by_experiment_design": False,
        "inference_role": (
            "dependency_sensitive_descriptive_diagnostic_only"
        ),
        "observed_gain_pp": observed_gain_pp,
        "null_mean_gain_pp": float(np.mean(null_gains)),
        "null_ci95_low_pp": float(np.quantile(null_gains, 0.025)),
        "null_ci95_high_pp": float(np.quantile(null_gains, 0.975)),
        "exceedances_at_least_observed": exceedances,
        "diagnostic_exceedance_fraction": diagnostic_exceedance_fraction,
        "conditional_p_value": diagnostic_exceedance_fraction,
        "conditional_p_value_is_valid_p_value": False,
        "legacy_field_notice": (
            "conditional_p_value is retained for artifact compatibility. "
            "The allocation weights share one global softmax normalization "
            "and were not independently randomized by shard, so this value "
            "is not a valid hypothesis-test p value and is never a "
            "completion or support gate."
        ),
        "finite_sample_correction": "(k+1)/(B+1)",
    }


def heldout_hypothesis_gate(
    *,
    tail_share_gain_pp: float,
    extreme_underallocation_reduction_pp: float,
    bootstrap: dict[str, float],
    randomization: dict[str, Any],
) -> dict[str, Any]:
    metric_values = {
        "tail_share_gain_pp": float(tail_share_gain_pp),
        "extreme_underallocation_reduction_pp": float(
            extreme_underallocation_reduction_pp
        ),
        "ci95_low_pp": float(bootstrap["ci95_low_pp"]),
        "ci95_high_pp": float(bootstrap["ci95_high_pp"]),
    }
    if not all(np.isfinite(value) for value in metric_values.values()):
        raise ValueError("held-out gate metrics must be finite")
    if metric_values["ci95_low_pp"] > metric_values["ci95_high_pp"]:
        raise ValueError("held-out bootstrap CI bounds are reversed")
    gate: dict[str, Any] = {
        "name": "heldout_tail_allocation_outcome_v4",
        "minimum_tail_share_gain_pp": 2.0,
        "requires_ci95_low_at_least_minimum": True,
        "requires_positive_extreme_underallocation_reduction": True,
        "inference_scope": randomization["inference_scope"],
        "completion_role": "outcome_only_not_experiment_execution_gate",
        "randomization_diagnostic_is_valid_p_value": False,
    }
    supported = bool(
        metric_values["tail_share_gain_pp"]
        >= gate["minimum_tail_share_gain_pp"]
        and metric_values["ci95_low_pp"]
        >= gate["minimum_tail_share_gain_pp"]
        and metric_values["extreme_underallocation_reduction_pp"] > 0.0
    )
    not_supported = bool(
        metric_values["ci95_high_pp"]
        < gate["minimum_tail_share_gain_pp"]
        or metric_values["extreme_underallocation_reduction_pp"] <= 0.0
    )
    gate["outcome"] = (
        "supported"
        if supported
        else "not_supported"
        if not_supported
        else "inconclusive"
    )
    gate["supported"] = supported
    gate["passed"] = supported
    gate["legacy_passed_notice"] = (
        "passed is a compatibility alias for outcome=supported; experiment "
        "completion never depends on this field."
    )
    return gate


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def publish_training_completion_marker(
    marker_dir: Path | None,
    *,
    formal_run: bool,
) -> bool:
    if marker_dir is None or not formal_run:
        return False
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / "DROID_MODEL_TRAINING_COMPLETE").touch()
    return True


def build_intermediate_checkpoint_manifest(
    *,
    out: Path,
    steps: int,
    checkpoint_every_steps: int,
    seed: int,
    device: torch.device,
    environment_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    checkpoint_dir = out / "intermediate_checkpoints"
    manifest_path = out / "droid_intermediate_checkpoint_manifest.json"
    labels = (
        "evaluation_source",
        "evaluation_qtail",
        "deployment_source",
        "deployment_qtail",
    )
    expected_steps = checkpoint_expected_steps(
        steps,
        checkpoint_every_steps,
    )
    expected_paths = {
        checkpoint_path_for(checkpoint_dir, label, step)
        for label in labels
        for step in expected_steps
    }
    observed_paths = set(checkpoint_dir.glob("*.pt"))
    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    for expected_label in labels:
        previous_path: Path | None = None
        previous_step: int | None = None
        previous_sha256: str | None = None
        for expected_step in expected_steps:
            path = checkpoint_path_for(
                checkpoint_dir,
                expected_label,
                expected_step,
            )
            if not path.is_file():
                errors.append(f"missing checkpoint: {path}")
                previous_path = None
                previous_step = None
                previous_sha256 = None
                continue
            try:
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(payload, dict):
                    errors.append(f"checkpoint payload is not a dict: {path}")
                    previous_path = None
                    previous_step = None
                    previous_sha256 = None
                    continue
                content_errors = checkpoint_content_errors(payload)
                expected_parent_name = (
                    previous_path.name if previous_path else None
                )
                metadata_valid = (
                    payload.get("model") == expected_label
                    and int(payload.get("step", -1)) == expected_step
                    and int(
                        payload.get("optimizer_updates_completed", -1)
                    )
                    == expected_step
                    and int(payload.get("steps", -1)) == steps
                    and int(payload.get("seed", -1)) == seed
                    and payload.get("device") == str(device)
                    and payload.get("optimizer") == OPTIMIZER_SIGNATURE
                    and payload.get("environment_fingerprint")
                    == environment_sha256
                    and len(
                        str(payload.get("training_signature", ""))
                    )
                    == 64
                    and len(str(payload.get("feature_sha256", ""))) == 64
                    and len(
                        str(payload.get("initialized_state_sha256", ""))
                    )
                    == 64
                    and payload.get("parent_checkpoint_name")
                    == expected_parent_name
                    and payload.get("parent_checkpoint_step")
                    == previous_step
                    and payload.get("parent_checkpoint_sha256")
                    == previous_sha256
                    and (
                        expected_step != 0
                        or payload.get("model_state_sha256")
                        == payload.get("initialized_state_sha256")
                    )
                )
                if content_errors or not metadata_valid:
                    detail = ",".join(content_errors) or (
                        "metadata_or_parent"
                    )
                    errors.append(
                        f"checkpoint contract mismatch: {path}: {detail}"
                    )
                    previous_path = None
                    previous_step = None
                    previous_sha256 = None
                    continue
                entry = {
                    **artifact_entry(path),
                    "checkpoint_format_version": int(
                        payload["format_version"]
                    ),
                    "checkpoint_chain_version": str(
                        payload["checkpoint_chain_version"]
                    ),
                    "model_stage": expected_label,
                    "step": expected_step,
                    "optimizer_updates_completed": int(
                        payload["optimizer_updates_completed"]
                    ),
                    "target_steps": int(payload["steps"]),
                    "seed": int(payload["seed"]),
                    "device": str(payload["device"]),
                    "optimizer": str(payload["optimizer"]),
                    "environment_fingerprint": str(
                        payload["environment_fingerprint"]
                    ),
                    "training_signature": str(
                        payload["training_signature"]
                    ),
                    "feature_sha256": str(payload["feature_sha256"]),
                    "initialized_state_sha256": str(
                        payload["initialized_state_sha256"]
                    ),
                    "model_state_sha256": str(
                        payload["model_state_sha256"]
                    ),
                    "optimizer_state_sha256": str(
                        payload["optimizer_state_sha256"]
                    ),
                    "parent_checkpoint_name": (
                        payload["parent_checkpoint_name"]
                    ),
                    "parent_checkpoint_step": (
                        payload["parent_checkpoint_step"]
                    ),
                    "parent_checkpoint_sha256": (
                        payload["parent_checkpoint_sha256"]
                    ),
                    "history_points": len(payload.get("history", [])),
                }
                entries.append(entry)
                previous_path = path
                previous_step = expected_step
                previous_sha256 = entry["sha256"]
            except (
                EOFError,
                OSError,
                pickle.UnpicklingError,
                RuntimeError,
                KeyError,
                ValueError,
                TypeError,
            ) as error:
                errors.append(f"checkpoint unreadable: {path}: {error}")
                previous_path = None
                previous_step = None
                previous_sha256 = None
    for path in sorted(observed_paths - expected_paths):
        errors.append(f"unexpected checkpoint: {path}")
    feature_signatures = {
        label: {
            str(entry["feature_sha256"])
            for entry in entries
            if entry["model_stage"] == label
        }
        for label in labels
    }
    initialization_signatures = {
        label: {
            str(entry["initialized_state_sha256"])
            for entry in entries
            if entry["model_stage"] == label
        }
        for label in labels
    }
    paired_feature_signatures_equal = bool(
        len(feature_signatures["evaluation_source"]) == 1
        and feature_signatures["evaluation_source"]
        == feature_signatures["evaluation_qtail"]
        and len(feature_signatures["deployment_source"]) == 1
        and feature_signatures["deployment_source"]
        == feature_signatures["deployment_qtail"]
    )
    initialized_state_signatures_equal = bool(
        all(len(values) == 1 for values in initialization_signatures.values())
        and len(
            {
                next(iter(values))
                for values in initialization_signatures.values()
            }
        )
        == 1
    )
    if not paired_feature_signatures_equal:
        errors.append("paired Source/Q-Tail feature fingerprints differ")
    if not initialized_state_signatures_equal:
        errors.append("Source/Q-Tail initialized state fingerprints differ")
    payload = {
        "generated_at": now(),
        "status": "complete" if not errors else "failed",
        "checkpoint_directory": str(checkpoint_dir),
        "contract": {
            "model_stages": list(labels),
            "steps_per_stage": steps,
            "checkpoint_every_steps": checkpoint_every_steps,
            "expected_steps": expected_steps,
            "expected_checkpoint_count": (
                len(labels) * len(expected_steps)
            ),
            "step_semantics": OPTIMIZER_UPDATE_SEMANTICS,
            "device": str(device),
            "optimizer": OPTIMIZER_SIGNATURE,
            "seed": seed,
            "environment_fingerprint": environment_sha256,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_chain_version": CHECKPOINT_CHAIN_VERSION,
            "checkpoint_content_hashes_recomputed": True,
            "parent_checkpoint_hash_chains_verified": not errors,
            "paired_feature_signatures_equal": (
                paired_feature_signatures_equal
            ),
            "initialized_state_signatures_equal": (
                initialized_state_signatures_equal
            ),
        },
        "actual_checkpoint_count": len(entries),
        "entries": entries,
        "errors": errors,
    }
    atomic_write_json(manifest_path, payload)
    if errors:
        raise ValueError(
            "intermediate checkpoint manifest failed: "
            + "; ".join(errors[:10])
        )
    return manifest_path, payload


def write_feature_cache_manifest(
    *,
    out: Path,
    status: str,
    shard_count: int,
    represented_bytes: int,
    cache_files: list[Path],
    source_snapshot_at: str,
    source_shard_paths: list[str],
) -> Path:
    cache_dir = out / "feature_cache"
    cache_files = sorted(set(cache_files))
    selected = {path.resolve() for path in cache_files}
    directory_cache_files = sorted(cache_dir.glob("*.json"))
    unreferenced_cache_files = [
        path
        for path in directory_cache_files
        if path.resolve() not in selected
    ]
    unreferenced_names = "\n".join(
        path.name for path in unreferenced_cache_files
    )
    manifest_path = out / "droid_feature_cache_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "generated_at": now(),
            "source_snapshot_at": source_snapshot_at,
            "source_shard_count": len(source_shard_paths),
            "source_shard_paths_sha256": hashlib.sha256(
                "\n".join(sorted(source_shard_paths)).encode("utf-8")
            ).hexdigest(),
            "status": status,
            "cache_dir": str(cache_dir),
            "cache_count": len(cache_files),
            "cache_directory_count": len(directory_cache_files),
            "expected_shard_count": shard_count,
            "represented_bytes": represented_bytes,
            "all_expected_caches_present": len(cache_files) == shard_count,
            "unreferenced_cache_count": len(unreferenced_cache_files),
            "unreferenced_cache_bytes": sum(
                path.stat().st_size for path in unreferenced_cache_files
            ),
            "unreferenced_cache_name_sha256": hashlib.sha256(
                unreferenced_names.encode("utf-8")
            ).hexdigest(),
            "selection_contract": (
                "Only artifacts listed below are training inputs; "
                "unreferenced cache files are excluded."
            ),
            "artifacts": [artifact_entry(path) for path in cache_files],
        },
    )
    return manifest_path


def summarize_release_composition(
    rows: list[dict[str, Any]],
    data_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    summaries = []
    for release, release_rows in sorted(grouped.items()):
        info_path = data_dir / release / "dataset_info.json"
        dataset_name = None
        official_shards = 0
        official_records = 0
        official_split_bytes = 0
        metadata_status = "missing"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            dataset_name = str(info.get("name", ""))
            splits = info.get("splits", [])
            official_shards = sum(
                len(split.get("shardLengths", []))
                for split in splits
            )
            official_records = sum(
                sum(int(value) for value in split.get("shardLengths", []))
                for split in splits
            )
            official_split_bytes = sum(
                int(split.get("numBytes", 0))
                for split in splits
            )
            metadata_status = "verified"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata_status = "invalid_or_missing"

        observed_shards = len(release_rows)
        observed_records = sum(
            int(row["records_decoded"])
            for row in release_rows
        )
        full_shard_coverage = (
            metadata_status == "verified"
            and observed_shards == official_shards
        )
        summaries.append(
            {
                "release": release,
                "official_dataset_name": dataset_name,
                "dataset_info": str(info_path),
                "metadata_status": metadata_status,
                "observed_tfrecord_shards": observed_shards,
                "official_tfrecord_shards": official_shards,
                "observed_records_decoded": observed_records,
                "official_expected_records": official_records,
                "observed_tfrecord_bytes": sum(
                    int(row["bytes"])
                    for row in release_rows
                ),
                "official_split_bytes": official_split_bytes,
                "full_shard_coverage": full_shard_coverage,
                "full_record_count_match": (
                    full_shard_coverage
                    and observed_records == official_records
                ),
            }
        )
    return summaries


def verified_mirror_audit(
    *,
    data_dir: Path,
    shards: list[Path],
    object_manifest_path: Path | None,
    checksum_manifest_path: Path | None,
    checksum_ledger_path: Path | None,
    transport_status_path: Path | None,
    download_marker_path: Path | None,
    download_verification_path: Path | None,
    md5_rehash_status_path: Path | None,
    required: bool,
) -> dict[str, Any]:
    required_paths = {
        "object_manifest": object_manifest_path,
        "checksum_manifest": checksum_manifest_path,
        "checksum_ledger": checksum_ledger_path,
        "transport_status": transport_status_path,
        "download_marker": download_marker_path,
        "download_verification": download_verification_path,
    }
    missing_inputs = sorted(
        name
        for name, path in required_paths.items()
        if path is None or not path.is_file()
    )
    if missing_inputs:
        if required:
            raise SystemExit(
                "Formal full run requires current mirror-binding inputs: "
                + ", ".join(missing_inputs)
            )
        return {
            "required": False,
            "verified": False,
            "missing_inputs": missing_inputs,
        }

    manifest = json.loads(object_manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(
        download_verification_path.read_text(encoding="utf-8")
    )
    download_marker = json.loads(
        download_marker_path.read_text(encoding="utf-8")
    )
    binding, binding_checks, binding_file_errors = build_binding(
        data_dir=data_dir,
        manifest_path=object_manifest_path,
        checksum_manifest_path=checksum_manifest_path,
        checksum_ledger_path=checksum_ledger_path,
        transport_status_path=transport_status_path,
        expected_bytes=FORMAL_EXPECTED_BYTES,
        expected_objects=FORMAL_EXPECTED_OBJECTS,
        expected_tfrecords=FORMAL_EXPECTED_TFRECORDS,
    )
    local_byte_md5_audit = (
        official_md5_byte_audit(
            data_dir=data_dir,
            checksum_manifest_path=checksum_manifest_path,
            status_path=md5_rehash_status_path,
        )
        if required
        else {
            "verified": False,
            "method": "not_required_for_bounded_run",
        }
    )
    expected_tfrecords = {
        str(item["relative_path"])
        for item in manifest.get("objects", [])
        if "tfrecord" in str(item.get("relative_path", "")).lower()
    }
    actual_tfrecords = {
        str(path.relative_to(data_dir))
        for path in shards
    }
    missing_tfrecords = sorted(expected_tfrecords - actual_tfrecords)
    extra_tfrecords = sorted(actual_tfrecords - expected_tfrecords)
    expected_release_counts = Counter(
        Path(relative).parts[0]
        for relative in expected_tfrecords
    )
    actual_release_counts = Counter(
        Path(relative).parts[0]
        for relative in actual_tfrecords
    )
    exact_release_counts = {
        release: int(expected["tfrecord_shards"])
        for release, expected in FORMAL_RELEASE_CONTRACT.items()
    }
    marker_matches_current_binding = bool(
        download_marker.get("marker_version") == DOWNLOAD_MARKER_VERSION
        and download_marker.get("status") == "complete"
        and download_marker.get("immutable") is True
        and download_marker.get("binding") == binding
    )
    verified = (
        manifest.get("status") in {"verified", "complete"}
        and int(manifest.get("object_count", -1))
        == len(manifest.get("objects", []))
        == FORMAL_EXPECTED_OBJECTS
        and int(manifest.get("total_bytes", -1))
        == FORMAL_EXPECTED_BYTES
        and verification.get("status") == "complete"
        and verification.get("ready_for_full_allocation_training")
        is True
        and int(verification.get("checksum_rsync_returncode", -1)) == 0
        and int(verification.get("manifest_object_count", -1))
        == FORMAL_EXPECTED_OBJECTS
        and int(verification.get("local_official_bytes", -1))
        == FORMAL_EXPECTED_BYTES
        and verification.get("manifest_sha256")
        == file_sha256(object_manifest_path)
        and verification.get("checksum_manifest_sha256")
        == file_sha256(checksum_manifest_path)
        and verification.get("checksum_ledger_sha256")
        == file_sha256(checksum_ledger_path)
        and len(expected_tfrecords) == FORMAL_EXPECTED_TFRECORDS
        and dict(expected_release_counts) == exact_release_counts
        and dict(actual_release_counts) == exact_release_counts
        and not missing_tfrecords
        and not extra_tfrecords
        and all(binding_checks.values())
        and not binding_file_errors
        and marker_matches_current_binding
        and (not required or local_byte_md5_audit["verified"] is True)
    )
    audit = {
        "required": required,
        "verified": verified,
        "object_manifest": artifact_entry(object_manifest_path),
        "download_verification": artifact_entry(download_verification_path),
        "manifest_object_count": int(manifest.get("object_count", -1)),
        "manifest_total_bytes": int(manifest.get("total_bytes", -1)),
        "formal_expected_object_count": FORMAL_EXPECTED_OBJECTS,
        "formal_expected_total_bytes": FORMAL_EXPECTED_BYTES,
        "expected_tfrecord_shards": len(expected_tfrecords),
        "actual_tfrecord_shards": len(actual_tfrecords),
        "formal_expected_tfrecord_shards": FORMAL_EXPECTED_TFRECORDS,
        "expected_release_tfrecord_counts": dict(
            expected_release_counts
        ),
        "actual_release_tfrecord_counts": dict(actual_release_counts),
        "formal_release_tfrecord_counts": exact_release_counts,
        "missing_tfrecord_count": len(missing_tfrecords),
        "missing_tfrecord_sample": missing_tfrecords[:20],
        "extra_tfrecord_count": len(extra_tfrecords),
        "extra_tfrecord_sample": extra_tfrecords[:20],
        "download_marker": artifact_entry(download_marker_path),
        "download_marker_matches_current_binding": (
            marker_matches_current_binding
        ),
        "current_binding": binding,
        "current_binding_checks": binding_checks,
        "current_binding_file_error_count": len(binding_file_errors),
        "current_binding_file_error_sample": binding_file_errors[:20],
        "local_official_md5_byte_audit": local_byte_md5_audit,
    }
    if required and not verified:
        raise SystemExit(
            "Verified-mirror input gate failed: "
            f"missing_tfrecords={len(missing_tfrecords)} "
            f"extra_tfrecords={len(extra_tfrecords)} "
            f"binding_errors={len(binding_file_errors)} "
            f"local_md5_rehash={local_byte_md5_audit.get('verified')} "
            f"marker_matches={marker_matches_current_binding} "
            f"verification_status={verification.get('status')}"
        )
    return audit


def formal_protocol_mismatches(
    *,
    require_verified_mirror: bool,
    seed: int,
    steps: int,
    checkpoint_every_steps: int,
    bootstrap_samples: int,
    holdout_fraction: float,
    min_record_parse_rate: float,
    min_record_scan_complete_rate: float,
    pt_source_sha256: str,
) -> list[str]:
    mismatches = []
    if not require_verified_mirror:
        mismatches.append("require_verified_mirror=false")
    if seed != FORMAL_SEED:
        mismatches.append(f"seed={seed}")
    if steps != FORMAL_STEPS_PER_STAGE:
        mismatches.append(f"steps={steps}")
    if checkpoint_every_steps != FORMAL_CHECKPOINT_EVERY_STEPS:
        mismatches.append(
            f"checkpoint_every_steps={checkpoint_every_steps}"
        )
    if bootstrap_samples != FORMAL_BOOTSTRAP_SAMPLES:
        mismatches.append(f"bootstrap_samples={bootstrap_samples}")
    if holdout_fraction != FORMAL_HOLDOUT_FRACTION:
        mismatches.append(f"holdout_fraction={holdout_fraction}")
    if min_record_parse_rate != FORMAL_MIN_RECORD_PARSE_RATE:
        mismatches.append(f"min_record_parse_rate={min_record_parse_rate}")
    if (
        min_record_scan_complete_rate
        != FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE
    ):
        mismatches.append(
            "min_record_scan_complete_rate="
            f"{min_record_scan_complete_rate}"
        )
    if pt_source_sha256 != FORMAL_PT_SOURCE_SHA256:
        mismatches.append("pt_source_sha256")
    return mismatches


def formal_record_closure_errors(
    release_composition: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not release_composition:
        return ["release_composition_empty"]
    by_release = {
        str(row.get("release", "unknown")): row
        for row in release_composition
    }
    expected_names = set(FORMAL_RELEASE_CONTRACT)
    observed_names = set(by_release)
    for missing in sorted(expected_names - observed_names):
        errors.append(f"{missing}:release_missing")
    for extra in sorted(observed_names - expected_names):
        errors.append(f"{extra}:unexpected_release")
    total_shards = 0
    total_records = 0
    for name, expected in FORMAL_RELEASE_CONTRACT.items():
        release = by_release.get(name)
        if release is None:
            continue
        observed_shards = int(
            release.get("observed_tfrecord_shards", -1)
        )
        official_shards = int(
            release.get("official_tfrecord_shards", -1)
        )
        observed_records = int(
            release.get("observed_records_decoded", -1)
        )
        official_records = int(
            release.get("official_expected_records", -1)
        )
        total_shards += max(0, observed_shards)
        total_records += max(0, observed_records)
        if release.get("metadata_status") != "verified":
            errors.append(f"{name}:metadata_not_verified")
        if official_shards != int(expected["tfrecord_shards"]):
            errors.append(
                f"{name}:official_shards={official_shards}"
            )
        if observed_shards != int(expected["tfrecord_shards"]):
            errors.append(
                f"{name}:observed_shards={observed_shards}"
            )
        if official_records != int(expected["records"]):
            errors.append(
                f"{name}:official_records={official_records}"
            )
        if observed_records != int(expected["records"]):
            errors.append(
                f"{name}:observed_records={observed_records}"
            )
        if release.get("full_shard_coverage") is not True:
            errors.append(f"{name}:full_shard_coverage_false")
        if release.get("full_record_count_match") is not True:
            errors.append(f"{name}:full_record_count_match_false")
    if total_shards != FORMAL_EXPECTED_TFRECORDS:
        errors.append(f"total_shards={total_shards}")
    if total_records != FORMAL_EXPECTED_RECORDS:
        errors.append(f"total_records={total_records}")
    return errors


def load_bounded_shard_list(
    *,
    data_dir: Path,
    shard_list_path: Path,
) -> list[Path]:
    payload = json.loads(shard_list_path.read_text(encoding="utf-8"))
    relative_paths = payload.get("relative_paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise SystemExit(
            "--shard-list must contain a non-empty relative_paths array."
        )
    if any(not isinstance(item, str) for item in relative_paths):
        raise SystemExit("--shard-list paths must all be strings.")
    if len(set(relative_paths)) != len(relative_paths):
        raise SystemExit("--shard-list contains duplicate paths.")
    if relative_paths != sorted(relative_paths):
        raise SystemExit("--shard-list relative_paths must be sorted.")
    data_root = data_dir.resolve()
    shards: list[Path] = []
    for relative in relative_paths:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SystemExit(
                f"--shard-list contains an unsafe path: {relative}"
            )
        candidate = data_dir / relative_path
        if (
            not candidate.is_file()
            or "tfrecord" not in candidate.name.lower()
            or base.is_partial(candidate)
        ):
            raise SystemExit(
                f"--shard-list path is not a complete TFRecord: {relative}"
            )
        if not candidate.resolve().is_relative_to(data_root):
            raise SystemExit(
                f"--shard-list path escapes data-dir: {relative}"
            )
        shards.append(candidate)
    observed_digest = hashlib.sha256(
        "\n".join(relative_paths).encode("utf-8")
    ).hexdigest()
    expected_digest = payload.get("relative_paths_sha256")
    if expected_digest and expected_digest != observed_digest:
        raise SystemExit("--shard-list membership digest mismatch.")
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--marker-dir", type=Path)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=0,
        help="0 streams every record; positive values are test-only caps.",
    )
    parser.add_argument("--min-shards", type=int, default=64)
    parser.add_argument("--min-record-parse-rate", type=float, default=0.95)
    parser.add_argument("--min-record-scan-complete-rate", type=float, default=0.95)
    parser.add_argument("--status-every-shards", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5_000)
    parser.add_argument("--max-shards", type=int, default=0, help="Test-only cap; 0 means every complete shard.")
    parser.add_argument(
        "--shard-list",
        type=Path,
        help=(
            "Test-only frozen relative-path JSON. It is mutually exclusive "
            "with --max-shards and can never enter the formal protocol."
        ),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
        help="Release-stratified deterministic shard holdout fraction.",
    )
    parser.add_argument(
        "--pt-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "uploaded_data.csv",
        help="Audited empirical PT probability CSV used for Q-Tail rank allocation.",
    )
    parser.add_argument("--object-manifest", type=Path)
    parser.add_argument("--checksum-manifest", type=Path)
    parser.add_argument("--checksum-ledger", type=Path)
    parser.add_argument("--transport-status", type=Path)
    parser.add_argument("--download-marker", type=Path)
    parser.add_argument("--download-verification", type=Path)
    parser.add_argument("--require-verified-mirror", action="store_true")
    parser.add_argument("--required-mount", type=Path)
    parser.add_argument(
        "--process-lock",
        type=Path,
        help=(
            "Exclusive writer lock. Defaults to "
            "<out>/.qtail_train_droid_full.lock."
        ),
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Precompute audited per-shard feature caches without training or "
            "writing completion markers. Formal training still requires the "
            "verified full mirror."
        ),
    )
    args = parser.parse_args()
    require_mount(args.required_mount)

    if args.records_per_shard < 0:
        raise SystemExit("--records-per-shard must be >= 0")
    if args.shard_list and args.max_shards:
        raise SystemExit("--shard-list and --max-shards are mutually exclusive.")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise SystemExit("--holdout-fraction must be between 0 and 0.5")
    if not args.max_shards and args.records_per_shard > 0:
        print(
            "Full-run safety override: --max-shards=0 requires all-record mode; "
            f"ignoring --records-per-shard={args.records_per_shard}.",
            flush=True,
        )
        args.records_per_shard = 0
    args.out.mkdir(parents=True, exist_ok=True)
    process_lock_path = (
        args.process_lock
        if args.process_lock is not None
        else args.out / ".qtail_train_droid_full.lock"
    )
    process_lock_handle = acquire_process_lock(process_lock_path)
    if args.marker_dir:
        args.marker_dir.mkdir(parents=True, exist_ok=True)

    source_snapshot_at = now()
    shards = (
        load_bounded_shard_list(
            data_dir=args.data_dir,
            shard_list_path=args.shard_list,
        )
        if args.shard_list
        else base.find_shards(
            args.data_dir,
            max_shards=args.max_shards,
        )
    )
    source_shard_paths = [
        str(path.relative_to(args.data_dir)) for path in shards
    ]
    if len(shards) < args.min_shards:
        raise SystemExit(f"Only {len(shards)} complete TFRecord shards found; require at least {args.min_shards}.")
    formal_training_requested = bool(
        not args.features_only
        and args.max_shards == 0
        and args.shard_list is None
        and args.records_per_shard == 0
    )
    if formal_training_requested:
        formal_errors = formal_protocol_mismatches(
            require_verified_mirror=args.require_verified_mirror,
            seed=args.seed,
            steps=args.steps,
            checkpoint_every_steps=args.checkpoint_every_steps,
            bootstrap_samples=args.bootstrap_samples,
            holdout_fraction=args.holdout_fraction,
            min_record_parse_rate=args.min_record_parse_rate,
            min_record_scan_complete_rate=(
                args.min_record_scan_complete_rate
            ),
            pt_source_sha256=file_sha256(args.pt_source),
        )
        if formal_errors:
            raise SystemExit(
                "Formal DROID protocol is immutable; mismatches: "
                + ", ".join(formal_errors)
            )
    formal_run = formal_training_requested
    input_audit = verified_mirror_audit(
        data_dir=args.data_dir,
        shards=shards,
        object_manifest_path=args.object_manifest,
        checksum_manifest_path=args.checksum_manifest,
        checksum_ledger_path=args.checksum_ledger,
        transport_status_path=args.transport_status,
        download_marker_path=args.download_marker,
        download_verification_path=args.download_verification,
        md5_rehash_status_path=(
            args.out / "droid_local_md5_rehash_status.json"
        ),
        required=args.require_verified_mirror,
    )
    if formal_run:
        if args.marker_dir:
            for marker_name in (
                "FINAL_PAGE_QA_PREVIEW",
                "FINAL_PAGE_QA_COMPLETE",
                "DROID_TRAINING_COMPLETE",
            ):
                (args.marker_dir / marker_name).unlink(missing_ok=True)

    pt_source_snapshot = args.out / "empirical_pt_source.csv"
    pt_source_temporary = temporary_path(pt_source_snapshot)
    try:
        shutil.copyfile(args.pt_source, pt_source_temporary)
        pt_source_temporary.replace(pt_source_snapshot)
    finally:
        pt_source_temporary.unlink(missing_ok=True)
    pt_values, pt_source_audit = base.load_pt_probabilities(pt_source_snapshot)
    pt_source_audit["original_path"] = str(args.pt_source)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime_environment = runtime_environment_contract(device)
    runtime_environment_sha256 = environment_fingerprint(
        runtime_environment
    )
    architecture = f"AllocationHead({len(base.FEATURE_NAMES)}→32→16→1)"
    run_manifest_path = args.out / (
        "droid_feature_prewarm_run.json"
        if args.features_only
        else "droid_full_run_manifest.json"
    )
    atomic_write_json(
        run_manifest_path,
        {
            "generated_at": now(),
            "source_snapshot_at": source_snapshot_at,
            "source_shard_count": len(source_shard_paths),
            "source_shard_paths_sha256": hashlib.sha256(
                "\n".join(sorted(source_shard_paths)).encode("utf-8")
            ).hexdigest(),
            "bounded_shard_list": (
                str(args.shard_list) if args.shard_list else None
            ),
            "status": (
                "prewarming_complete_shards"
                if args.features_only
                else "running"
            ),
            "dataset": "DROID full RLDS mirror",
            "data_dir": str(args.data_dir),
            "out": str(args.out),
            "full_record_mode": args.records_per_shard == 0,
            "records_per_shard": args.records_per_shard,
            "shard_count": len(shards),
            "steps_per_arm": args.steps,
            "training_stages_per_arm": 2,
            "total_steps_per_arm": args.steps * 2,
            "seed": args.seed,
            "device": str(device),
            "runtime_environment": runtime_environment,
            "runtime_environment_fingerprint": (
                runtime_environment_sha256
            ),
            "input_audit": input_audit,
            "pt_source_audit": pt_source_audit,
            "holdout_contract": {
                "method": (
                    "release_stratified_official_relative_path_hash_v2"
                ),
                "fraction": args.holdout_fraction,
                "seed": args.seed,
                "normalization_fit": "training_shards_only",
                "tail_taxonomy_fit": "training_shards_only",
                "instruction_rarity_fit": "training_shards_only",
                "pt_allocation_fit": "training_shards_only",
                "deployment_refit": "all_shards",
            },
            "same_compute_contract": {
                "architecture": architecture,
                "optimizer": OPTIMIZER_SIGNATURE,
                "evaluation_source_steps": args.steps,
                "evaluation_qtail_steps": args.steps,
                "deployment_source_steps": args.steps,
                "deployment_qtail_steps": args.steps,
                "source_total_steps": args.steps * 2,
                "qtail_total_steps": args.steps * 2,
                "evaluation_source_optimizer_updates": args.steps,
                "evaluation_qtail_optimizer_updates": args.steps,
                "deployment_source_optimizer_updates": args.steps,
                "deployment_qtail_optimizer_updates": args.steps,
                "source_total_optimizer_updates": args.steps * 2,
                "qtail_total_optimizer_updates": args.steps * 2,
                "optimizer_update_semantics": OPTIMIZER_UPDATE_SEMANTICS,
                "same_seed": True,
                "same_features": True,
                "same_device": True,
            },
        },
    )

    rows, selected_cache_paths = build_rows_resumable(
        data_dir=args.data_dir,
        shards=shards,
        out=args.out,
        records_per_shard=args.records_per_shard,
        status_every_shards=args.status_every_shards,
        status_label=(
            "prewarming_complete_shards"
            if args.features_only
            else "extracting_all_records"
        ),
        required_mount=args.required_mount,
    )
    parsed_shards = sum(int(row["record_parse_ok"]) for row in rows)
    parse_rate = parsed_shards / max(len(rows), 1)
    scan_complete_shards = sum(int(row.get("record_scan_complete", 0)) for row in rows)
    scan_complete_rate = scan_complete_shards / max(len(rows), 1)
    release_composition = summarize_release_composition(rows, args.data_dir)
    if formal_run:
        record_closure_errors = formal_record_closure_errors(
            release_composition
        )
        if record_closure_errors:
            raise SystemExit(
                "Formal DROID record closure failed: "
                + ", ".join(record_closure_errors)
            )
    if args.features_only:
        represented_bytes = sum(int(row["bytes"]) for row in rows)
        cache_manifest_path = write_feature_cache_manifest(
            out=args.out,
            status="prewarm_complete_shards",
            shard_count=len(rows),
            represented_bytes=represented_bytes,
            cache_files=selected_cache_paths,
            source_snapshot_at=source_snapshot_at,
            source_shard_paths=source_shard_paths,
        )
        prewarm_status = {
            "generated_at": now(),
            "status": (
                "prewarm_complete"
                if (
                    parse_rate >= args.min_record_parse_rate
                    and (
                        args.records_per_shard != 0
                        or scan_complete_rate >= args.min_record_scan_complete_rate
                    )
                )
                else "prewarm_complete_with_coverage_errors"
            ),
            "formal_training_started": False,
            "completion_markers_written": False,
            "full_record_mode": args.records_per_shard == 0,
            "shard_count": len(rows),
            "represented_bytes": represented_bytes,
            "records_decoded": sum(int(row["records_decoded"]) for row in rows),
            "parsed_shards": parsed_shards,
            "parse_rate": parse_rate,
            "record_scan_complete_shards": scan_complete_shards,
            "record_scan_complete_rate": scan_complete_rate,
            "release_composition": release_composition,
            "parse_errors": [
                {
                    "path": str(row["path"]),
                    "error": str(row.get("record_parse_error", "")),
                }
                for row in rows
                if row.get("record_parse_error")
            ],
            "feature_cache_dir": str(args.out / "feature_cache"),
            "feature_cache_manifest": str(cache_manifest_path),
            "formal_gate": (
                "Cache reuse is allowed only after the full mirror checksum "
                "gate and verified input audit pass."
            ),
        }
        atomic_write_json(
            args.out / "droid_feature_prewarm_status.json",
            prewarm_status,
        )
        atomic_write_json(
            run_manifest_path,
            {
                **json.loads(run_manifest_path.read_text(encoding="utf-8")),
                "status": prewarm_status["status"],
                "completed_at": prewarm_status["generated_at"],
                "prewarm_status": str(
                    args.out / "droid_feature_prewarm_status.json"
                ),
            },
        )
        print(json.dumps(prewarm_status, ensure_ascii=False), flush=True)
        return
    if parse_rate < args.min_record_parse_rate:
        raise SystemExit(
            f"Record parse coverage {parse_rate:.4f} is below {args.min_record_parse_rate:.4f} "
            f"({parsed_shards}/{len(rows)} shards)."
        )
    if args.records_per_shard == 0 and scan_complete_rate < args.min_record_scan_complete_rate:
        raise SystemExit(
            f"Full-record scan completion {scan_complete_rate:.4f} is below "
            f"{args.min_record_scan_complete_rate:.4f} ({scan_complete_shards}/{len(rows)} shards)."
        )
    cache_manifest_path = write_feature_cache_manifest(
        out=args.out,
        status="verified_full_input",
        shard_count=len(rows),
        represented_bytes=sum(int(row["bytes"]) for row in rows),
        cache_files=selected_cache_paths,
        source_snapshot_at=source_snapshot_at,
        source_shard_paths=source_shard_paths,
    )
    if args.marker_dir:
        (args.marker_dir / "DROID_FEATURE_EXTRACTION_COMPLETE").touch()

    training_indices, holdout_indices, holdout_contract = (
        deterministic_release_stratified_split(
            rows,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
    )
    if formal_run:
        holdout_by_release = {
            str(item["release"]): int(item["holdout_shards"])
            for item in holdout_contract["per_release"]
        }
        if (
            set(holdout_by_release) != set(BOOTSTRAP_STRATA)
            or any(
                count != FORMAL_HOLDOUT_SHARDS_PER_RELEASE
                for count in holdout_by_release.values()
            )
            or holdout_contract["holdout_relative_path_sha256"]
            != FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
            or len(holdout_contract["holdout_relative_paths"])
            != FORMAL_HOLDOUT_SHARDS_PER_RELEASE * len(BOOTSTRAP_STRATA)
        ):
            raise SystemExit(
                "Formal DROID holdout membership does not match the locked "
                "official relative-path list/digest"
            )
    (
        evaluation_features,
        evaluation_source,
        evaluation_qtail,
        evaluation_tail_scores,
        datasets,
        evaluation_normalization,
    ) = (
        base.make_training_matrix(
            rows,
            pt_values=pt_values,
            normalization_fit_indices=training_indices,
            allocation_fit_indices=training_indices,
        )
    )
    (
        deployment_features,
        deployment_source,
        deployment_qtail,
        deployment_tail_scores,
        deployment_datasets,
        deployment_normalization,
    ) = base.make_training_matrix(
        rows,
        pt_values=pt_values,
    )
    if deployment_datasets != datasets:
        raise RuntimeError("Evaluation and deployment dataset contracts differ")
    evaluation_source_target = evaluation_source[training_indices].copy()
    evaluation_source_target /= evaluation_source_target.sum()
    evaluation_qtail_target = evaluation_qtail[training_indices].copy()
    evaluation_qtail_target /= evaluation_qtail_target.sum()
    require_mount(args.required_mount)
    if args.marker_dir:
        (args.marker_dir / "DROID_MODEL_TRAINING_STARTED").touch()
    evaluation_source_hist, _, evaluation_source_model, evaluation_source_resume = train_once_audited(
        features=evaluation_features[training_indices],
        target=evaluation_source_target,
        steps=args.steps,
        seed=args.seed,
        label="evaluation_source",
        device=device,
        out=args.out,
        checkpoint_every_steps=args.checkpoint_every_steps,
        completed_models=[],
        environment_contract=runtime_environment,
    )
    require_mount(args.required_mount)
    evaluation_qtail_hist, _, evaluation_qtail_model, evaluation_qtail_resume = train_once_audited(
        features=evaluation_features[training_indices],
        target=evaluation_qtail_target,
        steps=args.steps,
        seed=args.seed,
        label="evaluation_qtail",
        device=device,
        out=args.out,
        checkpoint_every_steps=args.checkpoint_every_steps,
        completed_models=["evaluation_source"],
        environment_contract=runtime_environment,
    )
    holdout_source_pred = predict_allocation(
        evaluation_source_model,
        evaluation_features[holdout_indices],
    )
    holdout_qtail_pred = predict_allocation(
        evaluation_qtail_model,
        evaluation_features[holdout_indices],
    )
    rare_coverage = rare_instruction_fingerprint_coverage(
        rows,
        training_indices,
        holdout_indices,
        holdout_source_pred,
        holdout_qtail_pred,
    )
    rare_coverage_path = (
        args.out / "droid_rare_instruction_fingerprint_coverage.json"
    )
    atomic_write_json(rare_coverage_path, rare_coverage)

    require_mount(args.required_mount)
    deployment_source_hist, deployment_source_pred, source_model, deployment_source_resume = train_once_audited(
        features=deployment_features,
        target=deployment_source,
        steps=args.steps,
        seed=args.seed,
        label="deployment_source",
        device=device,
        out=args.out,
        checkpoint_every_steps=args.checkpoint_every_steps,
        completed_models=["evaluation_source", "evaluation_qtail"],
        environment_contract=runtime_environment,
    )
    require_mount(args.required_mount)
    deployment_qtail_hist, deployment_qtail_pred, qtail_model, deployment_qtail_resume = train_once_audited(
        features=deployment_features,
        target=deployment_qtail,
        steps=args.steps,
        seed=args.seed,
        label="deployment_qtail",
        device=device,
        out=args.out,
        checkpoint_every_steps=args.checkpoint_every_steps,
        completed_models=[
            "evaluation_source",
            "evaluation_qtail",
            "deployment_source",
        ],
        environment_contract=runtime_environment,
    )

    require_mount(args.required_mount)
    holdout_position = {
        int(row_index): position
        for position, row_index in enumerate(holdout_indices)
    }
    for idx, row in enumerate(rows):
        split = "holdout" if idx in holdout_position else "training"
        holdout_idx = holdout_position.get(idx)
        row["evaluation_split"] = split
        row["tail_score"] = float(evaluation_tail_scores[idx])
        row["deployment_tail_score"] = float(deployment_tail_scores[idx])
        row["evaluation_source_target"] = float(evaluation_source[idx])
        row["evaluation_qtail_target"] = float(evaluation_qtail[idx])
        row["source_target"] = float(deployment_source[idx])
        row["qtail_target"] = float(deployment_qtail[idx])
        row["deployment_source_pred"] = float(deployment_source_pred[idx])
        row["deployment_qtail_pred"] = float(deployment_qtail_pred[idx])
        row["holdout_source_pred"] = (
            float(holdout_source_pred[holdout_idx])
            if holdout_idx is not None
            else ""
        )
        row["holdout_qtail_pred"] = (
            float(holdout_qtail_pred[holdout_idx])
            if holdout_idx is not None
            else ""
        )

    rows_path = args.out / "droid_shard_training_rows.csv"
    curve_path = args.out / "droid_training_curve.csv"
    write_csv(rows_path, rows)
    write_csv(
        curve_path,
        [
            {"model": model, "step": item["step"], "kl": item["kl"]}
            for model, history in (
                ("evaluation_source", evaluation_source_hist),
                ("evaluation_qtail", evaluation_qtail_hist),
                ("deployment_source", deployment_source_hist),
                ("deployment_qtail", deployment_qtail_hist),
            )
            for item in history
        ],
    )

    checkpoint_path = args.out / "qtail_droid_allocation_head.pt"
    atomic_torch_save(
        {
            "format_version": 4,
            "model_class": "AllocationHead",
            "dataset": "DROID full RLDS",
            "feature_names": base.FEATURE_NAMES,
            "feature_normalization": deployment_normalization,
            "evaluation_feature_normalization": evaluation_normalization,
            "deployment_qtail_state_dict": qtail_model.state_dict(),
            "deployment_source_state_dict": source_model.state_dict(),
            "evaluation_qtail_state_dict": evaluation_qtail_model.state_dict(),
            "evaluation_source_state_dict": evaluation_source_model.state_dict(),
            "training_steps_per_stage": args.steps,
            "total_steps_per_arm": args.steps * 2,
            "records_per_shard": args.records_per_shard,
            "full_record_mode": args.records_per_shard == 0,
            "seed": args.seed,
            "runtime_environment": runtime_environment,
            "runtime_environment_fingerprint": (
                runtime_environment_sha256
            ),
            "datasets": datasets,
            "holdout_contract": holdout_contract,
            "pt_source_audit": pt_source_audit,
        },
        checkpoint_path,
    )
    (
        intermediate_checkpoint_manifest_path,
        intermediate_checkpoint_manifest,
    ) = build_intermediate_checkpoint_manifest(
        out=args.out,
        steps=args.steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        seed=args.seed,
        device=device,
        environment_sha256=runtime_environment_sha256,
    )

    holdout_tail_scores = evaluation_tail_scores[holdout_indices]
    holdout_paths = np.asarray(
        [str(rows[index]["path"]) for index in holdout_indices]
    )
    ranked_holdout = np.lexsort((holdout_paths, -holdout_tail_scores))
    tail_count = max(1, int(np.ceil(len(holdout_indices) * 0.30)))
    extreme_count = max(1, int(np.ceil(len(holdout_indices) * 0.10)))
    tail_mask = np.zeros(len(holdout_indices), dtype=bool)
    tail_mask[ranked_holdout[:tail_count]] = True
    extreme_mask = np.zeros(len(holdout_indices), dtype=bool)
    extreme_mask[ranked_holdout[:extreme_count]] = True
    tail_cut = float(holdout_tail_scores[ranked_holdout[tail_count - 1]])
    extreme_cut = float(
        holdout_tail_scores[ranked_holdout[extreme_count - 1]]
    )
    uniform_share = 1.0 / len(holdout_indices)
    source_tail_share = float(holdout_source_pred[tail_mask].sum())
    qtail_tail_share = float(holdout_qtail_pred[tail_mask].sum())
    source_extreme_underallocation = float(
        np.mean(holdout_source_pred[extreme_mask] < uniform_share)
    )
    qtail_extreme_underallocation = float(
        np.mean(holdout_qtail_pred[extreme_mask] < uniform_share)
    )
    bootstrap = paired_bootstrap(
        holdout_source_pred,
        holdout_qtail_pred,
        tail_mask,
        np.asarray(
            [str(rows[index]["dataset"]) for index in holdout_indices]
        ),
        samples=args.bootstrap_samples,
        seed=FORMAL_BOOTSTRAP_SEED,
    )
    randomization = paired_arm_swap_randomization(
        holdout_source_pred,
        holdout_qtail_pred,
        tail_mask,
        samples=FORMAL_RANDOMIZATION_SAMPLES,
        seed=FORMAL_RANDOMIZATION_SEED,
    )
    predicted_tail_share_gain_pp = (qtail_tail_share - source_tail_share) * 100.0
    extreme_underallocation_reduction_pp = (
        source_extreme_underallocation - qtail_extreme_underallocation
    ) * 100.0
    hypothesis_gate = heldout_hypothesis_gate(
        tail_share_gain_pp=predicted_tail_share_gain_pp,
        extreme_underallocation_reduction_pp=extreme_underallocation_reduction_pp,
        bootstrap=bootstrap,
        randomization=randomization,
    )
    total_bytes = sum(int(row["bytes"]) for row in rows)
    source_parameter_count = sum(
        parameter.numel() for parameter in source_model.parameters()
    )
    qtail_parameter_count = sum(
        parameter.numel() for parameter in qtail_model.parameters()
    )

    report = {
        "generated_at": now(),
        "status": "complete",
        "dataset": "DROID full RLDS mirror",
        "training_scope": (
            "all_complete_shards_all_decodable_records"
            if formal_run
            else "bounded_test_subset"
        ),
        "claim_boundary": [
            "This is a real DROID record-informed Q-Tail allocation-head training run.",
            "In full mode every complete local TFRecord shard is streamed through its final decodable record.",
            f"Held-out metrics use a deterministic release-stratified {args.holdout_fraction:.0%} shard split that is never used to fit feature transforms, instruction rarity, PT allocation targets, or model parameters.",
            "Source and Q-Tail use identical architecture, optimizer, features, seed, device, and training steps in both evaluation and deployment stages.",
            "Bootstrap intervals and paired arm-swap randomization are conditional on the fixed split, fixed tail taxonomy, and fixed trained model predictions; they do not include retraining or split uncertainty.",
            "Tail and extreme membership are preregistered top-30% and top-10% rankings within the held-out score distribution; these evaluation memberships and cutoffs are not training-fit quantities.",
            "This does not claim end-to-end DROID robot-policy improvement; that requires same-policy retraining and evaluation.",
            "Rare instruction coverage is an auxiliary byte-level fingerprint discovery proxy, not semantic task coverage or policy success.",
        ],
        "data_dir": str(args.data_dir),
        "input_audit": input_audit,
        "datasets": datasets,
        "release_composition": release_composition,
        "shard_count": len(rows),
        "total_bytes": total_bytes,
        "total_tib": total_bytes / (1024**4),
        "steps": args.steps,
        "total_steps_per_arm": args.steps * 2,
        "seed": args.seed,
        "formal_protocol": {
            "locked": formal_run,
            "seed": FORMAL_SEED,
            "steps_per_stage": FORMAL_STEPS_PER_STAGE,
            "checkpoint_every_steps": FORMAL_CHECKPOINT_EVERY_STEPS,
            "expected_checkpoint_steps": [
                0,
                5_000,
                10_000,
                15_000,
                20_000,
            ],
            "expected_checkpoint_count": 20,
            "holdout_fraction": FORMAL_HOLDOUT_FRACTION,
            "holdout_shards_per_release": FORMAL_HOLDOUT_SHARDS_PER_RELEASE,
            "holdout_relative_path_sha256": (
                FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
            ),
            "holdout_membership_path_scope": (
                "official_release_relative_path"
            ),
            "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
            "randomization_samples": FORMAL_RANDOMIZATION_SAMPLES,
            "randomization_seed": FORMAL_RANDOMIZATION_SEED,
            "min_record_parse_rate": FORMAL_MIN_RECORD_PARSE_RATE,
            "min_record_scan_complete_rate": (
                FORMAL_MIN_RECORD_SCAN_COMPLETE_RATE
            ),
            "require_verified_mirror": True,
            "pt_source_sha256": FORMAL_PT_SOURCE_SHA256,
            "expected_objects": FORMAL_EXPECTED_OBJECTS,
            "expected_tfrecords": FORMAL_EXPECTED_TFRECORDS,
            "expected_bytes": FORMAL_EXPECTED_BYTES,
            "expected_records": FORMAL_EXPECTED_RECORDS,
            "release_contract": FORMAL_RELEASE_CONTRACT,
        },
        "pt_source_audit": pt_source_audit,
        "holdout_evaluation": {
            **holdout_contract,
            "normalization_fit": "training_shards_only",
            "tail_taxonomy_scope": "training_shards_fit_applied_to_holdout",
            "instruction_rarity_fit": "training_shards_only",
            "pt_allocation_fit": "training_shards_only",
            "evaluation_predictions_scope": "holdout_shards_only",
        },
        "tail_score_contract": evaluation_normalization["tail_score_contract"],
        "deployment_tail_score_contract": deployment_normalization[
            "tail_score_contract"
        ],
        "compute_audit": {
            "source_steps": args.steps * 2,
            "qtail_steps": args.steps * 2,
            "evaluation_source_steps": args.steps,
            "evaluation_qtail_steps": args.steps,
            "deployment_source_steps": args.steps,
            "deployment_qtail_steps": args.steps,
            "evaluation_source_optimizer_updates": args.steps,
            "evaluation_qtail_optimizer_updates": args.steps,
            "deployment_source_optimizer_updates": args.steps,
            "deployment_qtail_optimizer_updates": args.steps,
            "source_optimizer_updates": args.steps * 2,
            "qtail_optimizer_updates": args.steps * 2,
            "optimizer_update_semantics": OPTIMIZER_UPDATE_SEMANTICS,
            "architecture": architecture,
            "same_architecture": True,
            "same_optimizer": OPTIMIZER_SIGNATURE,
            "same_seed": True,
            "same_features": True,
            "same_device": True,
            "training_device": str(device),
            "runtime_environment": runtime_environment,
            "runtime_environment_fingerprint": (
                runtime_environment_sha256
            ),
            "same_environment_fingerprint": True,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "source_parameter_count": source_parameter_count,
            "qtail_parameter_count": qtail_parameter_count,
            "same_parameter_count": source_parameter_count == qtail_parameter_count,
            "parameter_count": qtail_parameter_count,
            "resume": {
                "evaluation_source": evaluation_source_resume,
                "evaluation_qtail": evaluation_qtail_resume,
                "deployment_source": deployment_source_resume,
                "deployment_qtail": deployment_qtail_resume,
            },
        },
        "intermediate_checkpoint_audit": {
            "status": intermediate_checkpoint_manifest["status"],
            "manifest": str(intermediate_checkpoint_manifest_path),
            "contract": intermediate_checkpoint_manifest["contract"],
            "actual_checkpoint_count": (
                intermediate_checkpoint_manifest[
                    "actual_checkpoint_count"
                ]
            ),
            "paired_feature_signatures_equal": (
                intermediate_checkpoint_manifest["contract"][
                    "paired_feature_signatures_equal"
                ]
            ),
            "initialized_state_signatures_equal": (
                intermediate_checkpoint_manifest["contract"][
                    "initialized_state_signatures_equal"
                ]
            ),
            "all_checkpoint_hashes_recorded": all(
                len(str(entry.get("sha256", ""))) == 64
                for entry in intermediate_checkpoint_manifest["entries"]
            ),
        },
        "model_artifact": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "feature_names": base.FEATURE_NAMES,
        },
        "trajectory_evidence": {
            "tfrecord_shards_attempted": len(rows),
            "tfrecord_shards_parsed": parsed_shards,
            "record_parse_rate": parse_rate,
            "full_record_mode": args.records_per_shard == 0,
            "record_scan_complete_shards": scan_complete_shards,
            "record_scan_complete_rate": scan_complete_rate if args.records_per_shard == 0 else None,
            "records_per_shard_cap": args.records_per_shard or None,
            "records_decoded": sum(int(row["records_decoded"]) for row in rows),
            "mean_episode_steps": float(np.mean([row["mean_episode_steps"] for row in rows])),
        },
        "effect_metrics": {
            "evaluation_scope": "deterministic_release_stratified_heldout_shards",
            "tail_definition": "heldout_top_30_percent_by_record_informed_tail_score_v2",
            "tail_selected_shards": tail_count,
            "tail_total_holdout_shards": len(holdout_indices),
            "tail_boundary_score": tail_cut,
            "source_pred_tail_share": source_tail_share,
            "qtail_pred_tail_share": qtail_tail_share,
            "predicted_tail_share_gain_pp": predicted_tail_share_gain_pp,
            "extreme_definition": "heldout_top_10_percent_by_record_informed_tail_score_v2",
            "extreme_selected_shards": extreme_count,
            "extreme_total_holdout_shards": len(holdout_indices),
            "extreme_boundary_score": extreme_cut,
            "source_extreme_underallocation_rate": source_extreme_underallocation,
            "qtail_extreme_underallocation_rate": qtail_extreme_underallocation,
            "extreme_underallocation_reduction_pp": extreme_underallocation_reduction_pp,
            "paired_bootstrap": bootstrap,
            "paired_arm_swap_randomization": randomization,
            "hypothesis_gate": hypothesis_gate,
            "consistent_with_pt_tail_goal": qtail_tail_share > source_tail_share,
        },
        "rare_instruction_fingerprint_coverage": rare_coverage,
        "evaluation_source_final_kl": evaluation_source_hist[-1]["kl"],
        "evaluation_qtail_final_kl": evaluation_qtail_hist[-1]["kl"],
        "deployment_source_final_kl": deployment_source_hist[-1]["kl"],
        "deployment_qtail_final_kl": deployment_qtail_hist[-1]["kl"],
        "training_histories": {
            "evaluation_source": evaluation_source_hist,
            "evaluation_qtail": evaluation_qtail_hist,
            "deployment_source": deployment_source_hist,
            "deployment_qtail": deployment_qtail_hist,
        },
        "artifacts": {
            "run_manifest": str(run_manifest_path),
            "feature_status": str(args.out / "droid_feature_extraction_status.json"),
            "feature_rows": str(args.out / "droid_shard_features.csv"),
            "feature_cache_manifest": str(cache_manifest_path),
            "feature_prewarm_status": str(
                args.out / "droid_feature_prewarm_status.json"
            ),
            "training_rows": str(rows_path),
            "training_curve": str(curve_path),
            "rare_instruction_fingerprint_coverage": str(
                rare_coverage_path
            ),
            "model_status": str(args.out / "droid_model_training_status.json"),
            "checkpoint": str(checkpoint_path),
            "intermediate_checkpoint_dir": str(args.out / "intermediate_checkpoints"),
            "intermediate_checkpoint_manifest": str(
                intermediate_checkpoint_manifest_path
            ),
            "empirical_pt_source": str(pt_source_snapshot),
            "environment_manifest": (
                str(args.out / "droid_environment_manifest.json")
                if (
                    args.out / "droid_environment_manifest.json"
                ).is_file()
                else None
            ),
            "environment_contract_selftest": (
                str(args.out / "droid_environment_contract_selftest.json")
                if (
                    args.out / "droid_environment_contract_selftest.json"
                ).is_file()
                else None
            ),
            "training_gate_order_selftest": (
                str(args.out / "droid_training_gate_order_selftest.json")
                if (
                    args.out / "droid_training_gate_order_selftest.json"
                ).is_file()
                else None
            ),
        },
    }
    report_path = args.out / "droid_full_training_report.json"
    atomic_write_json(report_path, report)

    completed_at = now()
    model_status_path = args.out / "droid_model_training_status.json"
    final_model_status = (
        json.loads(model_status_path.read_text(encoding="utf-8"))
        if model_status_path.is_file()
        else {}
    )
    atomic_write_json(
        model_status_path,
        {
            **final_model_status,
            "generated_at": completed_at,
            "status": "complete",
            "active_model": None,
            "completed_models": [
                "evaluation_source",
                "evaluation_qtail",
                "deployment_source",
                "deployment_qtail",
            ],
            "step": args.steps,
            "optimizer_updates_completed": args.steps,
            "total_steps": args.steps,
            "progress_percent": 100.0,
            "total_optimizer_updates_per_arm": args.steps * 2,
            "final_checkpoint": str(checkpoint_path),
        },
    )
    atomic_write_json(
        run_manifest_path,
        {
            **json.loads(run_manifest_path.read_text(encoding="utf-8")),
            "status": "complete",
            "completed_at": completed_at,
            "model_status": str(model_status_path),
            "training_report": str(report_path),
        },
    )
    artifact_manifest_path = args.out / "droid_artifact_manifest.json"
    training_status_path = args.out / "training_status.json"
    atomic_write_json(
        training_status_path,
        {
            **report,
            "artifact_manifest": str(artifact_manifest_path),
        },
    )

    artifact_paths = [
        run_manifest_path,
        args.out / "droid_local_md5_rehash_status.json",
        args.out / "droid_feature_extraction_status.json",
        args.out / "droid_shard_features.csv",
        cache_manifest_path,
        args.out / "droid_feature_prewarm_status.json",
        args.out / "droid_feature_prewarm_run.json",
        rows_path,
        curve_path,
        rare_coverage_path,
        model_status_path,
        checkpoint_path,
        intermediate_checkpoint_manifest_path,
        pt_source_snapshot,
        report_path,
        training_status_path,
        *(
            [args.out / "droid_environment_manifest.json"]
            if (args.out / "droid_environment_manifest.json").exists()
            else []
        ),
        *(
            [args.out / "droid_environment_contract_selftest.json"]
            if (
                args.out / "droid_environment_contract_selftest.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "droid_download_marker_selftest.json"]
            if (args.out / "droid_download_marker_selftest.json").exists()
            else []
        ),
        *(
            [args.out / "droid_mirror_verifier_selftest.json"]
            if (args.out / "droid_mirror_verifier_selftest.json").exists()
            else []
        ),
        *(
            [args.out / "droid_downloader_single_writer_selftest.json"]
            if (
                args.out
                / "droid_downloader_single_writer_selftest.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "droid_runtime_process_contract_selftest.json"]
            if (
                args.out
                / "droid_runtime_process_contract_selftest.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "droid_training_gate_order_selftest.json"]
            if (
                args.out / "droid_training_gate_order_selftest.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "uniclash_pre_checksum_gate.json"]
            if (args.out / "uniclash_pre_checksum_gate.json").exists()
            else []
        ),
        *(
            [args.out / "uniclash_pre_checksum_gate_selftest.json"]
            if (
                args.out / "uniclash_pre_checksum_gate_selftest.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "droid_live_partial_marker_rejection.json"]
            if (
                args.out / "droid_live_partial_marker_rejection.json"
            ).exists()
            else []
        ),
        *(
            [args.out / "droid_release_metadata_audit.json"]
            if (args.out / "droid_release_metadata_audit.json").exists()
            else []
        ),
        *(
            [args.out / "droid_transport_tuning_audit.json"]
            if (args.out / "droid_transport_tuning_audit.json").exists()
            else []
        ),
        *(
            [args.object_manifest]
            if args.object_manifest and args.object_manifest.exists()
            else []
        ),
        *(
            [args.download_verification]
            if args.download_verification and args.download_verification.exists()
            else []
        ),
        *(
            [args.out / "droid_protocol_selftest.json"]
            if (args.out / "droid_protocol_selftest.json").exists()
            else []
        ),
        *(
            [
                args.out / "droid_preflight_training_smoke.json",
                args.out / "droid_preflight_training_smoke_report.json",
            ]
            if (
                args.out / "droid_preflight_training_smoke.json"
            ).exists()
            and (
                args.out / "droid_preflight_training_smoke_report.json"
            ).exists()
            else []
        ),
        *sorted((args.out / "intermediate_checkpoints").glob("*.pt")),
    ]
    atomic_write_json(
        artifact_manifest_path,
        {
            "generated_at": now(),
            "status": "complete",
            "artifacts": [artifact_entry(path) for path in artifact_paths if path.exists()],
        },
    )
    publish_training_completion_marker(
        args.marker_dir,
        formal_run=formal_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    process_lock_handle.close()


if __name__ == "__main__":
    main()
