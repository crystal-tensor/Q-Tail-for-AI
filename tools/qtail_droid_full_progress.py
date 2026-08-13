#!/usr/bin/env python3
"""Build a bounded live status artifact for the ORICO DROID full pipeline."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

from qtail_merge_droid_artifact_manifest import (
    formal_droid_artifact_paths,
)
from qtail_verify_droid_stage_markers import (
    PIPELINE_GENERATION_CHECKS,
    PIPELINE_GENERATION_GATES,
    validate_final_bootstrap,
    validate_final_marker,
    validate_pipeline_generation_gate,
    validate_public_projection_marker,
    validate_training_marker,
)

REMOTE_BYTES = 3_700_745_265_151
OFFICIAL_TFRECORD_OBJECT_BYTES = 3_700_745_150_555
REMOTE_URI = "gs://gresearch/robotics/droid"
OPENX_EXPECTED_BYTES = 184_278_228_991
FORMAL_SEED = 11
FORMAL_HOLDOUT_FRACTION = 0.20
FORMAL_HOLDOUT_SHARDS_PER_RELEASE = 410
FORMAL_HOLDOUT_RELATIVE_PATH_SHA256 = (
    "16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767"
)
FORMAL_PT_SOURCE_SHA256 = (
    "59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f"
)
TIMELINE_VERSION = "qtail_droid_pipeline_timeline_v1"
EXPECTED_RELEASE_METADATA = {
    "1.0.0": {
        "dataset_name": "r2d2_faceblur",
        "dataset_version": "1.4.0",
        "shards": 2_048,
        "records": 92_233,
        "split_bytes": 1_834_749_018_029,
    },
    "1.0.1": {
        "dataset_name": "droid_101",
        "dataset_version": "0.0.1",
        "shards": 2_048,
        "records": 95_658,
        "split_bytes": 1_865_993_126_270,
    },
}
FORMAL_PROCESS_LOG_ARTIFACTS = (
    "droid_process_log_manifest.json",
    "process_logs_final/droid_full_pipeline.log",
    "process_logs_final/droid_feature_prewarm.log",
    "process_logs_final/pipeline_watchdog.log",
    "process_logs_final/progress_loop.log",
    "process_logs_final/progress_refresh.log",
    "process_logs_final/pipeline_generation_handoff.log",
    "process_logs_final/manual_endpoint_generation_handoff.log",
    "process_logs_final/qtail-web-services.log",
)
FORMAL_FINAL_QA_ARTIFACTS = (
    "uniclash_transport_guard_final.json",
    "download_progress_samples_final.json",
    "final_page_qa.json",
    "final_page_desktop.png",
    "final_page_mobile.png",
)
EMPTY_SUPERVISION_LOG_HREFS = {
    (
        "results/qtail_droid_full/live_logs/"
        "qtail_droid_launchd_stdout.log"
    ),
    (
        "results/qtail_droid_full/live_logs/"
        "qtail_uniclash_guard_stdout.log"
    ),
    (
        "results/qtail_droid_full/process_logs_final/"
        "qtail_droid_launchd_stdout.log"
    ),
    (
        "results/qtail_droid_full/process_logs_final/"
        "qtail_uniclash_guard_stdout.log"
    ),
}


class ArtifactLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = str(values.get("class") or "").split()
        href = str(values.get("href") or "")
        if "artifact" in classes and href:
            self.hrefs.append(href)


def json_artifact_semantics(path: Path) -> tuple[bool, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "JSON 产物无法解析"
    if not isinstance(payload, (dict, list)):
        return False, "JSON 产物顶层结构无效"
    if not isinstance(payload, dict):
        return True, None

    status = str(payload.get("status", "")).strip().lower()
    failure_statuses = {
        "failed",
        "error",
        "invalid",
        "blocked",
        "unavailable",
    }
    if status in failure_statuses or status.startswith(
        ("failed_", "error_", "invalid_", "blocked_")
    ):
        return False, f"产物语义状态为 {status}"
    for field in ("valid", "passed"):
        if field in payload and payload[field] is False:
            return False, f"产物语义字段 {field}=false"
    return True, None


def artifact_link_availability(repo_root: Path) -> dict:
    page_path = repo_root / "qtail-droid-full-training.html"
    parser = ArtifactLinkParser()
    try:
        parser.feed(page_path.read_text(encoding="utf-8"))
    except OSError:
        return {
            "status": "unavailable",
            "total": 0,
            "available": 0,
            "missing": 0,
            "items": {},
        }
    items: dict[str, dict] = {}
    final_qa_hrefs = {
        "results/qtail_droid_full/final_page_qa.json",
        "results/qtail_droid_full/final_page_desktop.png",
        "results/qtail_droid_full/final_page_mobile.png",
    }
    qa_path = repo_root / "results/qtail_droid_full/final_page_qa.json"
    qa_complete = read_json(qa_path).get("status") == "complete"
    for href in sorted(set(parser.hrefs)):
        local = (
            not href.startswith(("http://", "https://", "#", "/"))
            and "?" not in href
        )
        path = repo_root / href if local else None
        size = None
        if path:
            try:
                metadata = path.stat()
                if stat.S_ISREG(metadata.st_mode) and (
                    metadata.st_size > 0
                    or href in EMPTY_SUPERVISION_LOG_HREFS
                ):
                    size = metadata.st_size
            except OSError:
                size = None
        nonempty = size is not None
        semantics_valid = True
        semantics_reason = None
        if nonempty and path and path.suffix.lower() == ".json":
            semantics_valid, semantics_reason = json_artifact_semantics(path)
        exists = bool(
            nonempty
            and semantics_valid
            and (href not in final_qa_hrefs or qa_complete)
        )
        items[href] = {
            "available": exists,
            "bytes": size if exists else None,
            "empty_supervision_log": bool(
                exists
                and size == 0
                and href in EMPTY_SUPERVISION_LOG_HREFS
            ),
            "reason": (
                None
                if exists
                else (
                    "最终 QA 尚未成功完成"
                    if href in final_qa_hrefs
                    else (
                        semantics_reason
                        or "该证据尚未生成或为空"
                    )
                )
            ),
        }
    return {
        "status": "complete",
        "total": len(items),
        "available": sum(item["available"] for item in items.values()),
        "missing": sum(not item["available"] for item in items.values()),
        "items": items,
    }
EXPECTED_DROID_STEP_SCHEMA = {
    "action",
    "action_dict",
    "discount",
    "is_first",
    "is_last",
    "is_terminal",
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "observation",
    "reward",
}
REQUIRED_METADATA_GATES = {
    "official_checksum_manifest",
    "both_releases_verified",
    "combined_shards_4096",
    "combined_records_187891",
    "combined_split_bytes_match",
    "step_schemas_identical",
    "training_features_present",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_age_seconds(
    value: object,
    *,
    reference: datetime | None = None,
) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = reference or datetime.now(timezone.utc)
    return max(
        0.0,
        (current - parsed.astimezone(timezone.utc)).total_seconds(),
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def active_final_qa_preview(path: Path) -> bool:
    payload = read_json(path)
    try:
        if payload.get("status") != "preview_active":
            return False
        owner_pid = int(payload["owner_pid"])
        expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        if expires_at <= datetime.now(timezone.utc):
            return False
        os.kill(owner_pid, 0)
        command = subprocess.run(
            ["ps", "-p", str(owner_pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return (
            "qtail_verify_droid_page.mjs" in command
            or "qtail_orico_full_pipeline.sh" in command
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def resolve_pipeline_stage(marker_state: dict) -> str:
    marker_state["final_page_qa_effective"] = bool(
        marker_state.get("final_page_qa_complete", False)
    )
    marker_state["final_page_qa_in_progress"] = bool(
        (
            marker_state.get("final_page_qa_preview_active", False)
            or marker_state.get("final_page_qa_bootstrap_active", False)
        )
        and not marker_state["final_page_qa_complete"]
    )
    if marker_state["final_page_qa_effective"]:
        return "complete"
    if (
        marker_state["final_page_qa_in_progress"]
        or marker_state.get("droid_training_complete", False)
    ):
        return "final_page_qa"
    if marker_state.get("droid_checksum_verified", False):
        return "training"
    if marker_state.get("droid_download_complete", False):
        return "checksum_verification"
    if marker_state.get("openx_migration_complete", False):
        return "droid_full_download"
    return "migrating_existing_assets"


def formal_artifact_requirement_paths(
    result_root: Path,
    marker_state: dict,
) -> dict[str, set[str]]:
    baseline = {
        str(path.resolve())
        for path in formal_droid_artifact_paths(result_root)
    }
    process_logs = {
        str(result_root / name)
        for name in FORMAL_PROCESS_LOG_ARTIFACTS
    }
    final_qa = {
        str(result_root / name)
        for name in FORMAL_FINAL_QA_ARTIFACTS
    }
    required = set(baseline)
    if marker_state.get("final_page_qa_effective", False):
        required.update(process_logs)
    if marker_state.get("final_page_qa_complete", False):
        required.update(final_qa)
    return {
        "baseline": baseline,
        "process_logs": process_logs,
        "final_qa": final_qa,
        "required": required,
    }


def public_final_projection_is_committed(latest: dict, audit: dict) -> bool:
    requirements = audit.get("requirements", [])
    final_requirement = next(
        (
            item
            for item in requirements
            if isinstance(item, dict) and item.get("id") == "final_page_qa"
        ),
        {},
    )
    evidence = final_requirement.get("evidence", {})
    return bool(
        latest.get("status") == "complete"
        and latest.get("stage") == "complete"
        and audit.get("status") == "complete"
        and int(audit.get("passed_requirements", -1)) == 9
        and int(audit.get("total_requirements", -1)) == 9
        and final_requirement.get("passed") is True
        and evidence.get("committed") is True
        and evidence.get("preview_active") is False
        and evidence.get("qa_state") == "committed"
    )


def atomic_write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{id(payload)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def bind_timeline_sample(
    sample: dict,
    sequence: int,
    previous_sample_sha256: str | None,
) -> dict:
    bound = {
        "sequence": sequence,
        "previous_sample_sha256": previous_sample_sha256,
        **sample,
    }
    bound["sample_sha256"] = canonical_sha256(bound)
    return bound


def validate_timeline_chain(samples: list[dict]) -> None:
    previous: str | None = None
    previous_time: datetime | None = None
    for sequence, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"timeline sample {sequence} is not an object")
        sample = dict(raw_sample)
        observed_hash = sample.pop("sample_sha256", None)
        if sample.get("sequence") != sequence:
            raise ValueError(f"timeline sequence mismatch at {sequence}")
        if sample.get("previous_sample_sha256") != previous:
            raise ValueError(f"timeline previous hash mismatch at {sequence}")
        expected_hash = canonical_sha256(sample)
        if observed_hash != expected_hash:
            raise ValueError(f"timeline sample hash mismatch at {sequence}")
        generated_at = datetime.fromisoformat(str(sample["generated_at"]))
        if previous_time is not None and generated_at < previous_time:
            raise ValueError(f"timeline timestamp moved backwards at {sequence}")
        previous = observed_hash
        previous_time = generated_at


def update_pipeline_timeline(
    path: Path,
    download_samples_path: Path,
    legacy_download_samples: list[dict],
    live_sample: dict,
) -> dict:
    existing = read_json(path)
    samples = existing.get("samples", []) if existing else []
    if existing and existing.get("version") != TIMELINE_VERSION:
        raise ValueError("pipeline timeline version mismatch")
    if not isinstance(samples, list):
        raise ValueError("pipeline timeline samples must be a list")
    validate_timeline_chain(samples)

    legacy_import = existing.get("legacy_import", {}) if existing else {}
    if not samples:
        source_sha256 = file_sha256(download_samples_path)
        for raw_sample in legacy_download_samples:
            imported = {
                "generated_at": raw_sample["generated_at"],
                "kind": "legacy_download_sample",
                "download": {
                    "physical_bytes": int(raw_sample.get("bytes", 0)),
                    "completed_logical_bytes": int(
                        raw_sample.get("completed_logical_bytes", 0)
                    ),
                    "partial_allocated_bytes": int(
                        raw_sample.get("partial_allocated_bytes", 0)
                    ),
                },
                "scope": (
                    "Imported from the original full-retention download history; "
                    "non-download metrics were not recorded in this legacy sample."
                ),
            }
            previous = samples[-1]["sample_sha256"] if samples else None
            samples.append(
                bind_timeline_sample(imported, len(samples), previous)
            )
        legacy_import = {
            "source": str(download_samples_path),
            "source_sha256_at_import": source_sha256,
            "imported_samples": len(legacy_download_samples),
        }

    if (
        not samples
        or samples[-1].get("generated_at") != live_sample["generated_at"]
    ):
        previous = samples[-1]["sample_sha256"] if samples else None
        samples.append(
            bind_timeline_sample(live_sample, len(samples), previous)
        )
    validate_timeline_chain(samples)
    payload = {
        "version": TIMELINE_VERSION,
        "status": (
            "complete"
            if live_sample.get("stage") == "complete"
            else "recording"
        ),
        "retention": "full_pipeline_history",
        "sample_count": len(samples),
        "first_generated_at": (
            samples[0]["generated_at"] if samples else None
        ),
        "last_generated_at": (
            samples[-1]["generated_at"] if samples else None
        ),
        "chain_head_sha256": (
            samples[-1]["sample_sha256"] if samples else None
        ),
        "legacy_import": legacy_import,
        "samples": samples,
    }
    atomic_write_json(path, payload)
    return {
        key: payload[key]
        for key in (
            "version",
            "status",
            "retention",
            "sample_count",
            "first_generated_at",
            "last_generated_at",
            "chain_head_sha256",
        )
    }


def tail(path: Path, lines: int = 80) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except Exception:
        return []


def scan_tree(root: Path) -> dict:
    logical_bytes = 0
    allocated_bytes = 0
    completed_logical_bytes = 0
    partial_logical_bytes = 0
    partial_allocated_bytes = 0
    inflight_logical_bytes = 0
    inflight_allocated_bytes = 0
    files = 0
    tfrecords = 0
    tfrecord_bytes = 0
    partials = 0
    inflight_files = 0
    ignored_metadata_files = 0
    ignored_metadata_bytes = 0
    transport_metadata_files = 0
    transport_metadata_bytes = 0
    errors = 0
    if not root.exists():
        return {
            "bytes": 0,
            "logical_bytes": 0,
            "allocated_bytes": 0,
            "completed_logical_bytes": 0,
            "partial_logical_bytes": 0,
            "partial_allocated_bytes": 0,
            "inflight_logical_bytes": 0,
            "inflight_allocated_bytes": 0,
            "files": 0,
            "tfrecords": 0,
            "tfrecord_bytes": 0,
            "partials": 0,
            "inflight_files": 0,
            "ignored_metadata_files": 0,
            "ignored_metadata_bytes": 0,
            "transport_metadata_files": 0,
            "transport_metadata_bytes": 0,
            "scan_errors": 0,
        }
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            try:
                stat = path.stat()
                size = stat.st_size
                allocated = getattr(stat, "st_blocks", 0) * 512
            except OSError:
                errors += 1
                continue
            if name == ".DS_Store" or name.startswith("._"):
                ignored_metadata_files += 1
                ignored_metadata_bytes += size
                continue
            lowered = name.lower()
            if ".qtail.part" in lowered and lowered.endswith(".headers"):
                transport_metadata_files += 1
                transport_metadata_bytes += size
                continue
            logical_bytes += size
            allocated_bytes += allocated
            files += 1
            if lowered.endswith(".inflight"):
                partials += 1
                inflight_files += 1
                inflight_logical_bytes += size
                inflight_allocated_bytes += min(allocated, size)
            elif (
                ".gstmp" in lowered
                or ".invalid-" in lowered
                or lowered.endswith((".part", ".tmp"))
            ):
                partials += 1
                partial_logical_bytes += size
                partial_allocated_bytes += min(allocated, size)
            elif "tfrecord" in lowered:
                completed_logical_bytes += size
                tfrecords += 1
                tfrecord_bytes += size
            else:
                completed_logical_bytes += size
    progress_bytes = completed_logical_bytes + partial_allocated_bytes
    return {
        "bytes": progress_bytes,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "completed_logical_bytes": completed_logical_bytes,
        "partial_logical_bytes": partial_logical_bytes,
        "partial_allocated_bytes": partial_allocated_bytes,
        "inflight_logical_bytes": inflight_logical_bytes,
        "inflight_allocated_bytes": inflight_allocated_bytes,
        "files": files,
        "tfrecords": tfrecords,
        "tfrecord_bytes": tfrecord_bytes,
        "partials": partials,
        "inflight_files": inflight_files,
        "ignored_metadata_files": ignored_metadata_files,
        "ignored_metadata_bytes": ignored_metadata_bytes,
        "transport_metadata_files": transport_metadata_files,
        "transport_metadata_bytes": transport_metadata_bytes,
        "scan_errors": errors,
    }


def git_state(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {"exists": False, "path": str(repo)}
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return {
        "exists": True,
        "path": str(repo),
        "commit": proc.stdout.strip() if proc.returncode == 0 else None,
    }


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def droid_source_probe_marker_summary(
    marker_path: Path,
    report_path: Path,
    *,
    job_root: Path,
) -> dict:
    marker = read_json(marker_path)
    report = read_json(report_path)
    report_sha256 = file_sha256(report_path)
    storage = report.get("storage", {})
    checks = {
        "semantic_marker": (
            marker.get("format_version")
            == "qtail_droid_source_probe_marker_v2"
        ),
        "marker_status": marker.get("status") == "verified",
        "marker_source": marker.get("source") == REMOTE_URI,
        "marker_bytes": marker.get("remote_bytes") == REMOTE_BYTES,
        "marker_job_root": marker.get("job_root") == str(job_root),
        "marker_report_path": marker.get("report") == str(report_path),
        "marker_report_sha256": (
            bool(report_sha256)
            and marker.get("report_sha256") == report_sha256
        ),
        "marker_capacity_gate": (
            marker.get("capacity_gate_passed_at_probe") is True
        ),
        "report_status": report.get("status") == "verified",
        "report_source": report.get("source") == REMOTE_URI,
        "report_bytes": report.get("remote_bytes") == REMOTE_BYTES,
        "report_job_root": report.get("job_root") == str(job_root),
        "report_capacity_gate": (
            isinstance(storage, dict)
            and storage.get("capacity_gate_passed") is True
        ),
    }
    return {
        "valid": all(checks.values()),
        "marker": str(marker_path),
        "report": str(report_path),
        "format_version": marker.get("format_version"),
        "report_sha256": report_sha256,
        "checks": checks,
        "claim_boundary": (
            "This validates the local semantic source marker and its report "
            "hash binding. It does not independently re-query the remote "
            "bucket or replace the live capacity gate."
        ),
    }


def pipeline_generation_gate_summary(
    payload: dict,
    *,
    gate_path: Path,
    script_path: Path,
) -> dict:
    expected_gates = PIPELINE_GENERATION_GATES
    expected_checks = PIPELINE_GENERATION_CHECKS
    entries = [
        item
        for item in payload.get("gates", [])
        if isinstance(item, dict)
    ]
    by_gate = {
        str(item.get("gate")): item
        for item in entries
        if item.get("gate")
    }
    all_checks_valid = bool(
        set(by_gate) == set(expected_gates)
        and len(entries) == len(expected_gates)
        and all(
            item.get("passed") is True
            and isinstance(item.get("checks"), dict)
            and set(item["checks"]) == expected_checks
            and all(
                value is True
                for value in item["checks"].values()
            )
            for item in by_gate.values()
        )
    )
    gate_order_valid = [
        item.get("gate")
        for item in entries
    ] == list(expected_gates)
    pids = {
        item.get("pid")
        for item in by_gate.values()
        if (
            isinstance(item.get("pid"), int)
            and not isinstance(item.get("pid"), bool)
            and item.get("pid") > 0
        )
    }
    lock_owner_pids = {
        item.get("lock_owner_pid")
        for item in by_gate.values()
        if (
            isinstance(item.get("lock_owner_pid"), int)
            and not isinstance(item.get("lock_owner_pid"), bool)
            and item.get("lock_owner_pid") > 0
        )
    }
    script_hashes = {
        item.get("current_script_sha256")
        for item in by_gate.values()
        if isinstance(item.get("current_script_sha256"), str)
    }
    marker_hashes = {
        item.get("marker_script_sha256")
        for item in by_gate.values()
        if isinstance(item.get("marker_script_sha256"), str)
    }
    expected_command = f"/bin/zsh {script_path}"
    command_binding_valid = bool(
        len(entries) == len(expected_gates)
        and all(
            item.get("command") == expected_command
            and item.get("expected_command") == expected_command
            for item in entries
        )
    )
    single_pipeline_pid = (
        next(iter(pids))
        if len(pids) == 1
        else None
    )
    pid_binding_valid = bool(
        single_pipeline_pid is not None
        and lock_owner_pids == {single_pipeline_pid}
    )
    current_script_sha256 = file_sha256(script_path)
    source_hash_binding_valid = bool(
        current_script_sha256
        and script_hashes == {current_script_sha256}
        and marker_hashes == {current_script_sha256}
    )
    current_gate_payload = read_json(gate_path)
    payload_matches_current_file = bool(
        current_gate_payload
        and current_gate_payload == payload
    )
    strict_errors = validate_pipeline_generation_gate(
        gate_path,
        script_path=script_path,
    )
    valid = bool(
        payload.get("format_version")
        == "qtail_pipeline_generation_gate_v1"
        and payload.get("status") == "passed"
        and payload.get("latest_gate") == "pre-formal-training"
        and all_checks_valid
        and gate_order_valid
        and pid_binding_valid
        and command_binding_valid
        and source_hash_binding_valid
        and payload_matches_current_file
        and not strict_errors
    )
    return {
        "valid": valid,
        "format_version": payload.get("format_version"),
        "status": payload.get("status", "missing"),
        "latest_gate": payload.get("latest_gate"),
        "expected_gates": list(expected_gates),
        "observed_gates": [
            item.get("gate")
            for item in entries
        ],
        "all_checks_valid": all_checks_valid,
        "gate_order_valid": gate_order_valid,
        "single_pipeline_pid": single_pipeline_pid,
        "pid_binding_valid": pid_binding_valid,
        "command_binding_valid": command_binding_valid,
        "source_hash_binding_valid": source_hash_binding_valid,
        "payload_matches_current_file": payload_matches_current_file,
        "strict_validation_errors": strict_errors,
        "single_script_sha256": (
            next(iter(script_hashes))
            if len(script_hashes) == 1
            else None
        ),
        "current_script_sha256": current_script_sha256,
        "claim_boundary": (
            "Formal completion requires all three irreversible-stage checks "
            "to pass in order under one pipeline PID, matching lock owner, "
            "exact command, and the current script SHA-256."
        ),
    }


def checkpoint_manifest_projection(
    manifest: dict,
    result_root: Path,
) -> dict:
    projected = copy.deepcopy(manifest) if isinstance(manifest, dict) else {}
    checkpoint_root = result_root / "intermediate_checkpoints"
    stages = (
        "evaluation_source",
        "evaluation_qtail",
        "deployment_source",
        "deployment_qtail",
    )
    steps = (0, 5_000, 10_000, 15_000, 20_000)
    saved_keys = []
    for stage in stages:
        for step in steps:
            path = checkpoint_root / f"{stage}_step_{step:06d}.pt"
            try:
                if path.is_file() and path.stat().st_size > 0:
                    saved_keys.append(f"{stage}:{step}")
            except OSError:
                continue
    projected_entries = []
    for raw in projected.get("entries", []):
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        stage = str(entry.get("model_stage", ""))
        try:
            step = int(entry.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        expected_path = (
            checkpoint_root / f"{stage}_step_{step:06d}.pt"
            if stage in stages and step in steps
            else None
        )
        path = Path(str(entry.get("path", "")))
        valid = False
        try:
            valid = bool(
                expected_path is not None
                and path == expected_path
                and path.is_file()
                and path.stat().st_size > 0
                and path.stat().st_size == int(entry.get("bytes", -1))
                and int(entry.get("optimizer_updates_completed", -1)) == step
                and int(entry.get("target_steps", -1)) == 20_000
                and len(str(entry.get("sha256", ""))) == 64
                and file_sha256(path) == entry.get("sha256")
            )
        except (OSError, TypeError, ValueError):
            valid = False
        entry["display_hash_verified"] = valid
        projected_entries.append(entry)
    projected["entries"] = projected_entries
    projected["saved_checkpoint_keys"] = saved_keys
    projected["display_contract"] = (
        "SAVED means a nonempty expected file exists; VERIFIED additionally "
        "requires a valid DROID_TRAINING_COMPLETE marker."
    )
    return projected


def directory_summary(root: Path) -> dict:
    files = 0
    total_bytes = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                total_bytes += path.stat().st_size
                files += 1
            except OSError:
                continue
    return {
        "exists": root.is_dir(),
        "path": str(root),
        "file_count": files,
        "bytes": total_bytes,
    }


def verify_sha256_manifest(root: Path, manifest: Path) -> dict:
    errors: list[dict[str, str]] = []
    verified_files = 0
    if not root.is_dir() or not manifest.is_file():
        return {
            "verified": False,
            "verified_file_count": 0,
            "error_count": 1,
            "errors": [{"path": str(manifest), "error": "manifest is missing"}],
        }
    root_resolved = root.resolve()
    for raw_line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*")
            path = (root / relative).resolve()
            if not path.is_relative_to(root_resolved):
                raise ValueError("path escapes snapshot root")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"sha256 mismatch: expected={expected} actual={actual}")
            verified_files += 1
        except (OSError, ValueError) as error:
            errors.append({"line": raw_line, "error": str(error)})
    return {
        "verified": bool(verified_files) and not errors,
        "verified_file_count": verified_files,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def verify_snapshot_source_parity(source_root: Path, manifest: Path) -> dict:
    errors: list[dict[str, str]] = []
    verified_files = 0
    seen_paths: set[str] = set()
    if not source_root.is_dir() or not manifest.is_file():
        return {
            "verified": False,
            "verified_file_count": 0,
            "error_count": 1,
            "errors": [
                {
                    "path": str(manifest),
                    "error": "source root or snapshot manifest is missing",
                }
            ],
        }
    for raw_line in manifest.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*")
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or str(relative_path) in seen_paths
            ):
                raise ValueError("unsafe or duplicate source-relative path")
            seen_paths.add(str(relative_path))
            source_path = source_root / relative_path
            actual = file_sha256(source_path)
            if actual != expected:
                raise ValueError(
                    f"source sha256 mismatch: expected={expected} actual={actual}"
                )
            verified_files += 1
        except (OSError, ValueError) as error:
            errors.append({"line": raw_line, "error": str(error)})
    return {
        "verified": bool(verified_files) and not errors,
        "verified_file_count": verified_files,
        "error_count": len(errors),
        "errors": errors[:20],
        "contract": (
            "Every path in the ORICO snapshot manifest must have identical "
            "content at the current workspace-relative source path."
        ),
    }


def snapshot_publication_projection(
    payload: dict,
    *,
    audit_path: Path,
    repo_root: Path,
    snapshot_root: Path,
    manifest_path: Path,
    manifest_sha256: str | None,
    verified_file_count: int,
) -> dict:
    try:
        valid = bool(
            payload.get("status") == "passed"
            and payload.get("format_version")
            == "qtail_orchestration_snapshot_sync_v1"
            and payload.get("repo_root") == str(repo_root)
            and payload.get("snapshot") == str(snapshot_root)
            and payload.get("manifest") == str(manifest_path)
            and payload.get("manifest_sha256") == manifest_sha256
            and int(payload.get("file_count", -1))
            == verified_file_count
            and int(payload.get("verified_file_count", -1))
            == verified_file_count
            and payload.get(
                "progress_refresh_lock_held_during_swap"
            )
            is True
            and payload.get(
                "audit_committed_before_progress_lock_release"
            )
            is True
            and payload.get("atomic_directory_swap")
            == "macos_renameatx_np_RENAME_SWAP"
        )
    except (TypeError, ValueError):
        valid = False
    return {
        **payload,
        "path": str(audit_path),
        "valid": valid,
    }


def verify_artifact_manifest(manifest: dict) -> dict:
    entries = manifest.get("artifacts", [])
    errors: list[dict[str, str]] = []
    verified_files = 0
    seen_paths: set[str] = set()
    if manifest.get("status") != "complete" or not isinstance(entries, list):
        return {
            "verified": False,
            "verified_file_count": 0,
            "error_count": 1,
            "errors": [{"error": "artifact manifest is not complete"}],
        }
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append({"error": "artifact entry is not an object"})
            continue
        raw_path = str(entry.get("path", ""))
        try:
            if not raw_path or raw_path in seen_paths:
                raise ValueError("artifact path is empty or duplicated")
            seen_paths.add(raw_path)
            path = Path(raw_path)
            if not path.is_file():
                raise ValueError("artifact file is missing")
            actual_bytes = path.stat().st_size
            expected_bytes = int(entry.get("bytes", -1))
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"byte mismatch: expected={expected_bytes} "
                    f"actual={actual_bytes}"
                )
            expected_sha256 = str(entry.get("sha256", ""))
            actual_sha256 = file_sha256(path)
            if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
                raise ValueError(
                    f"sha256 mismatch: expected={expected_sha256} "
                    f"actual={actual_sha256}"
                )
            verified_files += 1
        except (OSError, TypeError, ValueError) as error:
            errors.append({"path": raw_path, "error": str(error)})
    return {
        "verified": bool(verified_files) and not errors,
        "verified_file_count": verified_files,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def web_service_snapshot(port: int) -> dict:
    url = f"http://127.0.0.1:{port}/qtail-droid-full-training"
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "qtail-runtime-audit/1.0"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3) as response:
            sample = response.read(8192).decode("utf-8", errors="replace")
            status = int(response.status)
        content_gate_passed = "Q-Tail DROID 全量训练" in sample
        return {
            "port": port,
            "url": url,
            "http_status": status,
            "content_gate_passed": content_gate_passed,
            "healthy": status == 200 and content_gate_passed,
            "error": None,
        }
    except (OSError, urllib.error.URLError) as error:
        return {
            "port": port,
            "url": url,
            "http_status": None,
            "content_gate_passed": False,
            "healthy": False,
            "error": str(error),
        }


def launchd_supervision_snapshot(repo_root: Path) -> dict:
    definitions = {
        "pipeline": {
            "label": "com.qtail.droid-full-pipeline",
            "plist": "com.qtail.droid-full-pipeline.plist",
        },
        "transport_guard": {
            "label": "com.qtail.uniclash-transport-guard",
            "plist": "com.qtail.uniclash-transport-guard.plist",
        },
    }
    launchctl = subprocess.run(
        ["/bin/launchctl", "list"],
        text=True,
        capture_output=True,
        check=False,
    )
    loaded = {}
    for line in launchctl.stdout.splitlines():
        columns = line.split()
        if len(columns) < 3:
            continue
        if columns[0] != "-" and not columns[0].isdigit():
            continue
        if columns[1] != "-" and not columns[1].lstrip("-").isdigit():
            continue
        loaded[columns[-1]] = {
            "pid": None if columns[0] == "-" else int(columns[0]),
            "last_exit_status": (
                None if columns[1] == "-" else int(columns[1])
            ),
        }

    rows = {}
    for name, definition in definitions.items():
        source = repo_root / "launchd" / definition["plist"]
        installed = (
            Path.home() / "Library" / "LaunchAgents" / definition["plist"]
        )
        try:
            config = plistlib.loads(installed.read_bytes())
        except (OSError, plistlib.InvalidFileException):
            config = {}
        arguments = config.get("ProgramArguments", [])
        if not isinstance(arguments, list):
            arguments = []
        checks = {
            "loaded": definition["label"] in loaded,
            "source_plist_exists": source.is_file(),
            "installed_plist_exists": installed.is_file(),
            "installed_matches_source": (
                source.is_file()
                and installed.is_file()
                and file_sha256(source) == file_sha256(installed)
            ),
            "label_matches": config.get("Label") == definition["label"],
            "run_at_load": config.get("RunAtLoad") is True,
        }
        if name == "pipeline":
            checks.update(
                {
                    "scheduled_retry_300s": int(
                        config.get("StartInterval", -1)
                    )
                    == 300,
                    "launcher_bound": arguments
                    == [
                        "/bin/zsh",
                        str(
                            repo_root
                            / "scripts/qtail_droid_terminal_launcher.command"
                        ),
                    ],
                }
            )
        else:
            checks.update(
                {
                    "keep_alive": config.get("KeepAlive") is True,
                    "guard_bound": str(
                        repo_root / "tools/qtail_uniclash_transport_guard.py"
                    )
                    in arguments,
                    "interface_bound": (
                        "--expected-interface" in arguments
                        and "en1" in arguments
                    ),
                    "network_service_bound": (
                        "--network-service" in arguments
                        and "Wi-Fi" in arguments
                    ),
                    "two_second_heartbeat": (
                        "--interval-seconds" in arguments
                        and "2" in arguments
                    ),
                }
            )
        rows[name] = {
            "label": definition["label"],
            "source_plist": str(source),
            "installed_plist": str(installed),
            "source_sha256": file_sha256(source),
            "installed_sha256": file_sha256(installed),
            "launchctl": loaded.get(definition["label"]),
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "launchctl_returncode": launchctl.returncode,
        "labels": rows,
        "passed": (
            launchctl.returncode == 0
            and all(row["passed"] for row in rows.values())
        ),
        "claim_boundary": (
            "This proves the two user LaunchAgents are loaded and their "
            "installed plists match the audited repository configuration. "
            "It does not prove a future reboot will preserve external power, "
            "network availability, or the ORICO mount."
        ),
    }


def evaluate_runtime_process_contract(
    processes: dict[str, list[dict]],
    *,
    stage: str,
    heartbeat_age_seconds: float | None,
) -> dict:
    required_counts = {
        "pipeline": 1,
        "watchdog": 1,
        "progress": 1,
        "prewarm": 1 if stage == "droid_full_download" else 0,
        "downloader": 1 if stage == "droid_full_download" else 0,
        "handoff": 1 if stage == "droid_full_download" else 0,
        "transport_guard": 1,
    }
    count_gate_passed = all(
        len(processes.get(name, [])) == expected
        for name, expected in required_counts.items()
    )
    heartbeat_gate_passed = (
        stage != "droid_full_download"
        or (
            heartbeat_age_seconds is not None
            and heartbeat_age_seconds <= 300
        )
    )
    handoff_binding_passed = (
        stage != "droid_full_download"
        or (
            len(processes.get("pipeline", [])) == 1
            and len(processes.get("handoff", [])) == 1
            and processes["handoff"][0].get("target_pipeline_pid")
            == processes["pipeline"][0].get("pid")
        )
    )
    return {
        "required_counts": required_counts,
        "count_gate_passed": count_gate_passed,
        "heartbeat_gate_passed": heartbeat_gate_passed,
        "handoff_binding_passed": handoff_binding_passed,
        "passed": (
            count_gate_passed
            and heartbeat_gate_passed
            and handoff_binding_passed
        ),
    }


def command_invokes_script(command: str, script: Path) -> bool:
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    return bool(len(arguments) >= 2 and Path(arguments[1]) == script)


def power_policy_snapshot() -> dict:
    custom = subprocess.run(
        ["/usr/bin/pmset", "-g", "custom"],
        text=True,
        capture_output=True,
        check=False,
    )
    assertions = subprocess.run(
        ["/usr/bin/pmset", "-g", "assertions"],
        text=True,
        capture_output=True,
        check=False,
    )
    ac_values: dict[str, int] = {}
    in_ac_power = False
    for raw_line in custom.stdout.splitlines():
        line = raw_line.strip()
        if line.endswith("Power:"):
            in_ac_power = line == "AC Power:"
            continue
        if not in_ac_power:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] in {
            "sleep",
            "disksleep",
            "displaysleep",
        }:
            try:
                ac_values[parts[0]] = int(parts[1])
            except ValueError:
                continue

    assertion_values: dict[str, int] = {}
    in_assertion_status = False
    for raw_line in assertions.stdout.splitlines():
        line = raw_line.strip()
        if line == "Assertion status system-wide:":
            in_assertion_status = True
            continue
        if line.startswith("Listed by owning process:"):
            in_assertion_status = False
        if not in_assertion_status:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                assertion_values[parts[0]] = int(parts[1])
            except ValueError:
                continue

    sleep_disabled = ac_values.get("sleep") == 0
    disk_sleep_disabled = ac_values.get("disksleep") == 0
    external_media_asserted = assertion_values.get("ExternalMedia") == 1
    passed = bool(
        custom.returncode == 0
        and assertions.returncode == 0
        and sleep_disabled
        and disk_sleep_disabled
        and external_media_asserted
    )
    return {
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "ac_power": {
            "sleep_minutes": ac_values.get("sleep"),
            "disk_sleep_minutes": ac_values.get("disksleep"),
            "display_sleep_minutes": ac_values.get("displaysleep"),
        },
        "external_media_asserted": external_media_asserted,
        "pmset_custom_returncode": custom.returncode,
        "pmset_assertions_returncode": assertions.returncode,
        "claim_boundary": (
            "This is a live macOS sleep-policy and external-media assertion "
            "check. It does not guarantee utility power, USB cable integrity, "
            "or future mount availability."
        ),
    }


def process_snapshot(
    repo_root: Path,
    job_root: Path,
    parallel_status: dict,
    stage: str,
    prewarm_heartbeat: dict,
) -> dict:
    proc = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "command": parts[2],
            }
        )

    commands = {
        "pipeline": f"/bin/zsh {repo_root / 'scripts/qtail_orico_full_pipeline.sh'}",
        "watchdog": f"/bin/zsh {repo_root / 'scripts/qtail_droid_pipeline_watchdog.sh'}",
        "progress": f"/bin/zsh {repo_root / 'scripts/qtail_droid_progress_loop.sh'}",
        "prewarm": f"/bin/zsh {repo_root / 'scripts/qtail_droid_feature_prewarm_loop.sh'}",
    }
    processes = {
        name: [
            {"pid": row["pid"], "ppid": row["ppid"]}
            for row in rows
            if row["command"] == command
        ]
        for name, command in commands.items()
    }
    downloader_script = repo_root / "tools/qtail_parallel_gcs_download.py"
    processes["downloader"] = [
        {"pid": row["pid"], "ppid": row["ppid"]}
        for row in rows
        if command_invokes_script(row["command"], downloader_script)
    ]
    guard_script = repo_root / "tools/qtail_uniclash_transport_guard.py"
    processes["transport_guard"] = [
        {"pid": row["pid"], "ppid": row["ppid"]}
        for row in rows
        if command_invokes_script(row["command"], guard_script)
    ]
    handoff_prefix = (
        f"/bin/zsh "
        f"{repo_root / 'scripts/qtail_reload_pipeline_after_download.sh'} "
    )
    processes["handoff"] = []
    for row in rows:
        if not row["command"].startswith(handoff_prefix):
            continue
        target_text = row["command"][len(handoff_prefix) :].strip()
        processes["handoff"].append(
            {
                "pid": row["pid"],
                "ppid": row["ppid"],
                "target_pipeline_pid": (
                    int(target_text) if target_text.isdigit() else None
                ),
            }
        )

    heartbeat_age_seconds = None
    heartbeat = parallel_status.get("generated_at")
    if heartbeat:
        try:
            heartbeat_age_seconds = max(
                0.0,
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(heartbeat))
                ).total_seconds(),
            )
        except ValueError:
            heartbeat_age_seconds = None
    process_contract = evaluate_runtime_process_contract(
        processes,
        stage=stage,
        heartbeat_age_seconds=heartbeat_age_seconds,
    )
    pipeline_script = repo_root / "scripts/qtail_orico_full_pipeline.sh"
    pipeline_marker_path = job_root / "manifests/PIPELINE_STARTED"
    pipeline_marker = read_json(pipeline_marker_path)
    pipeline_sha256 = file_sha256(pipeline_script)
    pipeline_pids = processes.get("pipeline", [])
    unique_pipeline_pid = (
        int(pipeline_pids[0]["pid"]) if len(pipeline_pids) == 1 else None
    )
    lock_path = job_root / "manifests/pipeline.lock"
    lock_owner_pid = None
    try:
        lock_owner_pid = int(str(lock_path.readlink()))
    except (OSError, ValueError):
        pass
    marker_pid = None
    try:
        marker_pid = int(pipeline_marker.get("pid"))
    except (TypeError, ValueError):
        pass
    marker_lock_owner_pid = None
    try:
        marker_lock_owner_pid = int(
            pipeline_marker.get("lock_owner_pid")
        )
    except (TypeError, ValueError):
        pass
    generation_checks = {
        "semantic_marker": (
            pipeline_marker.get("format_version")
            == "qtail_pipeline_started_marker_v2"
        ),
        "running_status": pipeline_marker.get("status") == "running",
        "unique_pipeline": unique_pipeline_pid is not None,
        "marker_pid_matches_process": (
            marker_pid is not None
            and marker_pid == unique_pipeline_pid
        ),
        "marker_script_matches": (
            pipeline_marker.get("script") == str(pipeline_script)
        ),
        "marker_job_root_matches": (
            pipeline_marker.get("job_root") == str(job_root)
        ),
        "marker_sha_matches_current_source": (
            bool(pipeline_sha256)
            and pipeline_marker.get("script_sha256") == pipeline_sha256
        ),
        "live_lock_owner_matches_process": (
            lock_owner_pid is not None
            and lock_owner_pid == unique_pipeline_pid
        ),
        "marker_lock_owner_matches_process": (
            marker_lock_owner_pid is not None
            and marker_lock_owner_pid == unique_pipeline_pid
        ),
    }
    generation_hash_matched = all(generation_checks.values())
    legacy_marker = (
        pipeline_marker.get("format_version")
        != "qtail_pipeline_started_marker_v2"
    )
    legacy_handoff_pending = bool(
        stage == "droid_full_download"
        and legacy_marker
        and unique_pipeline_pid is not None
        and process_contract["handoff_binding_passed"]
    )
    if generation_hash_matched:
        generation_status = "hash_matched"
    elif legacy_handoff_pending:
        generation_status = "legacy_handoff_pending"
    else:
        generation_status = "blocked"
    pipeline_generation = {
        "status": generation_status,
        "passed": generation_hash_matched or legacy_handoff_pending,
        "hash_matched": generation_hash_matched,
        "legacy_handoff_pending": legacy_handoff_pending,
        "download_only_exception": legacy_handoff_pending,
        "marker_path": str(pipeline_marker_path),
        "marker_format_version": pipeline_marker.get("format_version"),
        "marker_pid": marker_pid,
        "process_pid": unique_pipeline_pid,
        "lock_owner_pid": lock_owner_pid,
        "marker_script_sha256": pipeline_marker.get("script_sha256"),
        "current_script_sha256": pipeline_sha256,
        "checks": generation_checks,
        "claim_boundary": (
            "HASH MATCH requires the semantic start marker, unique live "
            "pipeline PID, pipeline lock owner, and current script SHA-256 "
            "to agree. The legacy-marker exception is download-only and "
            "requires a handoff process bound to the unique pipeline PID; "
            "checksum verification and formal training require HASH MATCH."
        ),
    }
    prewarm_heartbeat_age_seconds = timestamp_age_seconds(
        prewarm_heartbeat.get("generated_at")
    )
    prewarm_pids = processes.get("prewarm", [])
    prewarm_heartbeat_pid_matches = bool(
        len(prewarm_pids) == 1
        and int(prewarm_heartbeat.get("pid") or -1)
        == int(prewarm_pids[0]["pid"])
    )
    prewarm_heartbeat_gate_passed = bool(
        stage != "droid_full_download"
        or (
            prewarm_heartbeat.get("control")
            == "droid_feature_prewarm_pid_heartbeat_v1"
            and prewarm_heartbeat.get("status") == "alive"
            and prewarm_heartbeat_age_seconds is not None
            and prewarm_heartbeat_age_seconds <= 150
            and prewarm_heartbeat_pid_matches
        )
    )
    required_mount = Path("/Volumes/ORICO")
    mount_gate_passed = os.path.ismount(required_mount)
    web_services = [
        web_service_snapshot(54655),
        web_service_snapshot(6222),
    ]
    web_gate_passed = all(service["healthy"] for service in web_services)
    launchd_supervision = launchd_supervision_snapshot(repo_root)
    power_policy = power_policy_snapshot()
    code_paths = {
        "pipeline": repo_root / "scripts/qtail_orico_full_pipeline.sh",
        "checksum_manifest_builder": (
            repo_root / "tools/qtail_build_droid_checksum_manifest.py"
        ),
        "release_metadata_auditor": (
            repo_root / "tools/qtail_audit_droid_release_metadata.py"
        ),
        "downloader": repo_root / "tools/qtail_parallel_gcs_download.py",
        "download_marker_verifier": (
            repo_root / "tools/qtail_verify_droid_download_marker.py"
        ),
        "download_marker_selftest": (
            repo_root / "tools/qtail_droid_download_marker_selftest.py"
        ),
        "mirror_verifier_selftest": (
            repo_root / "tools/qtail_droid_mirror_verifier_selftest.py"
        ),
        "downloader_single_writer_selftest": (
            repo_root / "tools/qtail_downloader_single_writer_selftest.py"
        ),
        "runtime_process_contract_selftest": (
            repo_root / "tools/qtail_runtime_process_contract_selftest.py"
        ),
        "uniclash_pre_checksum_gate": (
            repo_root / "tools/qtail_assert_uniclash_transport_gate.py"
        ),
        "uniclash_pre_checksum_gate_selftest": (
            repo_root / "tools/qtail_uniclash_transport_gate_selftest.py"
        ),
        "live_partial_marker_rejection_capture": (
            repo_root
            / "tools/qtail_capture_droid_partial_marker_rejection.py"
        ),
        "transport_cleanup": (
            repo_root / "tools/qtail_cleanup_droid_transport_artifacts.py"
        ),
        "environment_capture": (
            repo_root / "tools/qtail_capture_droid_environment.py"
        ),
        "mirror_verifier": repo_root / "tools/qtail_verify_droid_mirror.py",
        "cache_verifier": repo_root / "tools/qtail_verify_droid_feature_cache.py",
        "incremental_closure_auditor": (
            repo_root / "tools/qtail_audit_droid_incremental_closure.py"
        ),
        "incremental_closure_selftest": (
            repo_root / "tools/qtail_droid_incremental_closure_selftest.py"
        ),
        "artifact_manifest_merger": (
            repo_root / "tools/qtail_merge_droid_artifact_manifest.py"
        ),
        "release_milestone_sealer": (
            repo_root / "tools/qtail_seal_droid_release_milestones.py"
        ),
        "protocol_selftest": repo_root / "tools/qtail_droid_protocol_selftest.py",
        "trainer": repo_root / "tools/qtail_train_droid_full.py",
        "progress": repo_root / "tools/qtail_droid_full_progress.py",
        "stage_marker_verifier": (
            repo_root / "tools/qtail_verify_droid_stage_markers.py"
        ),
        "timeline_verifier": (
            repo_root / "tools/qtail_verify_droid_timeline.py"
        ),
        "timeline_monotonic_selftest": (
            repo_root
            / "tools"
            / "qtail_droid_timeline_monotonic_selftest.py"
        ),
        "forecast_summarizer": (
            repo_root / "tools/qtail_summarize_droid_forecast.py"
        ),
        "web_services": repo_root / "scripts/qtail_web_services.sh",
        "feature_prewarm": repo_root / "scripts/qtail_droid_feature_prewarm_loop.sh",
        "pipeline_watchdog": (
            repo_root / "scripts/qtail_droid_pipeline_watchdog.sh"
        ),
        "generation_handoff": (
            repo_root / "scripts/qtail_reload_pipeline_after_download.sh"
        ),
        "page_qa": repo_root / "tools/qtail_verify_droid_page.mjs",
        "page": repo_root / "qtail-droid-full-training.html",
    }
    return {
        "healthy": (
            process_contract["passed"]
            and pipeline_generation["passed"]
            and prewarm_heartbeat_gate_passed
            and mount_gate_passed
            and web_gate_passed
            and launchd_supervision["passed"]
            and power_policy["passed"]
        ),
        "stage": stage,
        "required_process_counts": process_contract["required_counts"],
        "processes": processes,
        "download_heartbeat": heartbeat,
        "download_heartbeat_age_seconds": heartbeat_age_seconds,
        "process_count_gate_passed": process_contract[
            "count_gate_passed"
        ],
        "heartbeat_gate_passed": process_contract[
            "heartbeat_gate_passed"
        ],
        "handoff_binding_passed": process_contract[
            "handoff_binding_passed"
        ],
        "pipeline_generation": pipeline_generation,
        "prewarm_heartbeat": {
            **prewarm_heartbeat,
            "age_seconds": prewarm_heartbeat_age_seconds,
            "pid_matches_unique_process": prewarm_heartbeat_pid_matches,
            "gate_passed": prewarm_heartbeat_gate_passed,
        },
        "required_mount": str(required_mount),
        "mount_gate_passed": mount_gate_passed,
        "web_services": web_services,
        "web_gate_passed": web_gate_passed,
        "launchd_supervision": launchd_supervision,
        "power_policy": power_policy,
        "code_sha256": {
            name: file_sha256(path)
            for name, path in code_paths.items()
        },
    }


def checksum_ledger_summary(
    *,
    data_root: Path,
    checksum_manifest: dict,
    checksum_ledger: dict,
) -> dict:
    expected = {
        str(item["relative_path"]): item
        for item in checksum_manifest.get("objects", [])
        if isinstance(item, dict) and item.get("relative_path")
    }
    entries = checksum_ledger.get("objects", {})
    if not isinstance(entries, dict):
        entries = {}
    valid_paths = []
    verified_bytes = 0
    stale_paths = []
    for relative, entry in entries.items():
        checksum = expected.get(relative)
        target = data_root / relative
        if not checksum or not isinstance(entry, dict):
            stale_paths.append(relative)
            continue
        try:
            stat = target.stat()
        except OSError:
            stale_paths.append(relative)
            continue
        valid = (
            stat.st_size == int(checksum.get("bytes", -1))
            and int(entry.get("bytes", -1)) == stat.st_size
            and int(entry.get("mtime_ns", -1)) == stat.st_mtime_ns
            and int(entry.get("ctime_ns", -1)) == stat.st_ctime_ns
            and entry.get("official_md5_base64")
            == checksum.get("md5_base64")
            and entry.get("local_md5_base64")
            == checksum.get("md5_base64")
            and entry.get("generation") == checksum.get("generation")
        )
        if valid:
            valid_paths.append(relative)
            verified_bytes += stat.st_size
        else:
            stale_paths.append(relative)
    expected_count = len(expected)
    manifest_valid = (
        checksum_manifest.get("status") == "verified"
        and expected_count == 4_102
        and int(checksum_manifest.get("total_bytes", -1)) == REMOTE_BYTES
        and all(
            item.get("md5_base64") and item.get("crc32c_base64")
            for item in expected.values()
        )
    )
    complete = (
        manifest_valid
        and len(valid_paths) == expected_count
        and not stale_paths
        and set(valid_paths) == set(expected)
    )
    return {
        "status": "complete" if complete else "in_progress",
        "manifest_valid": manifest_valid,
        "expected_objects": expected_count,
        "verified_objects": len(valid_paths),
        "verified_bytes": verified_bytes,
        "verified_percent": (
            len(valid_paths) / expected_count * 100.0
            if expected_count
            else 0.0
        ),
        "stale_or_unexpected_entries": len(stale_paths),
        "stale_or_unexpected_sample": sorted(stale_paths)[:20],
        "ledger_generated_at": checksum_ledger.get("generated_at"),
    }


def capacity_headroom_summary(
    *,
    data_root: Path,
    checksum_manifest: dict,
    checksum_summary: dict,
    free_bytes: int,
) -> dict:
    expected = {
        str(item["relative_path"]): int(item["bytes"])
        for item in checksum_manifest.get("objects", [])
        if isinstance(item, dict)
        and item.get("relative_path")
        and item.get("bytes") is not None
    }
    resumable_partial_objects = 0
    resumable_partial_logical_bytes = 0
    resumable_partial_allocated_bytes = 0
    invalid_partial_objects = 0
    for relative, expected_bytes in expected.items():
        target = data_root / relative
        try:
            target_bytes = target.stat().st_size
        except OSError:
            target_bytes = 0
        if target_bytes == expected_bytes:
            continue
        part = target.with_name(target.name + ".qtail.part")
        try:
            stat = part.stat()
        except OSError:
            continue
        if stat.st_size <= 0 or stat.st_size > expected_bytes:
            invalid_partial_objects += 1
            continue
        allocated = max(0, int(getattr(stat, "st_blocks", 0)) * 512)
        resumable_partial_objects += 1
        resumable_partial_logical_bytes += stat.st_size
        resumable_partial_allocated_bytes += min(stat.st_size, allocated)

    verified_bytes = min(
        REMOTE_BYTES,
        max(0, int(checksum_summary.get("verified_bytes", 0))),
    )
    trusted_reusable_bytes = min(
        REMOTE_BYTES,
        verified_bytes + resumable_partial_allocated_bytes,
    )
    remaining_official_bytes = max(0, REMOTE_BYTES - trusted_reusable_bytes)
    safety_reserve_bytes = (REMOTE_BYTES + 19) // 20
    required_free_bytes = remaining_official_bytes + safety_reserve_bytes
    headroom_bytes = int(free_bytes) - required_free_bytes
    return {
        "capacity_model_version": (
            "official_md5_plus_allocated_resumable_parts_v2"
        ),
        "verified_official_bytes": verified_bytes,
        "resumable_partial_objects": resumable_partial_objects,
        "resumable_partial_logical_bytes": (
            resumable_partial_logical_bytes
        ),
        "resumable_partial_allocated_bytes": (
            resumable_partial_allocated_bytes
        ),
        "invalid_partial_objects": invalid_partial_objects,
        "trusted_reusable_bytes": trusted_reusable_bytes,
        "remaining_official_bytes": remaining_official_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "required_free_bytes": required_free_bytes,
        "headroom_bytes": headroom_bytes,
        "capacity_gate_passed": headroom_bytes >= 0,
        "claim_boundary": (
            "This is a storage-allocation gate, not an integrity gate. "
            "Completed bytes count only after official-MD5 ledger validation; "
            "resumable parts count only their currently allocated filesystem "
            "blocks and still require final official-MD5 verification."
        ),
    }


def build_history_chart(
    timeline_path: Path,
    *,
    max_points: int = 240,
) -> dict:
    timeline = read_json(timeline_path)
    samples = timeline.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    source_count = len(samples)
    if source_count == 0:
        selected_indices: list[int] = []
    elif source_count <= max_points:
        selected_indices = list(range(source_count))
    else:
        mandatory = {0, source_count - 1}
        previous_token: tuple[object, object] | None = None
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            token = (sample.get("kind"), sample.get("stage"))
            if token != previous_token:
                mandatory.add(index)
                if index > 0:
                    mandatory.add(index - 1)
                previous_token = token
        uniform = {
            round(position * (source_count - 1) / (max_points - 1))
            for position in range(max_points)
        }
        selected = mandatory | uniform
        if len(selected) > max_points:
            mandatory_ordered = sorted(mandatory)
            if len(mandatory_ordered) >= max_points:
                selected = {
                    mandatory_ordered[
                        round(
                            position
                            * (len(mandatory_ordered) - 1)
                            / (max_points - 1)
                        )
                    ]
                    for position in range(max_points)
                }
            else:
                optional = sorted(selected - mandatory)
                optional_budget = max_points - len(mandatory)
                if len(optional) > optional_budget:
                    optional = [
                        optional[
                            round(
                                position
                                * (len(optional) - 1)
                                / max(1, optional_budget - 1)
                            )
                        ]
                        for position in range(optional_budget)
                    ]
                selected = mandatory | set(optional)
        selected_indices = sorted(selected)

    points = []
    for index in selected_indices:
        sample = samples[index]
        if not isinstance(sample, dict):
            continue
        download = sample.get("download", {})
        checksums = sample.get("object_checksums", {})
        features = sample.get("feature_extraction", {})
        storage = sample.get("external_storage", {})
        physical_bytes = int(download.get("physical_bytes", 0) or 0)
        download_percent = download.get("percent")
        if download_percent is None:
            download_percent = (
                physical_bytes / REMOTE_BYTES * 100.0
                if REMOTE_BYTES
                else 0.0
            )
        verified_objects = checksums.get("verified_objects")
        processed_shards = features.get("processed_shards")
        records_decoded = features.get("records_decoded")
        points.append(
            {
                "sequence": sample.get("sequence"),
                "generated_at": sample.get("generated_at"),
                "kind": sample.get("kind"),
                "stage": sample.get("stage"),
                "download_percent": float(download_percent),
                "md5_percent": (
                    int(verified_objects) / 4_102 * 100.0
                    if verified_objects is not None
                    else None
                ),
                "preparsed_shard_percent": (
                    int(processed_shards) / 4_096 * 100.0
                    if processed_shards is not None
                    else None
                ),
                "record_percent": (
                    int(records_decoded) / 187_891 * 100.0
                    if records_decoded is not None
                    else None
                ),
                "throughput_mib_per_second": (
                    float(download.get("throughput_bytes_per_second", 0) or 0)
                    / (1024**2)
                ),
                "capacity_headroom_tib": (
                    float(storage["headroom_bytes"]) / (1024**4)
                    if storage.get("headroom_bytes") is not None
                    else None
                ),
            }
        )
    return {
        "version": "qtail_droid_bounded_history_chart_v1",
        "source": str(timeline_path),
        "source_timeline_version": timeline.get("version"),
        "source_sample_count": source_count,
        "point_count": len(points),
        "max_points": max_points,
        "sampling": "uniform_index_plus_stage_boundaries_v1",
        "series": {
            "download_percent": "physical committed bytes / 3,700,745,265,151",
            "md5_percent": "official-MD5 ledger objects / 4,102",
            "preparsed_shard_percent": "full-record scanned TFRecords / 4,096",
            "record_percent": "decoded records / 187,891",
        },
        "claim_boundary": (
            "The chart is a bounded rendering view of the retained hash-chained "
            "timeline. Missing historical series stay null; points are not "
            "interpolated and the timeline remains the authoritative source."
        ),
        "points": points,
    }


def release_metadata_audit_summary(payload: dict) -> dict:
    gates = payload.get("gates", {})
    combined = payload.get("combined_official_metadata", {})
    releases = {
        str(item.get("release")): item
        for item in payload.get("releases", [])
        if isinstance(item, dict)
    }
    errors = []
    if payload.get("version") != "droid_release_metadata_audit_v1":
        errors.append("unexpected audit version")
    if payload.get("status") != "verified":
        errors.append("audit status is not verified")
    if (
        not isinstance(gates, dict)
        or set(gates) != REQUIRED_METADATA_GATES
        or not all(value is True for value in gates.values())
    ):
        errors.append("metadata audit gates are incomplete")
    if (
        int(combined.get("tfrecord_shards", -1)) != 4_096
        or int(combined.get("records", -1)) != 187_891
        or int(combined.get("split_bytes", -1)) != 3_700_742_144_299
    ):
        errors.append("combined official metadata is not exact")
    if set(releases) != set(EXPECTED_RELEASE_METADATA):
        errors.append("release set is not exactly 1.0.0 and 1.0.1")
    for release, expected in EXPECTED_RELEASE_METADATA.items():
        item = releases.get(release, {})
        if (
            item.get("verified") is not True
            or item.get("dataset_name") != expected["dataset_name"]
            or item.get("dataset_version") != expected["dataset_version"]
            or int(item.get("official_tfrecord_shards", -1))
            != expected["shards"]
            or int(item.get("official_records", -1))
            != expected["records"]
            or int(item.get("official_split_bytes", -1))
            != expected["split_bytes"]
            or set(item.get("step_feature_keys", []))
            != EXPECTED_DROID_STEP_SCHEMA
            or item.get("required_training_features_present") is not True
            or item.get("dataset_info_file", {}).get("verified") is not True
            or item.get("features_file", {}).get("verified") is not True
        ):
            errors.append(f"release metadata mismatch: {release}")
    return {
        "valid": not errors,
        "version": payload.get("version"),
        "status": payload.get("status", "missing"),
        "source": payload.get("source"),
        "gates": gates,
        "combined_official_metadata": combined,
        "releases": [
            {
                "release": release,
                "dataset_name": releases.get(release, {}).get(
                    "dataset_name"
                ),
                "dataset_version": releases.get(release, {}).get(
                    "dataset_version"
                ),
                "official_tfrecord_shards": releases.get(release, {}).get(
                    "official_tfrecord_shards"
                ),
                "official_records": releases.get(release, {}).get(
                    "official_records"
                ),
                "official_split_bytes": releases.get(release, {}).get(
                    "official_split_bytes"
                ),
                "verified": releases.get(release, {}).get("verified"),
            }
            for release in EXPECTED_RELEASE_METADATA
        ],
        "claim_boundary": payload.get("claim_boundary"),
        "errors": errors,
    }


def build_completion_audit(
    *,
    generated_at: str,
    openx: dict,
    marker_state: dict,
    source_probe: dict,
    source_probe_marker: dict,
    pipeline_generation_gate: dict,
    object_manifest: dict,
    checksum_manifest: dict,
    checksum_summary: dict,
    release_metadata_audit: dict,
    verification: dict,
    feature_status: dict,
    training_report: dict,
    artifact_manifest: dict,
    artifact_integrity: dict,
    environment_manifest: dict,
    environment_selftest: dict,
    download_marker_selftest: dict,
    mirror_verifier_selftest: dict,
    training_gate_order_selftest: dict,
    downloader_single_writer_selftest: dict,
    runtime_process_contract_selftest: dict,
    uniclash_pre_checksum_gate: dict,
    uniclash_pre_checksum_gate_selftest: dict,
    live_partial_marker_rejection: dict,
    transport_guard: dict,
    transport_adjudication: dict,
    cache_verification: dict,
    runtime: dict,
    backend: dict,
    expected_backend_commit: str | None,
    code_snapshot: dict,
    snapshot_publish_audit: dict,
    result_root: Path,
) -> dict:
    generation_gate_summary = pipeline_generation_gate_summary(
        pipeline_generation_gate,
        gate_path=result_root / "pipeline_generation_gate.json",
        script_path=(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "qtail_orico_full_pipeline.sh"
        ),
    )
    trajectory = training_report.get("trajectory_evidence", {})
    compute = training_report.get("compute_audit", {})
    input_audit = training_report.get("input_audit", {})
    effect = training_report.get("effect_metrics", {})
    bootstrap = effect.get("paired_bootstrap", {})
    randomization = effect.get("paired_arm_swap_randomization", {})
    hypothesis_gate = effect.get("hypothesis_gate", {})
    holdout = training_report.get("holdout_evaluation", {})
    tail_contract = training_report.get("tail_score_contract", {})
    pt_source = training_report.get("pt_source_audit", {})
    rare_coverage = training_report.get(
        "rare_instruction_fingerprint_coverage",
        {},
    )
    formal_protocol = training_report.get("formal_protocol", {})
    download_marker_controls = download_marker_selftest.get("controls", [])
    download_marker_selftest_valid = bool(
        download_marker_selftest.get("status") == "passed"
        and int(download_marker_selftest.get("controls_passed", -1)) == 8
        and int(download_marker_selftest.get("controls_total", -1)) == 8
        and len(download_marker_controls) == 8
        and all(
            isinstance(control, dict) and control.get("passed") is True
            for control in download_marker_controls
        )
    )
    mirror_verifier_controls = mirror_verifier_selftest.get("controls", [])
    mirror_verifier_selftest_valid = bool(
        mirror_verifier_selftest.get("status") == "passed"
        and int(mirror_verifier_selftest.get("controls_passed", -1)) == 8
        and int(mirror_verifier_selftest.get("controls_total", -1)) == 8
        and len(mirror_verifier_controls) == 8
        and all(
            isinstance(control, dict) and control.get("passed") is True
            for control in mirror_verifier_controls
        )
    )
    training_gate_order_controls = training_gate_order_selftest.get(
        "controls", []
    )
    training_gate_order_selftest_valid = bool(
        training_gate_order_selftest.get("version")
        == "qtail_droid_training_gate_order_selftest_v2"
        and training_gate_order_selftest.get("status") == "passed"
        and int(
            training_gate_order_selftest.get("controls_passed", -1)
        )
        == 11
        and int(
            training_gate_order_selftest.get("controls_total", -1)
        )
        == 11
        and len(training_gate_order_controls) == 11
        and all(
            isinstance(control, dict) and control.get("passed") is True
            for control in training_gate_order_controls
        )
    )
    downloader_single_writer_checks = (
        downloader_single_writer_selftest.get("checks", {})
    )
    downloader_single_writer_selftest_valid = bool(
        downloader_single_writer_selftest.get("status") == "passed"
        and int(
            downloader_single_writer_selftest.get("checks_passed", -1)
        )
        == 13
        and int(
            downloader_single_writer_selftest.get("checks_total", -1)
        )
        == 13
        and len(downloader_single_writer_checks) == 13
        and all(
            value is True
            for value in downloader_single_writer_checks.values()
        )
    )
    runtime_process_checks = runtime_process_contract_selftest.get(
        "checks", {}
    )
    runtime_process_contract_selftest_valid = bool(
        runtime_process_contract_selftest.get("status") == "passed"
        and runtime_process_contract_selftest.get("control")
        == "droid_runtime_process_contract_v11"
        and int(
            runtime_process_contract_selftest.get("checks_passed", -1)
        )
        == 16
        and int(
            runtime_process_contract_selftest.get("checks_total", -1)
        )
        == 16
        and len(runtime_process_checks) == 16
        and all(value is True for value in runtime_process_checks.values())
    )
    uniclash_gate_checks = uniclash_pre_checksum_gate.get("checks", {})
    uniclash_pre_checksum_gate_valid = bool(
        uniclash_pre_checksum_gate.get("status") == "passed"
        and int(uniclash_pre_checksum_gate.get("checks_passed", -1)) == 10
        and int(uniclash_pre_checksum_gate.get("checks_total", -1)) == 10
        and len(uniclash_gate_checks) == 10
        and all(value is True for value in uniclash_gate_checks.values())
    )
    uniclash_gate_selftest_checks = (
        uniclash_pre_checksum_gate_selftest.get("checks", {})
    )
    uniclash_pre_checksum_gate_selftest_valid = bool(
        uniclash_pre_checksum_gate_selftest.get("status") == "passed"
        and int(
            uniclash_pre_checksum_gate_selftest.get("checks_passed", -1)
        )
        == 13
        and int(
            uniclash_pre_checksum_gate_selftest.get("checks_total", -1)
        )
        == 13
        and len(uniclash_gate_selftest_checks) == 13
        and all(
            value is True
            for value in uniclash_gate_selftest_checks.values()
        )
    )
    live_partial_marker_rejection_valid = bool(
        not live_partial_marker_rejection
        or (
            live_partial_marker_rejection.get("status") == "passed"
            and live_partial_marker_rejection.get(
                "formal_completion_evidence"
            )
            is False
            and live_partial_marker_rejection.get("precondition", {}).get(
                "passed"
            )
            is True
            and live_partial_marker_rejection.get("result", {}).get(
                "rejected"
            )
            is True
            and live_partial_marker_rejection.get("result", {}).get(
                "marker_created"
            )
            is False
        )
    )
    metadata_audit_summary = release_metadata_audit_summary(
        release_metadata_audit
    )
    steps = int(training_report.get("steps", -1))
    formal_protocol_valid = bool(
        formal_protocol.get("locked") is True
        and int(formal_protocol.get("seed", -1)) == FORMAL_SEED
        and int(formal_protocol.get("steps_per_stage", -1)) == 20_000
        and float(formal_protocol.get("holdout_fraction", -1.0))
        == FORMAL_HOLDOUT_FRACTION
        and int(formal_protocol.get("holdout_shards_per_release", -1))
        == FORMAL_HOLDOUT_SHARDS_PER_RELEASE
        and formal_protocol.get("holdout_relative_path_sha256")
        == FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
        and formal_protocol.get("holdout_membership_path_scope")
        == "official_release_relative_path"
        and int(formal_protocol.get("bootstrap_samples", -1)) == 5_000
        and int(formal_protocol.get("randomization_samples", -1)) == 5_000
        and int(formal_protocol.get("checkpoint_every_steps", -1)) == 5_000
        and float(formal_protocol.get("min_record_parse_rate", -1.0))
        == 1.0
        and float(
            formal_protocol.get(
                "min_record_scan_complete_rate", -1.0
            )
        )
        == 1.0
        and formal_protocol.get("require_verified_mirror") is True
        and formal_protocol.get("pt_source_sha256")
        == FORMAL_PT_SOURCE_SHA256
        and int(training_report.get("seed", -1)) == FORMAL_SEED
    )
    release_composition = training_report.get("release_composition", [])
    expected_releases = {
        "1.0.0": {
            "dataset": "r2d2_faceblur",
            "shards": 2_048,
            "records": 92_233,
        },
        "1.0.1": {
            "dataset": "droid_101",
            "shards": 2_048,
            "records": 95_658,
        },
    }
    artifact_paths = {
        str(item.get("path"))
        for item in artifact_manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    environment_manifest_path = str(
        result_root / "droid_environment_manifest.json"
    )
    environment_selftest_path = str(
        result_root / "droid_environment_contract_selftest.json"
    )
    live_artifact_paths = {
        str(result_root / "uniclash_transport_guard.json"),
        str(result_root / "download_progress_samples.json"),
        str(result_root / "live_logs" / "droid_full_pipeline.log"),
        str(result_root / "live_logs" / "droid_feature_prewarm.log"),
        str(result_root / "live_logs" / "pipeline_watchdog.log"),
        str(result_root / "live_logs" / "qtail-web-services.log"),
        str(
            result_root
            / "live_logs"
            / "qtail_droid_terminal_launcher.log"
        ),
        str(
            result_root / "live_logs" / "qtail_droid_launchd_stderr.log"
        ),
        str(
            result_root / "live_logs" / "qtail_droid_launchd_stdout.log"
        ),
        str(
            result_root / "live_logs" / "qtail_uniclash_guard_stderr.log"
        ),
        str(
            result_root / "live_logs" / "qtail_uniclash_guard_stdout.log"
        ),
        str(
            result_root / "live_logs" / "qtail_web_services_local.log"
        ),
        str(
            result_root.parent.parent
            / "logs"
            / "droid_full_pipeline.log"
        ),
        str(
            result_root.parent.parent
            / "logs"
            / "droid_feature_prewarm.log"
        ),
        str(
            result_root.parent.parent
            / "logs"
            / "pipeline_watchdog.log"
        ),
        str(
            result_root.parent.parent
            / "logs"
            / "qtail-web-services.log"
        ),
    }
    immutable_final_artifact_paths = {
        str(result_root / "uniclash_transport_guard_final.json"),
        str(result_root / "download_progress_samples_final.json"),
    }
    immutable_final_artifact_contract = bool(
        not marker_state.get("final_page_qa_effective", False)
        or (
            immutable_final_artifact_paths.issubset(artifact_paths)
            and artifact_paths.isdisjoint(live_artifact_paths)
        )
    )
    final_guard_snapshot = read_json(
        result_root / "uniclash_transport_guard_final.json"
    )
    final_guard_cumulative = final_guard_snapshot.get("cumulative", {})
    immutable_final_guard_valid = bool(
        not marker_state.get("final_page_qa_effective", False)
        or (
            final_guard_snapshot.get("status") in {"passed", "passed_idle"}
            and final_guard_snapshot.get("policy", {}).get(
                "uniclash_core_must_continue"
            )
            is True
            and final_guard_snapshot.get("uniclash", {}).get("core_running")
            is True
            and final_guard_snapshot.get("uniclash", {}).get("tun_enabled")
            is False
            and int(
                final_guard_cumulative.get(
                    "forbidden_socket_observations",
                    -1,
                )
            )
            == 0
            and int(
                final_guard_cumulative.get("wrong_route_observations", -1)
            )
            == 0
            and not final_guard_snapshot.get("global_violations", [])
        )
    )
    formal_artifact_requirements = formal_artifact_requirement_paths(
        result_root,
        marker_state,
    )
    formal_pre_page_artifact_paths = formal_artifact_requirements["baseline"]
    required_artifact_paths = formal_artifact_requirements["required"]
    required_process_log_artifact_paths = formal_artifact_requirements[
        "process_logs"
    ]
    process_log_gate_required = marker_state.get(
        "final_page_qa_effective", False
    )
    missing_required_artifacts = sorted(
        required_artifact_paths - artifact_paths
    )
    physically_missing_required_artifacts = sorted(
        path
        for path in required_artifact_paths
        if not Path(path).is_file()
    )
    present_but_unsealed_required_artifacts = sorted(
        set(missing_required_artifacts)
        - set(physically_missing_required_artifacts)
    )
    process_log_manifest_path = (
        result_root / "droid_process_log_manifest.json"
    )
    process_log_manifest = read_json(process_log_manifest_path)
    process_log_entries = process_log_manifest.get("logs", [])
    process_log_contract = process_log_manifest.get("contract", {})
    process_log_manifest_valid = bool(
        process_log_manifest.get("status") == "complete"
        and not process_log_manifest.get("missing_required", [])
        and int(process_log_contract.get("required_log_count", 0)) == 8
        and int(
            process_log_contract.get("captured_required_log_count", 0)
        )
        == 8
        and isinstance(process_log_entries, list)
        and len(process_log_entries) >= 8
        and all(
            isinstance(entry, dict)
            and str(entry.get("path")) in artifact_paths
            and str(entry.get("path", "")).startswith(
                str(result_root / "process_logs_final") + "/"
            )
            and int(entry.get("bytes", -1)) >= 0
            and len(str(entry.get("sha256", ""))) == 64
            and int(entry.get("line_count", -1)) >= 0
            for entry in process_log_entries
        )
    )
    environment_gates = environment_manifest.get("gates", {})
    environment_checks = environment_selftest.get("checks", {})
    environment_manifest_valid = bool(
        environment_manifest.get("status") == "complete"
        and isinstance(environment_gates, dict)
        and environment_gates
        and all(value is True for value in environment_gates.values())
    )
    required_environment_checks = {
        "positive_control_completes",
        "one_byte_mirror_mismatch_fails",
        "orchestration_snapshot_code_drift_fails",
        "missing_official_md5_fails",
        "uniclash_violation_fails",
        "transport_classifier_v6_selftest_passes",
        "backend_commit_drift_fails",
        "backend_origin_drift_fails",
        "backend_worktree_dirty_fails",
    }
    environment_selftest_valid = bool(
        environment_selftest.get("status") == "passed"
        and environment_selftest.get("contract_version")
        == "qtail_droid_environment_contract_selftest_v3"
        and isinstance(environment_checks, dict)
        and set(environment_checks) == required_environment_checks
        and all(value is True for value in environment_checks.values())
    )
    guard_uniclash = transport_guard.get("uniclash", {})
    guard_bypass = transport_guard.get("system_proxy_bypass", {})
    guard_cumulative = transport_guard.get("cumulative", {})
    guard_age_seconds = timestamp_age_seconds(
        transport_guard.get("generated_at"),
        reference=datetime.fromisoformat(generated_at),
    )
    adjudication_required = bool(transport_adjudication)
    adjudication_valid = bool(
        not adjudication_required
        or (
            transport_adjudication.get("status")
            == "adjudicated_transport_epochs_v6"
            and len(transport_adjudication.get("findings", [])) >= 5
            and all(
                finding.get("data_transfer_violation") is False
                for finding in transport_adjudication.get("findings", [])
            )
            and any(
                finding.get("guard_epoch")
                == "droid_transport_root_environment_v3"
                and finding.get("coverage_gap") is True
                for finding in transport_adjudication.get("findings", [])
            )
            and transport_adjudication.get("remediation", {}).get(
                "classifier_version"
            )
            == "droid_transport_downloader_descendants_v6_interface_bound_live"
            and len(
                transport_adjudication.get("preservation", {}).get(
                    "archives", []
                )
            )
            >= 5
            and all(
                archive.get("sha256")
                == transport_adjudication.get(
                    "archive_hashes_actual", {}
                ).get(archive.get("path"))
                for archive in transport_adjudication.get(
                    "preservation", {}
                ).get("archives", [])
            )
            and any(
                archive.get("coverage_gap") is True
                for archive in transport_adjudication.get(
                    "preservation", {}
                ).get("archives", [])
            )
        )
    )
    transport_guard_valid = bool(
        transport_guard.get("status") in {"passed", "passed_idle"}
        and guard_age_seconds is not None
        and guard_age_seconds <= 10
        and transport_guard.get("policy", {}).get(
            "process_classifier_version"
        )
        == "droid_transport_downloader_descendants_v6_interface_bound_live"
        and guard_uniclash.get("core_running") is True
        and guard_uniclash.get("tun_enabled") is False
        and guard_bypass.get("passed") is True
        and not transport_guard.get("blocked_processes")
        and not transport_guard.get("global_violations")
        and int(guard_cumulative.get("samples", 0)) > 0
        and (
            int(guard_cumulative.get("blocked_samples", -1)) == 0
            or adjudication_valid
        )
        and int(
            guard_cumulative.get("forbidden_socket_observations", -1)
        )
        == 0
        and int(guard_cumulative.get("wrong_route_observations", -1)) == 0
        and not guard_cumulative.get("blocked_pids")
        and (
            not guard_cumulative.get("violation_events")
            or adjudication_valid
        )
        and adjudication_valid
    )

    def valid_release_composition() -> bool:
        if not isinstance(release_composition, list):
            return False
        rows = {
            str(item.get("release")): item
            for item in release_composition
            if isinstance(item, dict)
        }
        if set(rows) != set(expected_releases):
            return False
        try:
            for release, expected in expected_releases.items():
                item = rows[release]
                if (
                    item.get("official_dataset_name") != expected["dataset"]
                    or item.get("metadata_status") != "verified"
                    or int(item.get("observed_tfrecord_shards", -1))
                    != expected["shards"]
                    or int(item.get("official_tfrecord_shards", -1))
                    != expected["shards"]
                    or int(item.get("observed_records_decoded", -1))
                    != expected["records"]
                    or int(item.get("official_expected_records", -1))
                    != expected["records"]
                    or item.get("full_shard_coverage") is not True
                    or item.get("full_record_count_match") is not True
                ):
                    return False
            return (
                sum(
                    int(item.get("observed_tfrecord_shards", 0))
                    for item in rows.values()
                )
                == 4_096
                and sum(
                    int(item.get("observed_records_decoded", 0))
                    for item in rows.values()
                )
                == 187_891
                and sum(
                    int(item.get("observed_tfrecord_bytes", 0))
                    for item in rows.values()
                )
                == int(training_report.get("total_bytes", -1))
            )
        except (TypeError, ValueError):
            return False

    release_composition_valid = valid_release_composition()
    holdout_releases = {
        str(item.get("release")): item
        for item in holdout.get("per_release", [])
        if isinstance(item, dict)
    }
    holdout_relative_paths_raw = holdout.get(
        "holdout_relative_paths", []
    )
    holdout_relative_paths_typed = bool(
        isinstance(holdout_relative_paths_raw, list)
        and all(
            isinstance(value, str)
            for value in holdout_relative_paths_raw
        )
    )
    holdout_relative_paths = (
        holdout_relative_paths_raw
        if holdout_relative_paths_typed
        else []
    )
    holdout_relative_path_sha256 = hashlib.sha256(
        "\n".join(holdout_relative_paths).encode("utf-8")
    ).hexdigest()
    holdout_valid = (
        holdout.get("version")
        == "release_stratified_official_relative_path_hash_v2"
        and holdout.get("membership_path_scope")
        == "official_release_relative_path"
        and holdout.get("holdout_membership_locked") is True
        and holdout_relative_paths_typed
        and len(holdout_relative_paths)
        == FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
        and holdout_relative_paths == sorted(holdout_relative_paths)
        and len(set(holdout_relative_paths))
        == FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
        and holdout_relative_path_sha256
        == FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
        and holdout.get("holdout_relative_path_sha256")
        == FORMAL_HOLDOUT_RELATIVE_PATH_SHA256
        and holdout.get("normalization_fit") == "training_shards_only"
        and holdout.get("tail_taxonomy_scope")
        == "training_shards_fit_applied_to_holdout"
        and holdout.get("instruction_rarity_fit") == "training_shards_only"
        and holdout.get("pt_allocation_fit") == "training_shards_only"
        and holdout.get("evaluation_predictions_scope") == "holdout_shards_only"
        and int(holdout.get("training_shards", 0))
        + int(holdout.get("holdout_shards", 0))
        == 4_096
        and int(holdout.get("holdout_shards", 0)) > 0
        and int(holdout.get("holdout_shards", -1))
        == FORMAL_HOLDOUT_SHARDS_PER_RELEASE * 2
        and float(holdout.get("requested_holdout_fraction", -1.0))
        == FORMAL_HOLDOUT_FRACTION
        and int(holdout.get("seed", -1)) == FORMAL_SEED
        and set(holdout_releases) == set(expected_releases)
        and all(
            int(item.get("training_shards", 0)) > 0
            and int(item.get("holdout_shards", 0))
            == FORMAL_HOLDOUT_SHARDS_PER_RELEASE
            and int(item.get("training_shards", 0))
            + int(item.get("holdout_shards", 0))
            == expected_releases[release]["shards"]
            for release, item in holdout_releases.items()
        )
    )
    tail_contract_valid = (
        int(tail_contract.get("transform_fit_row_count", -1))
        == int(holdout.get("training_shards", -2))
        and int(tail_contract.get("allocation_fit_row_count", -1))
        == int(holdout.get("training_shards", -2))
        and tail_contract.get("instruction_document_frequency_fit")
        == "normalization_fit_rows_only"
    )
    try:
        pt_source_valid = (
            int(pt_source.get("count", 0)) >= 4_096
            and pt_source.get("sha256") == FORMAL_PT_SOURCE_SHA256
            and math.isfinite(
                float(pt_source.get("coefficient_of_variation", math.nan))
            )
        )
    except (TypeError, ValueError):
        pt_source_valid = False
    finite_effect_values = [
        effect.get("source_pred_tail_share"),
        effect.get("qtail_pred_tail_share"),
        effect.get("predicted_tail_share_gain_pp"),
        effect.get("source_extreme_underallocation_rate"),
        effect.get("qtail_extreme_underallocation_rate"),
        effect.get("extreme_underallocation_reduction_pp"),
        bootstrap.get("mean_gain_pp"),
        bootstrap.get("ci95_low_pp"),
        bootstrap.get("ci95_high_pp"),
        bootstrap.get("descriptive_fraction_gain_le_zero"),
        randomization.get("observed_gain_pp"),
        randomization.get("diagnostic_exceedance_fraction"),
    ]
    try:
        recomputed_tail_gain = (
            float(effect.get("qtail_pred_tail_share"))
            - float(effect.get("source_pred_tail_share"))
        ) * 100.0
        recomputed_extreme_reduction = (
            float(effect.get("source_extreme_underallocation_rate"))
            - float(effect.get("qtail_extreme_underallocation_rate"))
        ) * 100.0
        recomputed_supported = bool(
            recomputed_tail_gain >= 2.0
            and float(bootstrap.get("ci95_low_pp")) >= 2.0
            and recomputed_extreme_reduction > 0.0
        )
        recomputed_not_supported = bool(
            float(bootstrap.get("ci95_high_pp")) < 2.0
            or recomputed_extreme_reduction <= 0.0
        )
        recomputed_outcome = (
            "supported"
            if recomputed_supported
            else "not_supported"
            if recomputed_not_supported
            else "inconclusive"
        )
    except (TypeError, ValueError):
        recomputed_tail_gain = math.nan
        recomputed_extreme_reduction = math.nan
        recomputed_supported = False
        recomputed_outcome = "invalid"
    effect_metrics_valid = (
        all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in finite_effect_values
        )
        and effect.get("tail_definition")
        == "heldout_top_30_percent_by_record_informed_tail_score_v2"
        and effect.get("extreme_definition")
        == "heldout_top_10_percent_by_record_informed_tail_score_v2"
        and effect.get("evaluation_scope")
        == "deterministic_release_stratified_heldout_shards"
        and int(bootstrap.get("samples", -1)) == 5_000
        and bootstrap.get("method")
        == (
            "paired_release_stratified_shard_bootstrap_"
            "within_draw_renormalization"
        )
        and bootstrap.get("strata") == ["1.0.0", "1.0.1"]
        and sum(
            int(value)
            for value in bootstrap.get("strata_counts", {}).values()
        )
        == int(holdout.get("holdout_shards", -1))
        and bootstrap.get("p_gain_le_zero_is_p_value") is False
        and randomization.get("version")
        == "paired_shard_arm_swap_diagnostic_v2"
        and int(randomization.get("samples", -1)) == 5_000
        and randomization.get("unit")
        == "non_independent_heldout_shard_weight"
        and randomization.get("finite_sample_correction")
        == "(k+1)/(B+1)"
        and randomization.get(
            "exchangeability_justified_by_experiment_design"
        )
        is False
        and randomization.get("inference_role")
        == "dependency_sensitive_descriptive_diagnostic_only"
        and randomization.get("conditional_p_value_is_valid_p_value")
        is False
        and hypothesis_gate.get("name")
        == "heldout_tail_allocation_outcome_v4"
        and float(hypothesis_gate.get("minimum_tail_share_gain_pp", -1.0))
        == 2.0
        and hypothesis_gate.get("requires_ci95_low_at_least_minimum")
        is True
        and hypothesis_gate.get(
            "requires_positive_extreme_underallocation_reduction"
        )
        is True
        and hypothesis_gate.get("completion_role")
        == "outcome_only_not_experiment_execution_gate"
        and hypothesis_gate.get(
            "randomization_diagnostic_is_valid_p_value"
        )
        is False
        and hypothesis_gate.get("outcome") == recomputed_outcome
        and hypothesis_gate.get("supported") is recomputed_supported
        and hypothesis_gate.get("passed") is recomputed_supported
        and math.isclose(
            float(effect.get("predicted_tail_share_gain_pp", math.nan)),
            recomputed_tail_gain,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(
                effect.get(
                    "extreme_underallocation_reduction_pp",
                    math.nan,
                )
            ),
            recomputed_extreme_reduction,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and int(effect.get("tail_selected_shards", -1)) == 246
        and int(effect.get("tail_total_holdout_shards", -1)) == 820
        and int(effect.get("extreme_selected_shards", -1)) == 82
        and int(effect.get("extreme_total_holdout_shards", -1)) == 820
        and 0.0 <= float(effect.get("source_pred_tail_share", -1.0)) <= 1.0
        and 0.0 <= float(effect.get("qtail_pred_tail_share", -1.0)) <= 1.0
        and 0.0
        <= float(effect.get("source_extreme_underallocation_rate", -1.0))
        <= 1.0
        and 0.0
        <= float(effect.get("qtail_extreme_underallocation_rate", -1.0))
        <= 1.0
        and float(bootstrap.get("ci95_low_pp", 1.0))
        <= float(bootstrap.get("ci95_high_pp", -1.0))
        and 0.0
        <= float(
            bootstrap.get("descriptive_fraction_gain_le_zero", -1.0)
        )
        <= 1.0
        and 0.0
        < float(randomization.get("diagnostic_exceedance_fraction", -1.0))
        <= 1.0
        and math.isclose(
            float(randomization.get("conditional_p_value", math.nan)),
            float(
                randomization.get(
                    "diagnostic_exceedance_fraction",
                    math.nan,
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    rare_curve = rare_coverage.get("curve", [])
    rare_time = rare_coverage.get("time_to_coverage", [])
    try:
        rare_status = rare_coverage.get("status")
        rare_shape_valid = bool(
            (
                rare_status == "complete"
                and int(
                    rare_coverage.get(
                        "rare_holdout_fingerprint_count", 0
                    )
                )
                > 0
                and [
                    int(item.get("draw_budget", -1))
                    for item in rare_curve
                ]
                == [10, 25, 50, 100, 200, 400, 800]
                and [
                    float(item.get("coverage_threshold", -1.0))
                    for item in rare_time
                ]
                == [0.10, 0.25, 0.50, 0.75]
            )
            or (
                rare_status == "no_eligible_fingerprints"
                and int(
                    rare_coverage.get(
                        "rare_holdout_fingerprint_count", -1
                    )
                )
                == 0
                and int(
                    rare_coverage.get(
                        "unseen_in_training_fingerprint_count", -1
                    )
                )
                == 0
                and rare_curve == []
                and rare_time == []
                and bool(rare_coverage.get("status_reason"))
            )
        )
        rare_coverage_valid = bool(
            rare_coverage.get("version")
            == "heldout_instruction_fingerprint_coverage_v1"
            and rare_status
            in {"complete", "no_eligible_fingerprints"}
            and rare_shape_valid
            and rare_coverage.get("metric_role")
            == "auxiliary_descriptive_metric_not_a_completion_gate"
            and rare_coverage.get("rarity_fit_scope")
            == "training_shards_only"
            and rare_coverage.get("evaluation_scope") == "holdout_shards_only"
            and "not semantic task coverage"
            in str(rare_coverage.get("claim_boundary", ""))
            and int(rare_coverage.get("training_shards", -1)) == 3_276
            and int(rare_coverage.get("holdout_shards", -1)) == 820
            and int(
                rare_coverage.get(
                    "max_training_shard_document_frequency",
                    -1,
                )
            )
            == 1
            and all(
                0.0
                <= float(item.get("source_expected_coverage", -1.0))
                <= 1.0
                and 0.0
                <= float(item.get("qtail_expected_coverage", -1.0))
                <= 1.0
                and math.isclose(
                    float(item.get("gain_pp", math.nan)),
                    (
                        float(item.get("qtail_expected_coverage"))
                        - float(item.get("source_expected_coverage"))
                    )
                    * 100.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for item in rare_curve
            )
        )
    except (TypeError, ValueError):
        rare_coverage_valid = False
    feature_total_shards = int(feature_status.get("total_shards", 0) or 0)
    feature_records_decoded = trajectory.get("records_decoded")
    if feature_records_decoded is None:
        feature_records_decoded = feature_status.get("records_decoded")
    feature_parse_rate = trajectory.get("record_parse_rate")
    if feature_parse_rate is None and feature_total_shards:
        feature_parse_rate = (
            int(feature_status.get("parsed_shards", 0) or 0)
            / feature_total_shards
        )
    feature_scan_complete_rate = trajectory.get("record_scan_complete_rate")
    if feature_scan_complete_rate is None and feature_total_shards:
        feature_scan_complete_rate = (
            int(feature_status.get("record_scan_complete_shards", 0) or 0)
            / feature_total_shards
        )
    backend_workspace = Path(
        "/Users/avalok/work/Q-TAIL-MVP/external_data/embodied_full/"
        "training_backends/droid_policy_learning"
    )
    backend_orico = Path(str(backend.get("path", "")))
    backend_workspace_links_to_orico = bool(
        backend_workspace.is_symlink()
        and backend_orico.exists()
        and backend_workspace.resolve() == backend_orico.resolve()
    )
    droid_data_orico = result_root.parents[1] / "data" / "droid"
    workspace_droid_data = Path(
        "/Users/avalok/work/Q-TAIL-MVP/data/droid"
    )
    workspace_droid_data_links_to_orico = bool(
        workspace_droid_data.is_symlink()
        and droid_data_orico.exists()
        and workspace_droid_data.resolve() == droid_data_orico.resolve()
    )
    workspace_results = Path(
        "/Users/avalok/work/Q-TAIL-MVP/results/qtail_droid_full"
    )
    workspace_results_links_to_orico = bool(
        workspace_results.is_symlink()
        and result_root.exists()
        and workspace_results.resolve() == result_root.resolve()
    )
    droid_data_is_on_orico = bool(
        droid_data_orico.is_dir()
        and str(droid_data_orico.resolve()).startswith("/Volumes/ORICO/")
    )
    requirements = [
        {
            "id": "existing_assets_on_orico",
            "label": "既有 Open X 与训练代码迁移到 ORICO",
            "passed": (
                marker_state.get("openx_migration_complete", False)
                and openx.get("bytes") == OPENX_EXPECTED_BYTES
                and openx.get("partials") == 0
                and Path("/Users/avalok/work/Q-TAIL-MVP/data/openx_demo").is_symlink()
                and marker_state.get("droid_backend_ready", False)
                and backend.get("exists") is True
                and bool(expected_backend_commit)
                and backend.get("commit") == expected_backend_commit
                and backend_workspace_links_to_orico
                and droid_data_is_on_orico
                and workspace_droid_data_links_to_orico
                and workspace_results_links_to_orico
                and code_snapshot.get("verified") is True
                and int(code_snapshot.get("error_count", -1)) == 0
                and code_snapshot.get("workspace_parity", {}).get("verified")
                is True
                and int(
                    code_snapshot.get("workspace_parity", {}).get(
                        "error_count", -1
                    )
                )
                == 0
                and int(
                    code_snapshot.get("workspace_parity", {}).get(
                        "verified_file_count", -1
                    )
                )
                == int(code_snapshot.get("verified_file_count", -2))
                and snapshot_publish_audit.get("valid") is True
            ),
            "evidence": {
                "openx": {
                    "path": "/Volumes/ORICO/qtail_full_training/data/openx_demo",
                    "bytes": openx.get("bytes", 0),
                    "expected_bytes": OPENX_EXPECTED_BYTES,
                    "workspace_symlink": "/Users/avalok/work/Q-TAIL-MVP/data/openx_demo",
                },
                "droid_backend": {
                    **backend,
                    "expected_commit": expected_backend_commit,
                    "workspace_path": str(backend_workspace),
                    "workspace_is_external_symlink": (
                        backend_workspace_links_to_orico
                    ),
                },
                "droid_data": {
                    "path": str(droid_data_orico),
                    "is_on_orico": droid_data_is_on_orico,
                    "workspace_path": str(workspace_droid_data),
                    "workspace_is_external_symlink": (
                        workspace_droid_data_links_to_orico
                    ),
                },
                "results": {
                    "path": str(result_root),
                    "workspace_path": str(workspace_results),
                    "workspace_is_external_symlink": (
                        workspace_results_links_to_orico
                    ),
                },
                "qtail_orchestration_snapshot": code_snapshot,
                "qtail_orchestration_snapshot_publication": (
                    snapshot_publish_audit
                ),
            },
        },
        {
            "id": "official_source_and_manifest",
            "label": "官方 DROID 来源与对象清单",
            "passed": (
                source_probe_marker.get("valid") is True
                and source_probe.get("status") == "verified"
                and int(source_probe.get("remote_bytes", -1)) == REMOTE_BYTES
                and object_manifest.get("status") in {"verified", "complete"}
                and int(object_manifest.get("object_count", -1)) == 4102
                and int(object_manifest.get("total_bytes", -1)) == REMOTE_BYTES
                and checksum_summary.get("manifest_valid") is True
                and metadata_audit_summary.get("valid") is True
            ),
            "evidence": {
                "source": REMOTE_URI,
                "source_probe": str(result_root / "droid_source_probe.json"),
                "source_probe_marker": source_probe_marker,
                "object_manifest": str(result_root / "droid_object_manifest.json"),
                "object_count": object_manifest.get("object_count"),
                "bytes": object_manifest.get("total_bytes"),
                "checksum_manifest": str(
                    result_root / "droid_object_checksum_manifest.json"
                ),
                "checksum_object_count": checksum_manifest.get("object_count"),
                "release_metadata_audit": str(
                    result_root / "droid_release_metadata_audit.json"
                ),
                "release_metadata_summary": metadata_audit_summary,
            },
        },
        {
            "id": "uniclash_transport_isolation",
            "label": "UniClashCore 在线且 DROID 下载强制直连",
            "passed": transport_guard_valid,
            "evidence": {
                "guard": str(
                    Path("/Users/avalok/work/Q-TAIL-MVP/.tmp")
                    / "qtail-uniclash-transport-guard.json"
                ),
                "status": transport_guard.get("status"),
                "uniclash": guard_uniclash,
                "system_proxy_bypass": guard_bypass,
                "active_droid_transfers": transport_guard.get(
                    "active_droid_transfers"
                ),
                "cumulative": guard_cumulative,
                "guard_generated_at": transport_guard.get("generated_at"),
                "guard_age_seconds": guard_age_seconds,
                "guard_heartbeat_fresh": (
                    guard_age_seconds is not None
                    and guard_age_seconds <= 10
                ),
                "adjudication": transport_adjudication,
                "adjudication_required": adjudication_required,
                "adjudication_valid": adjudication_valid,
                "blocked_processes": transport_guard.get("blocked_processes"),
                "global_violations": transport_guard.get("global_violations"),
            },
        },
        {
            "id": "full_mirror_checksum",
            "label": "3.366 TiB 镜像与 checksum 完整性",
            "passed": (
                marker_state.get("droid_download_complete", False)
                and marker_state.get("droid_checksum_verified", False)
                and verification.get("status") == "complete"
                and verification.get("ready_for_full_allocation_training") is True
                and int(verification.get("checksum_rsync_returncode", -1)) == 0
                and int(verification.get("local_official_bytes", -1)) == REMOTE_BYTES
                and int(verification.get("missing_object_count", -1)) == 0
                and int(verification.get("size_mismatch_count", -1)) == 0
                and int(verification.get("checksum_error_count", -1)) == 0
                and int(verification.get("extra_file_count", -1)) == 0
                and int(verification.get("partial_file_count", -1)) == 0
                and checksum_summary.get("status") == "complete"
                and int(checksum_summary.get("verified_objects", -1)) == 4_102
            ),
            "evidence": {
                "download_verification": str(result_root / "download_verification.json"),
                "local_official_bytes": verification.get("local_official_bytes"),
                "checksum_returncode": verification.get("checksum_rsync_returncode"),
                "checksum_error_count": verification.get(
                    "checksum_error_count"
                ),
                "object_checksum_ledger": checksum_summary,
            },
        },
        {
            "id": "all_record_scan",
            "label": "全部 TFRecord 的 100% 记录扫描",
            "passed": (
                marker_state.get("droid_feature_extraction_complete", False)
                and training_report.get("training_scope")
                == "all_complete_shards_all_decodable_records"
                and trajectory.get("full_record_mode") is True
                and float(trajectory.get("record_parse_rate", 0.0)) == 1.0
                and float(trajectory.get("record_scan_complete_rate", 0.0)) == 1.0
                and input_audit.get("verified") is True
                and cache_verification.get("status") == "verified"
                and cache_verification.get("all_official_tfrecords") is True
                and cache_verification.get("full_official_record_count_match")
                is True
                and cache_verification.get(
                    "unreferenced_cache_excluded_from_training"
                )
                is True
                and int(cache_verification.get("error_count", -1)) == 0
                and cache_verification.get("feature_values_recomputed")
                is True
                and cache_verification.get("all_feature_values_recomputed")
                is True
                and int(
                    cache_verification.get("recomputed_feature_count", -1)
                )
                == 4_096
                and release_composition_valid
                and int(input_audit.get("expected_tfrecord_shards", -1))
                == int(input_audit.get("actual_tfrecord_shards", -2))
            ),
            "evidence": {
                "feature_status": str(result_root / "droid_feature_extraction_status.json"),
                "processed_shards": feature_status.get("processed_shards"),
                "total_shards": feature_status.get("total_shards"),
                "records_decoded": feature_records_decoded,
                "parse_rate": feature_parse_rate,
                "scan_complete_rate": feature_scan_complete_rate,
                "formal_gate_passed": marker_state.get(
                    "droid_feature_extraction_complete", False
                ),
                "cache_verification": str(
                    result_root / "droid_feature_cache_verification.json"
                ),
                "official_expected_records": cache_verification.get(
                    "official_expected_records"
                ),
                "verified_decoded_records": cache_verification.get(
                    "verified_decoded_records"
                ),
                "cache_directory_count": cache_verification.get(
                    "cache_directory_count"
                ),
                "unreferenced_cache_count": cache_verification.get(
                    "unreferenced_cache_count"
                ),
                "unreferenced_cache_bytes": cache_verification.get(
                    "unreferenced_cache_bytes"
                ),
                "unreferenced_cache_excluded_from_training": (
                    cache_verification.get(
                        "unreferenced_cache_excluded_from_training"
                    )
                ),
                "release_record_audit": cache_verification.get(
                    "release_record_audit", []
                ),
                "training_release_composition": release_composition,
                "training_release_composition_valid": release_composition_valid,
            },
        },
        {
            "id": "same_compute_training",
            "label": "Source 与 Q-Tail 同算力训练",
            "passed": (
                marker_state.get("droid_model_training_complete", False)
                and marker_state.get("droid_training_complete", False)
                and generation_gate_summary["valid"]
                and training_report.get("status") == "complete"
                and formal_protocol_valid
                and steps == 20_000
                and int(training_report.get("total_steps_per_arm", -1)) == 40_000
                and int(compute.get("source_steps", -2)) == 40_000
                and int(compute.get("qtail_steps", -3)) == 40_000
                and int(compute.get("evaluation_source_steps", -1)) == steps
                and int(compute.get("evaluation_qtail_steps", -1)) == steps
                and int(compute.get("deployment_source_steps", -1)) == steps
                and int(compute.get("deployment_qtail_steps", -1)) == steps
                and int(
                    compute.get(
                        "evaluation_source_optimizer_updates", -1
                    )
                )
                == steps
                and int(
                    compute.get(
                        "evaluation_qtail_optimizer_updates", -1
                    )
                )
                == steps
                and int(
                    compute.get(
                        "deployment_source_optimizer_updates", -1
                    )
                )
                == steps
                and int(
                    compute.get(
                        "deployment_qtail_optimizer_updates", -1
                    )
                )
                == steps
                and int(compute.get("source_optimizer_updates", -1))
                == steps * 2
                and int(compute.get("qtail_optimizer_updates", -1))
                == steps * 2
                and compute.get("optimizer_update_semantics")
                == (
                    "Checkpoint step k stores the state after exactly k "
                    "optimizer updates; each stage ends at k=steps."
                )
                and set(compute.get("resume", {}))
                == {
                    "evaluation_source",
                    "evaluation_qtail",
                    "deployment_source",
                    "deployment_qtail",
                }
                and all(
                    int(item.get("target_step", -1)) == steps
                    and int(
                        item.get("optimizer_updates_completed", -1)
                    )
                    == steps
                    and item.get("device")
                    == compute.get("training_device")
                    and item.get("optimizer")
                    == compute.get("same_optimizer")
                    and item.get("environment_fingerprint")
                    == compute.get("checkpoint_environment_fingerprint")
                    and (
                        not item.get("resumed")
                        or (
                            item.get("checkpoint_device")
                            == compute.get("training_device")
                            and item.get("checkpoint_optimizer")
                            == compute.get("same_optimizer")
                            and item.get(
                                "checkpoint_environment_fingerprint"
                            )
                            == compute.get(
                                "checkpoint_environment_fingerprint"
                            )
                        )
                    )
                    and item.get("step_semantics")
                    == (
                        "Checkpoint step k is the state after exactly k "
                        "optimizer updates."
                    )
                    for item in compute.get("resume", {}).values()
                )
                and compute.get("architecture")
                == "AllocationHead(10→32→16→1)"
                and compute.get("same_architecture") is True
                and compute.get("same_optimizer")
                == "AdamW(lr=0.002, weight_decay=0.0001)"
                and compute.get("same_seed") is True
                and compute.get("same_features") is True
                and compute.get("same_device") is True
                and compute.get("same_environment_fingerprint") is True
                and len(
                    str(
                        compute.get(
                            "runtime_environment_fingerprint", ""
                        )
                    )
                )
                == 64
                and len(
                    str(
                        compute.get(
                            "checkpoint_environment_fingerprint", ""
                        )
                    )
                )
                == 64
                and compute.get("checkpoint_environment_contract", {}).get(
                    "version"
                )
                == "qtail_checkpoint_environment_v2"
                and compute.get("same_parameter_count") is True
                and int(compute.get("source_parameter_count", -1)) > 0
                and int(compute.get("source_parameter_count", -1))
                == int(compute.get("qtail_parameter_count", -2))
                and effect_metrics_valid
                and holdout_valid
                and tail_contract_valid
                and pt_source_valid
                and rare_coverage_valid
            ),
            "evidence": {
                "training_report": str(result_root / "droid_full_training_report.json"),
                "formal_protocol_valid": formal_protocol_valid,
                "steps_per_arm": steps,
                "device": compute.get("training_device"),
                "architecture": compute.get("architecture"),
                "same_architecture": compute.get("same_architecture"),
                "optimizer": compute.get("same_optimizer"),
                "source_optimizer_updates": compute.get(
                    "source_optimizer_updates"
                ),
                "qtail_optimizer_updates": compute.get(
                    "qtail_optimizer_updates"
                ),
                "optimizer_update_semantics": compute.get(
                    "optimizer_update_semantics"
                ),
                "same_seed": compute.get("same_seed"),
                "same_features": compute.get("same_features"),
                "same_device": compute.get("same_device"),
                "source_parameter_count": compute.get("source_parameter_count"),
                "qtail_parameter_count": compute.get("qtail_parameter_count"),
                "same_parameter_count": compute.get("same_parameter_count"),
                "effect_metrics_valid": effect_metrics_valid,
                "holdout_valid": holdout_valid,
                "tail_contract_valid": tail_contract_valid,
                "holdout_shards": holdout.get("holdout_shards"),
                "pt_source_valid": pt_source_valid,
                "pt_source_sha256": pt_source.get("sha256"),
                "rare_instruction_fingerprint_coverage_valid": (
                    rare_coverage_valid
                ),
                "rare_instruction_fingerprint_count": rare_coverage.get(
                    "rare_holdout_fingerprint_count"
                ),
                "bootstrap_samples": bootstrap.get("samples"),
                "bootstrap_ci95_pp": [
                    bootstrap.get("ci95_low_pp"),
                    bootstrap.get("ci95_high_pp"),
                ],
                "bootstrap_descriptive_fraction_gain_le_zero": (
                    bootstrap.get(
                        "descriptive_fraction_gain_le_zero"
                    )
                ),
                "arm_swap_diagnostic_exceedance_fraction": randomization.get(
                    "diagnostic_exceedance_fraction"
                ),
                "arm_swap_diagnostic_is_valid_p_value": randomization.get(
                    "conditional_p_value_is_valid_p_value"
                ),
                "hypothesis_outcome": hypothesis_gate.get("outcome"),
                "hypothesis_supported": hypothesis_gate.get("supported"),
                "pipeline_generation_gate": str(
                    result_root / "pipeline_generation_gate.json"
                ),
                "pipeline_generation_gate_summary": (
                    generation_gate_summary
                ),
            },
        },
        {
            "id": "intermediate_artifacts",
            "label": "中间结果、checkpoint 与 SHA-256 清单",
            "passed": (
                artifact_manifest.get("status") == "complete"
                and len(artifact_manifest.get("artifacts", [])) >= 8
                and environment_manifest_path in artifact_paths
                and environment_selftest_path in artifact_paths
                and artifact_integrity.get("verified") is True
                and not missing_required_artifacts
                and environment_manifest_valid
                and environment_selftest_valid
                and download_marker_selftest_valid
                and mirror_verifier_selftest_valid
                and training_gate_order_selftest_valid
                and downloader_single_writer_selftest_valid
                and runtime_process_contract_selftest_valid
                and generation_gate_summary["valid"]
                and uniclash_pre_checksum_gate_valid
                and uniclash_pre_checksum_gate_selftest_valid
                and live_partial_marker_rejection_valid
                and (
                    not process_log_gate_required
                    or process_log_manifest_valid
                )
                and immutable_final_artifact_contract
                and immutable_final_guard_valid
            ),
            "evidence": {
                "artifact_manifest": str(result_root / "droid_artifact_manifest.json"),
                "artifact_count": len(artifact_manifest.get("artifacts", [])),
                "environment_manifest": environment_manifest_path,
                "environment_manifest_in_artifact_manifest": (
                    environment_manifest_path in artifact_paths
                ),
                "environment_manifest_valid": environment_manifest_valid,
                "environment_gates": environment_gates,
                "environment_selftest": environment_selftest_path,
                "environment_selftest_in_artifact_manifest": (
                    environment_selftest_path in artifact_paths
                ),
                "environment_selftest_valid": environment_selftest_valid,
                "environment_checks": environment_checks,
                "download_marker_selftest": str(
                    result_root / "droid_download_marker_selftest.json"
                ),
                "download_marker_selftest_valid": (
                    download_marker_selftest_valid
                ),
                "download_marker_controls": download_marker_controls,
                "mirror_verifier_selftest": str(
                    result_root / "droid_mirror_verifier_selftest.json"
                ),
                "mirror_verifier_selftest_valid": (
                    mirror_verifier_selftest_valid
                ),
                "mirror_verifier_controls": mirror_verifier_controls,
                "training_gate_order_selftest": str(
                    result_root / "droid_training_gate_order_selftest.json"
                ),
                "training_gate_order_selftest_valid": (
                    training_gate_order_selftest_valid
                ),
                "training_gate_order_controls": (
                    training_gate_order_controls
                ),
                "downloader_single_writer_selftest": str(
                    result_root
                    / "droid_downloader_single_writer_selftest.json"
                ),
                "downloader_single_writer_selftest_valid": (
                    downloader_single_writer_selftest_valid
                ),
                "downloader_single_writer_checks": (
                    downloader_single_writer_checks
                ),
                "downloader_single_writer_activation": (
                    "next_natural_recovery_or_generation"
                ),
                "runtime_process_contract_selftest": str(
                    result_root
                    / "droid_runtime_process_contract_selftest.json"
                ),
                "runtime_process_contract_selftest_valid": (
                    runtime_process_contract_selftest_valid
                ),
                "runtime_process_contract_checks": runtime_process_checks,
                "pipeline_generation_gate": str(
                    result_root / "pipeline_generation_gate.json"
                ),
                "pipeline_generation_gate_summary": (
                    generation_gate_summary
                ),
                "uniclash_pre_checksum_gate": str(
                    result_root / "uniclash_pre_checksum_gate.json"
                ),
                "uniclash_pre_checksum_gate_valid": (
                    uniclash_pre_checksum_gate_valid
                ),
                "uniclash_pre_checksum_gate_checks": uniclash_gate_checks,
                "uniclash_pre_checksum_gate_selftest": str(
                    result_root
                    / "uniclash_pre_checksum_gate_selftest.json"
                ),
                "uniclash_pre_checksum_gate_selftest_valid": (
                    uniclash_pre_checksum_gate_selftest_valid
                ),
                "uniclash_pre_checksum_gate_selftest_checks": (
                    uniclash_gate_selftest_checks
                ),
                "live_partial_marker_rejection": str(
                    result_root
                    / "droid_live_partial_marker_rejection.json"
                ),
                "live_partial_marker_rejection_valid": (
                    live_partial_marker_rejection_valid
                ),
                "process_log_manifest": str(process_log_manifest_path),
                "process_log_gate_required": (
                    process_log_gate_required
                ),
                "required_process_log_artifacts": sorted(
                    required_process_log_artifact_paths
                ),
                "process_log_manifest_valid": process_log_manifest_valid,
                "process_log_count": len(process_log_entries),
                "process_log_paths": [
                    entry.get("path")
                    for entry in process_log_entries
                    if isinstance(entry, dict)
                ],
                "live_artifacts_excluded_from_final_manifest": (
                    artifact_paths.isdisjoint(live_artifact_paths)
                ),
                "immutable_final_artifacts": sorted(
                    immutable_final_artifact_paths
                ),
                "immutable_final_artifacts_present": (
                    immutable_final_artifact_paths.issubset(artifact_paths)
                ),
                "immutable_final_artifact_contract": (
                    immutable_final_artifact_contract
                ),
                "artifact_integrity": artifact_integrity,
                "formal_pre_page_artifact_count": len(
                    formal_pre_page_artifact_paths
                ),
                "required_artifact_count": len(required_artifact_paths),
                "required_artifacts": sorted(required_artifact_paths),
                "artifact_seal_state": {
                    "sealed_required_count": (
                        len(required_artifact_paths)
                        - len(missing_required_artifacts)
                    ),
                    "unsealed_required_count": len(
                        missing_required_artifacts
                    ),
                    "physically_missing_required_count": len(
                        physically_missing_required_artifacts
                    ),
                    "present_but_unsealed_required_count": len(
                        present_but_unsealed_required_artifacts
                    ),
                },
                "missing_required_artifacts": missing_required_artifacts,
                "missing_required_artifacts_definition": (
                    "Compatibility field: required paths absent from the "
                    "complete SHA-256 artifact manifest. Entries may already "
                    "exist on disk as live, mutable, or pre-seal evidence."
                ),
                "unsealed_required_artifacts": missing_required_artifacts,
                "physically_missing_required_artifacts": (
                    physically_missing_required_artifacts
                ),
                "present_but_unsealed_required_artifacts": (
                    present_but_unsealed_required_artifacts
                ),
                "immutable_final_guard_valid": immutable_final_guard_valid,
            },
        },
        {
            "id": "runtime_health",
            "label": "下载与训练后台运行健康",
            "passed": runtime.get("healthy") is True,
            "evidence": {
                "processes": runtime.get("processes", {}),
                "heartbeat_age_seconds": runtime.get("download_heartbeat_age_seconds"),
                "web_services": runtime.get("web_services", []),
                "web_gate_passed": runtime.get("web_gate_passed"),
                "launchd_supervision": runtime.get(
                    "launchd_supervision", {}
                ),
                "code_sha256": runtime.get("code_sha256", {}),
            },
        },
        {
            "id": "final_page_qa",
            "label": "新页面最终数据与桌面/移动端验收",
            "passed": marker_state.get("final_page_qa_effective", False),
            "evidence": {
                "page": "http://localhost:54655/qtail-droid-full-training",
                "marker": str(
                    Path("/Volumes/ORICO/qtail_full_training/manifests")
                    / "FINAL_PAGE_QA_COMPLETE"
                ),
                "committed": marker_state.get(
                    "final_page_qa_complete", False
                ),
                "preview_active": marker_state.get(
                    "final_page_qa_preview_active", False
                ),
                "qa_state": (
                    "committed"
                    if marker_state.get("final_page_qa_complete", False)
                    else (
                        "sealing"
                        if marker_state.get(
                            "final_page_qa_bootstrap_active", False
                        )
                        or marker_state.get(
                            "final_page_qa_sealed", False
                        )
                        else (
                            "blocked_contract"
                            if marker_state.get(
                                "final_qa_contract_blocked", False
                            )
                            else (
                                "running"
                                if marker_state.get(
                                    "final_page_qa_preview_active", False
                                )
                                else "waiting"
                            )
                        )
                    )
                ),
                "contract_blocker": str(
                    result_root / "final_qa_contract_blocked.json"
                ),
            },
        },
    ]
    execution_requirement = next(
        item for item in requirements if item["id"] == "same_compute_training"
    )
    experiment_execution_valid = bool(execution_requirement["passed"])
    formal_results_publishable = bool(
        experiment_execution_valid
        and marker_state.get("droid_training_complete", False)
        and marker_state.get("training_marker_validation", {}).get(
            "valid", False
        )
        and marker_state.get("final_page_qa_complete", False)
        and marker_state.get("droid_public_projection_committed", False)
        and marker_state.get("public_projection_validation", {}).get(
            "valid", False
        )
    )
    return {
        "generated_at": generated_at,
        "status": (
            "complete"
            if all(item["passed"] for item in requirements)
            else "in_progress"
        ),
        "passed_requirements": sum(item["passed"] for item in requirements),
        "total_requirements": len(requirements),
        "experiment_execution_valid": experiment_execution_valid,
        "formal_results_publishable": formal_results_publishable,
        "hypothesis_outcome": hypothesis_gate.get("outcome"),
        "hypothesis_supported": hypothesis_gate.get("supported"),
        "outcome_is_completion_gate": False,
        "requirements": requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, default=Path("/Volumes/ORICO/qtail_full_training"))
    args = parser.parse_args()

    job_root = args.job_root
    result_root = job_root / "results" / "qtail_droid_full"
    result_root.mkdir(parents=True, exist_ok=True)
    progress_lock = (result_root / ".progress_refresh.lock").open("a+")
    fcntl.flock(progress_lock.fileno(), fcntl.LOCK_EX)
    markers = job_root / "manifests"
    sealed_final = validate_final_marker(
        job_root,
        require_public_state=False,
    )
    sealed_projection = (
        validate_public_projection_marker(job_root)
        if sealed_final["valid"]
        else {"valid": False}
    )
    if sealed_final["valid"] and sealed_projection["valid"]:
        latest = read_json(result_root / "latest.json")
        audit = read_json(result_root / "completion_audit.json")
        if public_final_projection_is_committed(latest, audit):
            print(
                json.dumps(
                    {
                        "latest": str(result_root / "latest.json"),
                        "stage": latest.get("stage", "complete"),
                        "percent": latest.get("download", {}).get(
                            "logical_percent", 100.0
                        ),
                        "sealed": True,
                    }
                )
            )
            return
    data_root = job_root / "data" / "droid"
    openx_root = job_root / "data" / "openx_demo"
    log_path = job_root / "logs" / "droid_full_pipeline.log"
    download = scan_tree(data_root)
    openx = scan_tree(openx_root)
    usage = shutil.disk_usage(job_root)
    percent = min(100.0, download["bytes"] / REMOTE_BYTES * 100.0)
    generated_at = now()
    samples_path = result_root / "download_progress_samples.json"
    samples_payload = read_json(samples_path)
    samples = samples_payload.get("samples", []) if isinstance(samples_payload, dict) else []
    samples.append(
        {
            "generated_at": generated_at,
            "bytes": download["bytes"],
            "completed_logical_bytes": download["completed_logical_bytes"],
            "partial_allocated_bytes": download["partial_allocated_bytes"],
        }
    )
    atomic_write_json(
        samples_path,
        {
            "source_bytes": REMOTE_BYTES,
            "retention": "full_pipeline_history",
            "sample_count": len(samples),
            "samples": samples,
        },
    )

    throughput = 0.0
    eta_seconds = None
    if len(samples) >= 2:
        latest_time = datetime.fromisoformat(samples[-1]["generated_at"])
        window_start = latest_time - timedelta(minutes=15)
        candidates = [
            sample
            for sample in samples[:-1]
            if datetime.fromisoformat(sample["generated_at"]) >= window_start
            and sample["bytes"] <= samples[-1]["bytes"]
        ]
        baseline = candidates[0] if candidates else samples[-2]
        elapsed = (
            latest_time - datetime.fromisoformat(baseline["generated_at"])
        ).total_seconds()
        if elapsed > 0 and samples[-1]["bytes"] >= baseline["bytes"]:
            throughput = (samples[-1]["bytes"] - baseline["bytes"]) / elapsed
    if throughput > 0:
        eta_seconds = max(0, REMOTE_BYTES - download["bytes"]) / throughput

    marker_names = [
        "PIPELINE_STARTED",
        "OPENX_MIGRATION_COMPLETE",
        "DROID_BACKEND_READY",
        "DROID_SOURCE_PROBED",
        "DROID_DOWNLOAD_COMPLETE",
        "DROID_CHECKSUM_VERIFIED",
        "DROID_FEATURE_EXTRACTION_COMPLETE",
        "DROID_MODEL_TRAINING_STARTED",
        "DROID_MODEL_TRAINING_COMPLETE",
        "DROID_TRAINING_COMPLETE",
        "FINAL_PAGE_QA_PREVIEW",
        "FINAL_PAGE_QA_COMPLETE",
        "DROID_PUBLIC_PROJECTION_COMMITTED",
    ]
    marker_state = {name.lower(): (markers / name).exists() for name in marker_names}
    source_probe_marker = droid_source_probe_marker_summary(
        markers / "DROID_SOURCE_PROBED",
        result_root / "droid_source_probe.json",
        job_root=job_root,
    )
    marker_state["droid_source_probed"] = bool(
        marker_state["droid_source_probed"]
        and source_probe_marker["valid"]
    )
    training_marker_validation = validate_training_marker(job_root)
    final_marker_validation = validate_final_marker(
        job_root,
        require_public_state=False,
    )
    public_projection_validation = (
        validate_public_projection_marker(job_root)
        if final_marker_validation["valid"]
        else {
            "valid": False,
            "errors": ["final marker is not valid"],
            "marker": str(
                markers / "DROID_PUBLIC_PROJECTION_COMMITTED"
            ),
        }
    )
    final_bootstrap_validation = validate_final_bootstrap(job_root)
    marker_state["droid_training_complete"] = bool(
        marker_state["droid_training_complete"]
        and training_marker_validation["valid"]
    )
    marker_state["final_page_qa_sealed"] = bool(
        marker_state["final_page_qa_complete"]
        and final_marker_validation["valid"]
    )
    marker_state["final_page_qa_complete"] = bool(
        marker_state["final_page_qa_sealed"]
        and marker_state["droid_public_projection_committed"]
        and public_projection_validation["valid"]
    )
    marker_state["final_page_qa_bootstrap_active"] = bool(
        marker_state["final_page_qa_complete"] is False
        and final_bootstrap_validation["valid"]
    )
    marker_state["training_marker_validation"] = training_marker_validation
    marker_state["final_marker_validation"] = final_marker_validation
    marker_state["public_projection_validation"] = (
        public_projection_validation
    )
    marker_state["final_bootstrap_validation"] = final_bootstrap_validation
    marker_state["final_page_qa_preview_active"] = active_final_qa_preview(
        markers / "FINAL_PAGE_QA_PREVIEW"
    )
    marker_state["final_qa_contract_blocked"] = (
        read_json(result_root / "final_qa_contract_blocked.json").get(
            "status"
        )
        == "blocked"
    )
    # A preview marker is only a live QA lease.  Public completion must be
    # backed by the committed final marker and its full validation.
    stage = resolve_pipeline_stage(marker_state)

    training_report_path = result_root / "droid_full_training_report.json"
    verification_path = result_root / "download_verification.json"
    source_probe_path = result_root / "droid_source_probe.json"
    feature_status_path = result_root / "droid_feature_extraction_status.json"
    feature_prewarm_status_path = result_root / "droid_feature_prewarm_status.json"
    feature_prewarm_heartbeat_path = (
        result_root / "droid_feature_prewarm_heartbeat.json"
    )
    model_status_path = result_root / "droid_model_training_status.json"
    local_md5_rehash_status_path = (
        result_root / "droid_local_md5_rehash_status.json"
    )
    parallel_download_status_path = result_root / "parallel_download_status.json"
    object_manifest_path = result_root / "droid_object_manifest.json"
    checksum_manifest_path = result_root / "droid_object_checksum_manifest.json"
    checksum_ledger_path = result_root / "droid_object_checksum_ledger.json"
    release_metadata_audit_path = (
        result_root / "droid_release_metadata_audit.json"
    )
    artifact_manifest_path = result_root / "droid_artifact_manifest.json"
    environment_manifest_path = result_root / "droid_environment_manifest.json"
    environment_selftest_path = (
        result_root / "droid_environment_contract_selftest.json"
    )
    protocol_selftest_path = (
        result_root / "droid_protocol_selftest.json"
    )
    download_marker_selftest_path = (
        result_root / "droid_download_marker_selftest.json"
    )
    mirror_verifier_selftest_path = (
        result_root / "droid_mirror_verifier_selftest.json"
    )
    training_gate_order_selftest_path = (
        result_root / "droid_training_gate_order_selftest.json"
    )
    downloader_single_writer_selftest_path = (
        result_root / "droid_downloader_single_writer_selftest.json"
    )
    runtime_process_contract_selftest_path = (
        result_root / "droid_runtime_process_contract_selftest.json"
    )
    pipeline_generation_gate_path = (
        result_root / "pipeline_generation_gate.json"
    )
    stage_marker_hardening_selftest_path = (
        result_root / "droid_stage_marker_hardening_selftest.json"
    )
    progress_preview_selftest_path = (
        result_root / "droid_progress_preview_selftest.json"
    )
    artifact_manifest_merge_selftest_path = (
        result_root / "droid_artifact_manifest_merge_selftest.json"
    )
    pipeline_shell_contract_selftest_path = (
        result_root / "droid_pipeline_shell_contract_selftest.json"
    )
    uniclash_pre_checksum_gate_path = (
        result_root / "uniclash_pre_checksum_gate.json"
    )
    uniclash_pre_checksum_gate_selftest_path = (
        result_root / "uniclash_pre_checksum_gate_selftest.json"
    )
    live_partial_marker_rejection_path = (
        result_root / "droid_live_partial_marker_rejection.json"
    )
    cache_verification_path = result_root / "droid_feature_cache_verification.json"
    partial_cache_verification_path = (
        result_root / "droid_feature_cache_partial_verification.json"
    )
    incremental_closure_path = (
        result_root / "droid_incremental_closure_audit.json"
    )
    incremental_closure_selftest_path = (
        result_root / "droid_incremental_closure_selftest.json"
    )
    release_milestone_status_path = (
        result_root / "droid_release_milestone_status.json"
    )
    preflight_training_smoke_path = (
        result_root / "droid_preflight_training_smoke.json"
    )
    forecast_908_path = (
        result_root / "droid_forecast_908_summary.json"
    )
    scalability_canary_path = (
        result_root / "droid_scalability_canary_summary.json"
    )
    intermediate_checkpoint_manifest_path = (
        result_root / "droid_intermediate_checkpoint_manifest.json"
    )
    timeline_current_verification_path = (
        result_root / "pipeline_timeline_current_verification.json"
    )
    feature_status = read_json(feature_status_path)
    feature_prewarm_status = read_json(feature_prewarm_status_path)
    feature_prewarm_heartbeat = read_json(feature_prewarm_heartbeat_path)
    model_status = read_json(model_status_path)
    local_md5_rehash_status = read_json(local_md5_rehash_status_path)
    parallel_download = read_json(parallel_download_status_path)
    training_report = read_json(training_report_path)
    verification = read_json(verification_path)
    source_probe = read_json(source_probe_path)
    object_manifest = read_json(object_manifest_path)
    checksum_manifest = read_json(checksum_manifest_path)
    checksum_ledger = read_json(checksum_ledger_path)
    release_metadata_audit = read_json(release_metadata_audit_path)
    incremental_closure = read_json(incremental_closure_path)
    incremental_closure_selftest = read_json(
        incremental_closure_selftest_path
    )
    release_milestones = read_json(release_milestone_status_path)
    checksum_summary = checksum_ledger_summary(
        data_root=data_root,
        checksum_manifest=checksum_manifest,
        checksum_ledger=checksum_ledger,
    )
    capacity_headroom = capacity_headroom_summary(
        data_root=data_root,
        checksum_manifest=checksum_manifest,
        checksum_summary=checksum_summary,
        free_bytes=usage.free,
    )
    artifact_manifest = read_json(artifact_manifest_path)
    artifact_integrity = verify_artifact_manifest(artifact_manifest)
    environment_manifest = read_json(environment_manifest_path)
    environment_selftest = read_json(environment_selftest_path)
    protocol_selftest = read_json(protocol_selftest_path)
    download_marker_selftest = read_json(download_marker_selftest_path)
    mirror_verifier_selftest = read_json(mirror_verifier_selftest_path)
    training_gate_order_selftest = read_json(
        training_gate_order_selftest_path
    )
    downloader_single_writer_selftest = read_json(
        downloader_single_writer_selftest_path
    )
    runtime_process_contract_selftest = read_json(
        runtime_process_contract_selftest_path
    )
    pipeline_generation_gate = read_json(pipeline_generation_gate_path)
    stage_marker_hardening_selftest = read_json(
        stage_marker_hardening_selftest_path
    )
    progress_preview_selftest = read_json(progress_preview_selftest_path)
    artifact_manifest_merge_selftest = read_json(
        artifact_manifest_merge_selftest_path
    )
    pipeline_shell_contract_selftest = read_json(
        pipeline_shell_contract_selftest_path
    )
    uniclash_pre_checksum_gate = read_json(
        uniclash_pre_checksum_gate_path
    )
    uniclash_pre_checksum_gate_selftest = read_json(
        uniclash_pre_checksum_gate_selftest_path
    )
    live_partial_marker_rejection = read_json(
        live_partial_marker_rejection_path
    )
    transport_guard = read_json(
        Path("/Users/avalok/work/Q-TAIL-MVP/.tmp")
        / "qtail-uniclash-transport-guard.json"
    )
    transport_adjudication_path = (
        result_root / "uniclash_transport_guard_adjudication.json"
    )
    transport_adjudication = read_json(transport_adjudication_path)
    if transport_adjudication:
        transport_adjudication["archive_hashes_actual"] = {
            str(archive.get("path")): file_sha256(
                Path(str(archive.get("path")))
            )
            for archive in transport_adjudication.get(
                "preservation", {}
            ).get("archives", [])
            if archive.get("path")
        }
    cache_verification = read_json(cache_verification_path)
    partial_cache_verification = read_json(partial_cache_verification_path)
    preflight_training_smoke = read_json(preflight_training_smoke_path)
    forecast_908 = read_json(forecast_908_path)
    scalability_canary = read_json(scalability_canary_path)
    intermediate_checkpoint_manifest = checkpoint_manifest_projection(
        read_json(intermediate_checkpoint_manifest_path),
        result_root,
    )
    timeline_current_verification = read_json(
        timeline_current_verification_path
    )
    committed_feature_processed_shards = int(
        feature_prewarm_status.get(
            "shard_count",
            feature_status.get("processed_shards", 0),
        )
        or 0
    )
    committed_feature_records_decoded = int(
        feature_prewarm_status.get(
            "records_decoded",
            feature_status.get("records_decoded", 0),
        )
        or 0
    )
    committed_feature_parsed_shards = int(
        feature_prewarm_status.get(
            "parsed_shards",
            feature_status.get("parsed_shards", 0),
        )
        or 0
    )
    committed_feature_scan_complete_shards = int(
        feature_prewarm_status.get(
            "record_scan_complete_shards",
            feature_status.get("record_scan_complete_shards", 0),
        )
        or 0
    )
    committed_feature_parse_errors = feature_prewarm_status.get(
        "parse_errors",
        feature_status.get("parse_errors", 0),
    )
    if isinstance(committed_feature_parse_errors, list):
        committed_feature_parse_errors = len(
            committed_feature_parse_errors
        )
    committed_feature_represented_bytes = int(
        feature_prewarm_status.get(
            "represented_bytes",
            feature_status.get("represented_bytes", 0),
        )
        or 0
    )
    committed_feature_status = {
        "generated_at": feature_prewarm_status.get(
            "generated_at",
            feature_status.get("generated_at"),
        ),
        "status": "committed_prewarm_snapshot",
        "source_status": feature_prewarm_status.get(
            "status",
            feature_status.get("status"),
        ),
        "counter_semantics": "monotonic_committed_prewarm_snapshot_v1",
        "processed_shards": committed_feature_processed_shards,
        "total_shards": 4_096,
        "progress_percent": (
            committed_feature_processed_shards / 4_096 * 100.0
        ),
        "records_decoded": committed_feature_records_decoded,
        "parsed_shards": committed_feature_parsed_shards,
        "parse_rate": (
            committed_feature_parsed_shards
            / max(committed_feature_processed_shards, 1)
        ),
        "record_scan_complete_shards": (
            committed_feature_scan_complete_shards
        ),
        "record_scan_complete_rate": (
            committed_feature_scan_complete_shards
            / max(committed_feature_processed_shards, 1)
        ),
        "parse_errors": int(committed_feature_parse_errors or 0),
        "represented_bytes": committed_feature_represented_bytes,
        "represented_percent": (
            committed_feature_represented_bytes
            / OFFICIAL_TFRECORD_OBJECT_BYTES
            * 100.0
        ),
        "cache_hits": committed_feature_processed_shards,
        "active_pass": {
            "status": feature_status.get("status"),
            "processed_shards": feature_status.get(
                "processed_shards", 0
            ),
            "total_shards": feature_status.get("total_shards", 0),
            "records_decoded": feature_status.get(
                "records_decoded", 0
            ),
            "generated_at": feature_status.get("generated_at"),
        },
        "claim_boundary": (
            "Cumulative counters come from the last atomically committed "
            "prewarm pass and must be monotonic. active_pass is transient "
            "scan activity and may restart from shard zero."
        ),
    }
    repo_root = Path(__file__).resolve().parent.parent
    backend = git_state(job_root / "code" / "droid_policy_learning")
    backend_commit_path = markers / "droid_policy_learning_commit.txt"
    try:
        expected_backend_commit = backend_commit_path.read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        expected_backend_commit = None
    code_snapshot_root = job_root / "code" / "qtail_orchestration"
    code_snapshot_manifest = code_snapshot_root / "SHA256SUMS"
    snapshot_publish_audit_path = (
        result_root / "qtail_orchestration_snapshot_sync_audit.json"
    )
    snapshot_publish_audit = read_json(snapshot_publish_audit_path)
    code_snapshot = {
        **directory_summary(code_snapshot_root),
        "manifest": str(code_snapshot_manifest),
        "manifest_sha256": file_sha256(code_snapshot_manifest),
        "served_manifest": str(
            result_root / "qtail_orchestration_snapshot" / "SHA256SUMS"
        ),
        **verify_sha256_manifest(code_snapshot_root, code_snapshot_manifest),
        "workspace_parity": verify_snapshot_source_parity(
            repo_root,
            code_snapshot_manifest,
        ),
    }
    snapshot_publish_audit = snapshot_publication_projection(
        snapshot_publish_audit,
        audit_path=snapshot_publish_audit_path,
        repo_root=repo_root,
        snapshot_root=code_snapshot_root,
        manifest_path=code_snapshot_manifest,
        manifest_sha256=code_snapshot.get("manifest_sha256"),
        verified_file_count=int(
            code_snapshot.get("verified_file_count", -2)
        ),
    )
    runtime = process_snapshot(
        repo_root,
        job_root,
        parallel_download,
        stage,
        feature_prewarm_heartbeat,
    )
    completion_audit = build_completion_audit(
        generated_at=generated_at,
        openx=openx,
        marker_state=marker_state,
        source_probe=source_probe,
        source_probe_marker=source_probe_marker,
        pipeline_generation_gate=pipeline_generation_gate,
        object_manifest=object_manifest,
        checksum_manifest=checksum_manifest,
        checksum_summary=checksum_summary,
        release_metadata_audit=release_metadata_audit,
        verification=verification,
        feature_status=feature_status,
        training_report=training_report,
        artifact_manifest=artifact_manifest,
        artifact_integrity=artifact_integrity,
        environment_manifest=environment_manifest,
        environment_selftest=environment_selftest,
        download_marker_selftest=download_marker_selftest,
        mirror_verifier_selftest=mirror_verifier_selftest,
        training_gate_order_selftest=training_gate_order_selftest,
        downloader_single_writer_selftest=(
            downloader_single_writer_selftest
        ),
        runtime_process_contract_selftest=(
            runtime_process_contract_selftest
        ),
        uniclash_pre_checksum_gate=uniclash_pre_checksum_gate,
        uniclash_pre_checksum_gate_selftest=(
            uniclash_pre_checksum_gate_selftest
        ),
        live_partial_marker_rejection=live_partial_marker_rejection,
        transport_guard=transport_guard,
        transport_adjudication=transport_adjudication,
        cache_verification=cache_verification,
        runtime=runtime,
        backend=backend,
        expected_backend_commit=expected_backend_commit,
        code_snapshot=code_snapshot,
        snapshot_publish_audit=snapshot_publish_audit,
        result_root=result_root,
    )
    if transport_guard:
        atomic_write_json(
            result_root / "uniclash_transport_guard.json",
            transport_guard,
        )
    atomic_write_json(result_root / "completion_audit.json", completion_audit)
    payload = {
        "generated_at": generated_at,
        "status": completion_audit["status"],
        "stage": stage,
        "external_storage": {
            "volume": "ORICO",
            "job_root": str(job_root),
            "filesystem": "APFS",
            "capacity_bytes": usage.total,
            "free_bytes": usage.free,
            "required_with_slack_bytes": (
                REMOTE_BYTES
                + capacity_headroom["safety_reserve_bytes"]
            ),
            **capacity_headroom,
        },
        "source": {
            "uri": REMOTE_URI,
            "remote_bytes": REMOTE_BYTES,
            "remote_tib": REMOTE_BYTES / (1024**4),
        },
        "existing_assets": {
            "openx": {
                **openx,
                "path": str(openx_root),
                "workspace_path": "/Users/avalok/work/Q-TAIL-MVP/data/openx_demo",
                "expected_source_bytes": OPENX_EXPECTED_BYTES,
                "migration_percent": min(
                    100.0, openx["bytes"] / OPENX_EXPECTED_BYTES * 100.0
                ),
                "workspace_is_external_symlink": Path(
                    "/Users/avalok/work/Q-TAIL-MVP/data/openx_demo"
                ).is_symlink(),
                "deduplication_boundary": (
                    "Open X and DROID are different datasets. They share the ORICO job root "
                    "but are not treated as byte-identical duplicates."
                ),
            }
        },
        "download": {
            **download,
            "progress_sample_count": len(samples),
            "local_tib": download["bytes"] / (1024**4),
            "percent": percent,
            "remaining_bytes": max(0, REMOTE_BYTES - download["bytes"]),
            "throughput_bytes_per_second": throughput,
            "throughput_mib_per_second": throughput / (1024**2),
            "eta_seconds": eta_seconds,
            "eta_at": (
                (datetime.fromisoformat(generated_at) + timedelta(seconds=eta_seconds)).isoformat()
                if eta_seconds is not None
                else None
            ),
        },
        "markers": marker_state,
        "backend": backend,
        "code_snapshot": code_snapshot,
        "download_verification": verification,
        "source_probe": source_probe,
        "source_probe_marker": source_probe_marker,
        "pipeline_generation_gate": pipeline_generation_gate_summary(
            pipeline_generation_gate,
            gate_path=pipeline_generation_gate_path,
            script_path=(
                repo_root / "scripts/qtail_orico_full_pipeline.sh"
            ),
        ),
        "object_checksum_manifest": checksum_manifest,
        "object_checksums": checksum_summary,
        "release_metadata_audit": release_metadata_audit,
        "parallel_download": parallel_download,
        "transport_isolation": transport_guard,
        "transport_isolation_adjudication": transport_adjudication,
        "runtime": runtime,
        "artifact_link_availability": artifact_link_availability(repo_root),
        "completion_audit": completion_audit,
        "protocol_selftest": protocol_selftest,
        "environment_selftest": environment_selftest,
        "download_marker_selftest": download_marker_selftest,
        "mirror_verifier_selftest": mirror_verifier_selftest,
        "training_gate_order_selftest": training_gate_order_selftest,
        "downloader_single_writer_selftest": (
            downloader_single_writer_selftest
        ),
        "runtime_process_contract_selftest": (
            runtime_process_contract_selftest
        ),
        "stage_marker_hardening_selftest": (
            stage_marker_hardening_selftest
        ),
        "progress_preview_selftest": progress_preview_selftest,
        "artifact_manifest_merge_selftest": (
            artifact_manifest_merge_selftest
        ),
        "pipeline_shell_contract_selftest": (
            pipeline_shell_contract_selftest
        ),
        "uniclash_pre_checksum_gate": uniclash_pre_checksum_gate,
        "uniclash_pre_checksum_gate_selftest": (
            uniclash_pre_checksum_gate_selftest
        ),
        "live_partial_marker_rejection": live_partial_marker_rejection,
        "feature_extraction": committed_feature_status,
        "feature_extraction_active_pass": feature_status,
        "feature_prewarm": feature_prewarm_status,
        "feature_prewarm_heartbeat": feature_prewarm_heartbeat,
        "feature_record_audit": partial_cache_verification,
        "incremental_closure": incremental_closure,
        "incremental_closure_selftest": incremental_closure_selftest,
        "release_milestones": release_milestones,
        "model_training": model_status,
        "local_md5_rehash": local_md5_rehash_status,
        "preflight_training_smoke": preflight_training_smoke,
        "forecast_908": forecast_908,
        "scalability_canary": scalability_canary,
        "intermediate_checkpoint_manifest": (
            intermediate_checkpoint_manifest
        ),
        "pipeline_timeline_current_verification": (
            timeline_current_verification
        ),
        "training": training_report,
        "claim_boundary": [
            "Download progress is measured from bytes physically present on ORICO.",
            "Full allocation training starts only after rsync and checksum/size gates pass.",
            "Official release metadata is an early schema/count contract, not proof of downloaded or decoded records.",
            "Full mode streams every decodable record in every complete TFRecord shard and caches resumable per-shard features.",
            "Allocation-head evidence is not presented as end-to-end robot-policy success.",
        ],
        "logs": {
            "pipeline": str(log_path),
            "tail": tail(log_path),
        },
    }
    timeline_sample = {
        "generated_at": generated_at,
        "kind": "full_pipeline_sample",
        "stage": stage,
        "status": completion_audit["status"],
        "download": {
            "physical_bytes": download["bytes"],
            "completed_logical_bytes": download["completed_logical_bytes"],
            "partial_allocated_bytes": download["partial_allocated_bytes"],
            "percent": percent,
            "throughput_bytes_per_second": throughput,
            "eta_seconds": eta_seconds,
            "completed_objects": parallel_download.get(
                "completed_objects", 0
            ),
            "active_transfers": len(
                parallel_download.get("active", [])
            ),
            "failure_count": len(
                parallel_download.get("failures", {})
            ),
        },
        "object_checksums": {
            "status": checksum_summary.get("status"),
            "verified_objects": checksum_summary.get(
                "verified_objects", 0
            ),
            "expected_objects": checksum_summary.get(
                "expected_objects", 0
            ),
            "checksum_errors": checksum_summary.get(
                "checksum_errors", 0
            ),
            "stale_or_unexpected_entries": checksum_summary.get(
                "stale_or_unexpected_entries", 0
            ),
        },
        "release_metadata_audit": {
            "status": release_metadata_audit.get("status"),
            "combined_official_metadata": release_metadata_audit.get(
                "combined_official_metadata", {}
            ),
            "gates": release_metadata_audit.get("gates", {}),
        },
        "feature_extraction": {
            "counter_semantics": committed_feature_status[
                "counter_semantics"
            ],
            "processed_shards": committed_feature_status[
                "processed_shards"
            ],
            "total_shards": committed_feature_status[
                "total_shards"
            ],
            "records_decoded": committed_feature_status[
                "records_decoded"
            ],
            "parse_errors": committed_feature_status["parse_errors"],
            "record_scan_complete_shards": (
                committed_feature_status[
                    "record_scan_complete_shards"
                ]
            ),
            "active_pass": committed_feature_status["active_pass"],
            "official_record_counts_verified": (
                partial_cache_verification.get(
                    "official_record_counts_verified", False
                )
            ),
            "record_count_mismatch_count": (
                partial_cache_verification.get(
                    "record_count_mismatch_count", 0
                )
            ),
            "unreferenced_cache_count": (
                partial_cache_verification.get(
                    "unreferenced_cache_count", 0
                )
            ),
            "unreferenced_cache_excluded_from_training": (
                partial_cache_verification.get(
                    "unreferenced_cache_excluded_from_training",
                    False,
                )
            ),
        },
        "model_training": {
            "active_model": model_status.get(
                "active_model", "not_started"
            ),
            "step": model_status.get("step", 0),
            "total_steps": model_status.get("total_steps", 0),
            "formal_report_status": training_report.get(
                "status", "not_started"
            ),
        },
        "local_md5_rehash": {
            "status": local_md5_rehash_status.get(
                "status", "not_started"
            ),
            "processed_objects": local_md5_rehash_status.get(
                "processed_objects", 0
            ),
            "total_objects": local_md5_rehash_status.get(
                "total_objects", 0
            ),
            "processed_bytes": local_md5_rehash_status.get(
                "processed_bytes", 0
            ),
            "total_bytes": local_md5_rehash_status.get(
                "total_bytes", 0
            ),
        },
        "transport_isolation": {
            "status": transport_guard.get("status"),
            "process_classifier_version": transport_guard.get(
                "policy", {}
            ).get("process_classifier_version"),
            "expected_interface": transport_guard.get(
                "policy", {}
            ).get("expected_interface"),
            "interface_bound_transfers": sum(
                transfer.get("bound_interface")
                == transport_guard.get("policy", {}).get(
                    "expected_interface"
                )
                for transfer in transport_guard.get("transfers", [])
                if transfer.get("transport_kind") == "curl"
            ),
            "active_droid_transfers": transport_guard.get(
                "active_droid_transfers", 0
            ),
            "core_running": transport_guard.get(
                "uniclash", {}
            ).get("core_running"),
            "tun_enabled": transport_guard.get(
                "uniclash", {}
            ).get("tun_enabled"),
            "guard_samples": transport_guard.get(
                "cumulative", {}
            ).get("samples", 0),
            "blocked_samples": transport_guard.get(
                "cumulative", {}
            ).get("blocked_samples", 0),
            "forbidden_socket_observations": transport_guard.get(
                "cumulative", {}
            ).get("forbidden_socket_observations", 0),
            "wrong_route_observations": transport_guard.get(
                "cumulative", {}
            ).get("wrong_route_observations", 0),
            "guard_generated_at": transport_guard.get("generated_at"),
            "guard_age_seconds": timestamp_age_seconds(
                transport_guard.get("generated_at"),
                reference=datetime.fromisoformat(generated_at),
            ),
            "guard_process_count": len(
                runtime.get("processes", {}).get("transport_guard", [])
            ),
        },
        "external_storage": {
            "free_bytes": usage.free,
            "required_free_bytes": payload[
                "external_storage"
            ]["required_free_bytes"],
            "headroom_bytes": payload[
                "external_storage"
            ]["headroom_bytes"],
            "capacity_model_version": payload[
                "external_storage"
            ]["capacity_model_version"],
            "capacity_gate_passed": payload[
                "external_storage"
            ]["capacity_gate_passed"],
        },
        "completion": {
            "passed_requirements": completion_audit[
                "passed_requirements"
            ],
            "total_requirements": completion_audit[
                "total_requirements"
            ],
        },
        "runtime": {
            "healthy": runtime.get("healthy"),
            "required_process_counts": runtime.get(
                "required_process_counts", {}
            ),
            "heartbeat_gate_passed": runtime.get(
                "heartbeat_gate_passed"
            ),
            "mount_gate_passed": runtime.get(
                "mount_gate_passed"
            ),
            "web_gate_passed": runtime.get("web_gate_passed"),
            "power_policy": {
                "status": runtime.get("power_policy", {}).get(
                    "status"
                ),
                "sleep_minutes": runtime.get(
                    "power_policy", {}
                ).get("ac_power", {}).get("sleep_minutes"),
                "disk_sleep_minutes": runtime.get(
                    "power_policy", {}
                ).get("ac_power", {}).get("disk_sleep_minutes"),
                "external_media_asserted": runtime.get(
                    "power_policy", {}
                ).get("external_media_asserted"),
            },
        },
        "committed_markers": sorted(
            name
            for name, committed in marker_state.items()
            if committed is True
        ),
    }
    pipeline_timeline_path = result_root / "pipeline_timeline.json"
    payload["pipeline_timeline"] = update_pipeline_timeline(
        pipeline_timeline_path,
        samples_path,
        samples,
        timeline_sample,
    )
    payload["history_chart"] = build_history_chart(pipeline_timeline_path)
    atomic_write_json(result_root / "latest.json", payload)
    (result_root / "STATUS.md").write_text(
        "\n".join(
            [
                "# DROID Full Training on ORICO",
                "",
                f"- Generated: `{payload['generated_at']}`",
                f"- Stage: `{stage}`",
                f"- Download: `{download['bytes']}` / `{REMOTE_BYTES}` bytes (`{percent:.4f}%`)",
                f"- Complete TFRecords: `{download['tfrecords']}`",
                f"- Partial files: `{download['partials']}`",
                (
                    f"- Official MD5 verified: "
                    f"`{checksum_summary['verified_objects']}` / "
                    f"`{checksum_summary['expected_objects']}` objects"
                ),
                f"- 15-minute throughput: `{throughput / (1024**2):.2f} MiB/s`",
                f"- ETA seconds: `{eta_seconds}`",
                f"- ORICO free: `{usage.free / (1024**4):.3f} TiB`",
                (
                    f"- Feature extraction: "
                    f"`{committed_feature_status['processed_shards']}` / "
                    f"`{committed_feature_status['total_shards']}` shards, "
                    f"`{committed_feature_status['records_decoded']}` records "
                    "(committed monotonic snapshot)"
                ),
                (
                    f"- Local MD5 rehash: "
                    f"`{local_md5_rehash_status.get('processed_objects', 0)}` / "
                    f"`{local_md5_rehash_status.get('total_objects', 0)}` objects, "
                    f"status `{local_md5_rehash_status.get('status', 'not_started')}`"
                ),
                (
                    f"- Model training: `{model_status.get('active_model', 'not_started')}` "
                    f"step `{model_status.get('step', 0)}` / `{model_status.get('total_steps', 0)}`"
                ),
                f"- Training complete: `{marker_state['droid_training_complete']}`",
                "",
                (
                    "This pipeline reports full-record AllocationHead results only. "
                    "It does not claim robot-policy improvement; that requires "
                    "separate same-policy retraining and environment rollouts."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"latest": str(result_root / "latest.json"), "stage": stage, "percent": percent}))


if __name__ == "__main__":
    main()
