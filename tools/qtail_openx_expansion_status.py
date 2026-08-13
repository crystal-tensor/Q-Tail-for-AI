#!/usr/bin/env python3
"""Publish a compact status ledger for the Open X 1 TiB expansion."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qtail_openx_stage_marker import marker_status
from qtail_openx_final_page_qa import validate_report as validate_page_qa


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def tail_lines(path: Path, count: int = 20) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def launchd_supervision() -> dict[str, Any]:
    label = "com.qtail.openx-1t-expansion"
    result = subprocess.run(
        ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout if result.returncode == 0 else result.stderr
    state_match = re.search(r"\bstate = ([^\n]+)", output)
    runs_match = re.search(r"\bruns = (\d+)", output)
    interval_match = re.search(r"\brun interval = (\d+) seconds", output)
    source = Path("/Users/avalok/work/Q-TAIL-MVP/launchd/com.qtail.openx-1t-expansion.plist")
    installed = Path.home() / "Library/LaunchAgents/com.qtail.openx-1t-expansion.plist"
    source_sha = file_sha256(source)
    installed_sha = file_sha256(installed)
    return {
        "label": label,
        "loaded": result.returncode == 0,
        "state": state_match.group(1).strip() if state_match else "unknown",
        "runs": int(runs_match.group(1)) if runs_match else None,
        "interval_seconds": int(interval_match.group(1)) if interval_match else None,
        "source_plist_sha256": source_sha,
        "installed_plist_sha256": installed_sha,
        "plist_sha256_match": bool(source_sha and source_sha == installed_sha),
    }


def web_page_health() -> dict[str, Any]:
    path = "/qtail-openx-training"
    connection = http.client.HTTPConnection("127.0.0.1", 54655, timeout=2)
    try:
        connection.request("GET", path, headers={"Connection": "close"})
        response = connection.getresponse()
        response.read(1)
        return {
            "status": "passed" if response.status == 200 else "failed",
            "http_status": response.status,
            "url": f"http://127.0.0.1:54655{path}",
        }
    except OSError as error:
        return {
            "status": "failed",
            "http_status": None,
            "url": f"http://127.0.0.1:54655{path}",
            "error": str(error),
        }
    finally:
        connection.close()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def matching_processes(token: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,%cpu=,%mem=,etime=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    matches = []
    for line in result.stdout.splitlines():
        if token not in line or "qtail_openx_expansion_status.py" in line:
            continue
        parts = line.strip().split(None, 5)
        if len(parts) != 6:
            continue
        matches.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "cpu_percent": float(parts[2]),
                "memory_percent": float(parts[3]),
                "elapsed": parts[4],
                "command": parts[5],
            }
        )
    return matches


def leaf_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_ids = {int(process["ppid"]) for process in processes}
    return [process for process in processes if int(process["pid"]) not in parent_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    generated_at = now()
    previous_status = read_json(args.out)
    catalog = read_json(args.root / "openx_bucket_catalog.json")
    manifest = read_json(args.root / "openx_1t_object_manifest.json")
    download = read_json(args.root / "download_status.json")
    verification = read_json(args.root / "download_verification.json")
    training_status = read_json(args.root / "training" / "training_status.json")
    training_runtime = read_json(args.root / "training" / "training_runtime_status.json")
    optimizer_progress = read_json(args.root / "training" / "optimizer_progress.json")
    training_report = read_json(args.root / "training" / "openx_demo_training_report.json")
    prewarm = read_json(args.root / "prewarm_status.json")
    training = (
        training_report
        if training_report.get("status") == "complete"
        else training_runtime or training_status
    )
    synthesis_runtime = read_json(args.root / "synthesis" / "synthesis_runtime_status.json")
    synthesis = read_json(args.root / "synthesis" / "qtail_service_delivery_report.json")
    preflight_root = args.root / "handoff_preflight"
    preflight_training_runtime = read_json(
        preflight_root / "training" / "training_runtime_status.json"
    )
    preflight_training = read_json(
        preflight_root / "training" / "openx_demo_training_report.json"
    )
    preflight_cache = read_json(
        preflight_root / "training" / "feature_cache_usage.json"
    )
    preflight_optimizer = read_json(
        preflight_root / "training" / "optimizer_progress.json"
    )
    preflight_synthesis_runtime = read_json(
        preflight_root / "synthesis" / "synthesis_runtime_status.json"
    )
    preflight_synthesis = read_json(
        preflight_root / "synthesis" / "qtail_service_delivery_report.json"
    )
    guard = read_json(Path("/Users/avalok/work/Q-TAIL-MVP/.tmp/qtail-uniclash-transport-guard.json"))
    supervision = launchd_supervision()
    web_page = web_page_health()
    usage = shutil.disk_usage("/Volumes/ORICO")
    expected = int(manifest.get("total_bytes", 0))
    progress = int(download.get("progress_bytes", 0))
    if verification.get("status") == "verified":
        progress = expected
    previous_download = previous_status.get("download", {})
    previous_rate = float(previous_download.get("transfer_rate_bps") or 0.0)
    previous_progress = int(previous_download.get("progress_bytes") or 0)
    try:
        current_time = datetime.fromisoformat(generated_at)
        previous_time = datetime.fromisoformat(str(previous_status.get("generated_at")))
        elapsed = (current_time - previous_time).total_seconds()
    except (TypeError, ValueError):
        elapsed = 0.0
    instantaneous_rate = 0.0
    if 1.0 <= elapsed <= 300.0 and progress >= previous_progress:
        instantaneous_rate = (progress - previous_progress) / elapsed
    if instantaneous_rate > 0:
        transfer_rate = (
            instantaneous_rate
            if previous_rate <= 0
            else previous_rate * 0.8 + instantaneous_rate * 0.2
        )
    else:
        transfer_rate = previous_rate
    remaining_bytes = max(0, expected - progress)
    reserve_free_bytes = int(
        download.get("disk_headroom", {}).get("reserve_free_bytes", 0)
    )
    projected_free_after_download = max(0, usage.free - remaining_bytes)
    projected_headroom_after_download = (
        projected_free_after_download - reserve_free_bytes
    )
    eta_seconds = remaining_bytes / transfer_rate if transfer_rate > 0 else None
    ledger = read_json(args.root / "download_checksum_ledger.json")
    ledger_objects = ledger.get("objects", {})
    if not isinstance(ledger_objects, dict):
        ledger_objects = {}
    dataset_progress: dict[str, dict[str, Any]] = {}
    for item in manifest.get("objects", []):
        relative = str(item.get("relative_path", ""))
        dataset = relative.split("/", 1)[0]
        if not dataset:
            continue
        row = dataset_progress.setdefault(
            dataset,
            {
                "dataset": dataset,
                "expected_objects": 0,
                "verified_objects": 0,
                "expected_bytes": 0,
                "verified_bytes": 0,
                "active_transfers": 0,
            },
        )
        expected_bytes_for_object = int(item.get("bytes", 0))
        row["expected_objects"] += 1
        row["expected_bytes"] += expected_bytes_for_object
        entry = ledger_objects.get(relative, {})
        if (
            int(entry.get("bytes", -1)) == expected_bytes_for_object
            and entry.get("official_md5_base64")
            and entry.get("official_md5_base64")
            == entry.get("local_md5_base64")
        ):
            row["verified_objects"] += 1
            row["verified_bytes"] += expected_bytes_for_object
    for item in download.get("active", []):
        dataset = str(item.get("relative_path", "")).split("/", 1)[0]
        if dataset in dataset_progress:
            dataset_progress[dataset]["active_transfers"] += 1
    for row in dataset_progress.values():
        row["expected_gib"] = round(row["expected_bytes"] / 2**30, 3)
        row["verified_gib"] = round(row["verified_bytes"] / 2**30, 3)
        row["object_progress_percent"] = (
            row["verified_objects"] / max(row["expected_objects"], 1) * 100.0
        )
        row["byte_progress_percent"] = (
            row["verified_bytes"] / max(row["expected_bytes"], 1) * 100.0
        )
    live_checksum_verified = sum(
        int(row["verified_objects"]) for row in dataset_progress.values()
    )
    live_verified_bytes = sum(
        int(row["verified_bytes"]) for row in dataset_progress.values()
    )
    verification_public = dict(verification)
    verification_public["status"] = (
        "verified"
        if verification.get("status") == "verified"
        else "verifying_while_downloading"
    )
    verification_public["expected_objects"] = int(
        verification.get("expected_objects") or manifest.get("object_count", 0)
    )
    verification_public["complete_objects"] = max(
        int(verification.get("complete_objects", 0)),
        int(download.get("completed_objects", 0)),
    )
    verification_public["expected_bytes"] = expected
    verification_public["complete_bytes"] = max(
        int(verification.get("complete_bytes", 0)), live_verified_bytes
    )
    verification_public["md5_verified_bytes"] = live_verified_bytes
    verification_public["md5_verified_objects"] = max(
        int(verification.get("md5_verified_objects", 0)),
        live_checksum_verified,
    )
    verification_public["md5_progress_percent"] = (
        verification_public["md5_verified_objects"]
        / max(verification_public["expected_objects"], 1)
        * 100.0
    )
    verification_public["missing_count"] = max(
        0,
        verification_public["expected_objects"]
        - verification_public["complete_objects"],
    )
    download_marker_status = marker_status(args.root, "download")
    training_marker_status = marker_status(args.root, "training")
    synthesis_marker_status = marker_status(args.root, "synthesis")
    download_marker = download_marker_status.get("completed_at")
    training_marker = training_marker_status.get("completed_at")
    synthesis_marker = synthesis_marker_status.get("completed_at")
    training_complete = bool(training_marker_status.get("valid"))
    synthesis_complete = bool(synthesis_marker_status.get("valid"))
    final_page_qa = (
        validate_page_qa(args.root, Path("/Users/avalok/work/Q-TAIL-MVP"))
        if synthesis_complete and (args.root / "final_page_qa.json").is_file()
        else {"valid": False, "generated_at": None, "error": "waiting for synthesis"}
    )
    if synthesis_complete:
        stage = "complete"
    elif matching_processes("qtail_run_openx_synthesis_with_status.py"):
        stage = "synthesizing"
    elif training_complete:
        stage = "training_complete"
    elif matching_processes("qtail_run_openx_training_with_status.py"):
        stage = "training"
    elif verification.get("status") == "verified":
        stage = "download_verified"
    elif matching_processes("qtail_parallel_gcs_download.py"):
        stage = "downloading"
    else:
        stage = "queued_or_restarting"
    downloader_running = bool(matching_processes("qtail_parallel_gcs_download.py"))
    trainer_running = bool(matching_processes("qtail_run_openx_training_with_status.py"))
    synthesizer_running = bool(matching_processes("qtail_run_openx_synthesis_with_status.py"))
    pipeline_gates = {
        "download": {
            "status": "pass" if download_marker else "run" if downloader_running else "wait",
            "completed_at": download_marker,
            "evidence": f"{int(download.get('completed_objects', 0))}/{int(manifest.get('object_count', 0))} objects",
        },
        "md5": {
            "status": "pass" if verification.get("status") == "verified" else "run" if live_checksum_verified else "wait",
            "completed_at": verification.get("generated_at") if verification.get("status") == "verified" else None,
            "evidence": f"{verification_public['md5_verified_objects']}/{verification_public['expected_objects']} official MD5",
        },
        "prewarm": {
            "status": (
                "pass"
                if prewarm.get("full_manifest_complete")
                else "run"
                if int(prewarm.get("cached_objects", 0)) > 0
                else "wait"
            ),
            "completed_at": (
                prewarm.get("generated_at")
                if prewarm.get("full_manifest_complete")
                else None
            ),
            "evidence": f"{int(prewarm.get('cached_objects', 0))}/{int(prewarm.get('expected_objects') or 0)} TFRecord cache · failed {int(prewarm.get('failed_objects', 0))}",
        },
        "training": {
            "status": "pass" if training_complete else "run" if trainer_running else "wait",
            "completed_at": training_marker,
            "evidence": f"{training.get('steps') or 0}/{training.get('steps_target') or 20000} optimizer steps",
        },
        "synthesis": {
            "status": "pass" if synthesis_complete else "run" if synthesizer_running else "wait",
            "completed_at": synthesis_marker,
            "evidence": (
                "validated delivery package"
                if synthesis_complete
                else synthesis_runtime.get("phase") or "training-gated"
            ),
        },
    }
    checkpoint_path = args.root / "training" / "qtail_allocation_head.pt"
    checkpoint_expected_sha = str(
        training_report.get("model_artifact", {}).get("sha256") or ""
    )
    checkpoint_actual_sha = (
        file_sha256(checkpoint_path) if training_complete else None
    )
    required_synthesis_artifacts = [
        "qtail_service_delivery_report.json",
        "qtail_service_synthetic_plan.csv",
        "qtail_synthetic_data.csv",
        "qtail_service_model_card.json",
        "qtail_data_engine_report.json",
        "README_QTAIL_DELIVERY.md",
        "qtail_delivery_package.zip",
    ]
    synthesis_artifacts = {
        name: (args.root / "synthesis" / name).is_file()
        for name in required_synthesis_artifacts
    }
    preflight_artifacts = {
        name: (preflight_root / "synthesis" / name).is_file()
        for name in required_synthesis_artifacts
    }
    preflight_checkpoint_path = (
        preflight_root / "training" / "qtail_allocation_head.pt"
    )
    preflight_checkpoint_expected_sha = str(
        preflight_training.get("model_artifact", {}).get("sha256") or ""
    )
    preflight_checkpoint_actual_sha = file_sha256(preflight_checkpoint_path)
    preflight_valid = bool(
        preflight_training_runtime.get("status") == "complete"
        and int(preflight_training_runtime.get("returncode", -1)) == 0
        and preflight_training.get("status") == "complete"
        and int(preflight_training.get("steps", 0)) == 5
        and int(preflight_training.get("shard_count", 0)) == 100
        and float(
            preflight_training.get("trajectory_evidence", {}).get(
                "record_parse_rate", 0.0
            )
        )
        == 1.0
        and preflight_checkpoint_expected_sha
        and preflight_checkpoint_actual_sha == preflight_checkpoint_expected_sha
        and preflight_optimizer.get("status") == "phase_complete"
        and preflight_optimizer.get("phase") == "qtail"
        and int(preflight_optimizer.get("overall_completed_updates", 0)) == 10
        and int(preflight_optimizer.get("overall_target_updates", 0)) == 10
        and (preflight_root / "training" / "resume_checkpoints" / "source.pt").is_file()
        and (preflight_root / "training" / "resume_checkpoints" / "qtail.pt").is_file()
        and preflight_synthesis_runtime.get("status") == "complete"
        and int(preflight_synthesis_runtime.get("returncode", -1)) == 0
        and preflight_synthesis.get("customer_package", {})
        .get("validation", {})
        .get("valid")
        and all(preflight_artifacts.values())
    )
    route_guard = download.get("route_guard", {})
    completion_checks = [
        {
            "id": "official_selection_manifest",
            "label": "1 TiB 官方对象清单",
            "passed": bool(expected > 0 and int(manifest.get("object_count", 0)) == 3126),
            "evidence": f"{int(manifest.get('object_count', 0))} objects · {round(expected / 2**30, 3)} GiB",
        },
        {
            "id": "direct_transport_isolation",
            "label": "UniClash 在线且下载物理网卡直连",
            "passed": bool(
                guard.get("uniclash", {}).get("core_running")
                and not guard.get("uniclash", {}).get("tun_enabled")
                and route_guard.get("status") == "passed"
                and route_guard.get("curl_interface_bound")
            ),
            "evidence": f"Core {bool(guard.get('uniclash', {}).get('core_running'))} · TUN {guard.get('uniclash', {}).get('tun_enabled')} · {route_guard.get('expected_interface', '-')}",
        },
        {
            "id": "cross_restart_supervision",
            "label": "跨重启自动恢复监督",
            "passed": bool(
                supervision.get("loaded")
                and supervision.get("plist_sha256_match")
                and supervision.get("interval_seconds") == 300
                and web_page.get("http_status") == 200
            ),
            "evidence": f"launchd loaded {supervision.get('loaded')} · interval {supervision.get('interval_seconds')}s · plist {'MATCH' if supervision.get('plist_sha256_match') else 'MISMATCH'} · page HTTP {web_page.get('http_status') or 'DOWN'}",
        },
        {
            "id": "download_and_official_md5",
            "label": "下载完成与官方 MD5",
            "passed": bool(
                download_marker
                and verification.get("status") == "verified"
                and verification_public["md5_verified_objects"]
                == verification_public["expected_objects"]
            ),
            "evidence": f"{verification_public['md5_verified_objects']}/{verification_public['expected_objects']} official MD5",
        },
        {
            "id": "model_training",
            "label": "20,000 步模型训练与 checkpoint",
            "passed": bool(
                training_marker
                and training_report.get("status") == "complete"
                and int(training_report.get("steps", 0)) == 20000
                and checkpoint_expected_sha
                and checkpoint_actual_sha == checkpoint_expected_sha
            ),
            "evidence": f"{int(training_report.get('steps', 0))}/20000 steps · checkpoint {'MATCH' if checkpoint_actual_sha and checkpoint_actual_sha == checkpoint_expected_sha else 'WAIT'}",
        },
        {
            "id": "long_tail_synthesis",
            "label": "PT 重尾合成数据与交付包",
            "passed": bool(synthesis_complete and all(synthesis_artifacts.values())),
            "evidence": f"{sum(synthesis_artifacts.values())}/{len(synthesis_artifacts)} artifacts · validation {bool(synthesis.get('customer_package', {}).get('validation', {}).get('valid'))}",
        },
        {
            "id": "final_page_projection",
            "label": "最终页面与公开交付链接",
            "passed": bool(final_page_qa.get("valid")),
            "evidence": (
                "page + status + 7 artifacts HTTP 200 · SHA MATCH"
                if final_page_qa.get("valid")
                else str(final_page_qa.get("error") or "WAIT")
            ),
        },
    ]
    completion_passed = sum(bool(item["passed"]) for item in completion_checks)
    if synthesis_complete:
        stage = (
            "complete"
            if completion_passed == len(completion_checks)
            else "completion_audit_failed"
        )
    history_path = args.root / "status_history.json"
    history_payload = read_json(history_path)
    history_samples = history_payload.get("samples", [])
    if not isinstance(history_samples, list):
        history_samples = []
    history_sample = {
        "generated_at": generated_at,
        "stage": stage,
        "progress_gib": round(progress / 2**30, 3),
        "progress_percent": progress / max(expected, 1) * 100.0,
        "md5_verified_objects": verification_public["md5_verified_objects"],
        "expected_objects": verification_public["expected_objects"],
        "prewarm_cached_objects": int(prewarm.get("cached_objects", 0)),
        "prewarm_verified_objects": int(prewarm.get("verified_objects", 0)),
        "prewarm_failed_objects": int(prewarm.get("failed_objects", 0)),
        "transfer_rate_mib_s": round(transfer_rate / 2**20, 3),
        "route_status": route_guard.get("status"),
        "route_interface": route_guard.get("expected_interface"),
        "completion_passed": completion_passed,
        "completion_total": len(completion_checks),
    }
    append_history = not history_samples
    if history_samples:
        previous_sample = history_samples[-1]
        try:
            history_elapsed = (
                datetime.fromisoformat(generated_at)
                - datetime.fromisoformat(str(previous_sample.get("generated_at")))
            ).total_seconds()
        except (TypeError, ValueError):
            history_elapsed = 301.0
        append_history = bool(
            stage != previous_sample.get("stage")
            or history_sample["progress_gib"]
            - float(previous_sample.get("progress_gib", 0.0))
            >= 0.25
            or history_sample["md5_verified_objects"]
            - int(previous_sample.get("md5_verified_objects", 0))
            >= 10
            or history_sample["prewarm_cached_objects"]
            - int(previous_sample.get("prewarm_cached_objects", 0))
            >= 10
            or history_elapsed >= 300.0
        )
    if append_history:
        history_samples.append(history_sample)
        history_samples = history_samples[-1000:]
        atomic_json(
            history_path,
            {
                "format_version": "qtail_openx_1t_status_history_v2",
                "generated_at": generated_at,
                "samples": history_samples,
            },
        )
    payload = {
        "format_version": "qtail_openx_1t_status_v2",
        "generated_at": generated_at,
        "status": (
            "complete"
            if completion_passed == len(completion_checks)
            else "running"
        ),
        "stage": stage,
        "selection": {
            "target_bytes": int(catalog.get("target_bytes", 0)),
            "selected_bytes": expected,
            "selected_gib": round(expected / 2**30, 3),
            "dataset_count": int(manifest.get("dataset_count", 0)),
            "datasets": manifest.get("datasets", []),
            "dataset_progress": list(dataset_progress.values()),
            "object_count": int(manifest.get("object_count", 0)),
        },
        "download": {
            "status": download.get("status", "waiting"),
            "progress_bytes": progress,
            "progress_gib": round(progress / 2**30, 3),
            "progress_percent": progress / max(expected, 1) * 100.0,
            "remaining_bytes": remaining_bytes,
            "transfer_rate_bps": round(transfer_rate, 3),
            "transfer_rate_mib_s": round(transfer_rate / 2**20, 3),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "completed_objects": int(download.get("completed_objects", 0)),
            "partial_bytes": int(download.get("partial_bytes", 0)),
            "active_transfers": len(download.get("active", [])),
            "worker_count": int(download.get("workers", 0)),
            "failed_objects": len(download.get("failures", {})),
            "transport_stalled_seconds": float(
                download.get("transport_stalled_seconds", 0.0)
            ),
            "checksum_verified_objects": live_checksum_verified,
            "route_guard": download.get("route_guard", {}),
            "disk_headroom": download.get("disk_headroom", {}),
        },
        "verification": verification_public,
        "prewarm": {
            "status": prewarm.get("status", "waiting_for_verified_shards"),
            "feature_extractor_version": prewarm.get("feature_extractor_version"),
            "records_per_shard": prewarm.get("records_per_shard", 4),
            "expected_objects": int(
                prewarm.get("expected_objects") or manifest.get("object_count", 0)
            ),
            "verified_objects": int(
                prewarm.get("verified_objects") or live_checksum_verified
            ),
            "cached_objects": int(prewarm.get("cached_objects", 0)),
            "parsed_objects": int(prewarm.get("parsed_objects", 0)),
            "failed_objects": int(prewarm.get("failed_objects", 0)),
            "records_decoded": int(prewarm.get("records_decoded", 0)),
            "active_relative_path": prewarm.get("active_relative_path", ""),
            "verified_cache_percent": float(
                prewarm.get("verified_cache_percent", 0.0)
            ),
            "full_manifest_complete": bool(
                prewarm.get("full_manifest_complete")
            ),
            "status_path": "results/openx_1t_expansion/prewarm_status.json",
            "cache_usage_path": "results/openx_1t_expansion/training/feature_cache_usage.json",
        },
        "training": {
            "status": training.get("status", "waiting_for_verified_download"),
            "steps": training.get("steps"),
            "steps_target": training.get("steps_target", 20000),
            "phase": training.get("phase"),
            "elapsed_seconds": training.get("elapsed_seconds"),
            "child_pid": training.get("child_pid"),
            "returncode": training.get("returncode"),
            "shard_count": training.get("shard_count"),
            "total_gib": training.get("total_gib"),
            "effect_metrics": training.get("effect_metrics"),
            "model_artifact": training.get("model_artifact"),
            "report": "results/openx_1t_expansion/training/openx_demo_training_report.json",
            "rows": "results/openx_1t_expansion/training/openx_shard_training_rows.csv",
            "optimizer_progress": {
                "status": optimizer_progress.get("status", "waiting"),
                "phase": optimizer_progress.get("phase"),
                "step": int(optimizer_progress.get("step", 0)),
                "steps_target": int(optimizer_progress.get("steps_target", 20000)),
                "overall_completed_updates": int(
                    optimizer_progress.get("overall_completed_updates", 0)
                ),
                "overall_target_updates": int(
                    optimizer_progress.get("overall_target_updates", 40000)
                ),
                "resumed": bool(optimizer_progress.get("resumed")),
                "checkpoint_interval": int(
                    optimizer_progress.get("checkpoint_interval", 1000)
                ),
                "checkpoint_path": optimizer_progress.get("checkpoint_path"),
                "kl": optimizer_progress.get("kl"),
                "generated_at": optimizer_progress.get("generated_at"),
            },
        },
        "synthesis": {
            "status": "complete" if synthesis_complete else synthesis_runtime.get("status", "waiting_for_training"),
            "capability": "allocation_and_scenario_specification",
            "raw_sample_generator_status": "not_trained",
            "raw_sample_output": False,
            "claim_boundary": (
                "The completed artifacts are PT-heavy-tail allocation rows and "
                "scenario specifications. They are not newly rendered sensor, "
                "action, or robot-trajectory samples."
            ),
            "phase": synthesis_runtime.get("phase"),
            "elapsed_seconds": synthesis_runtime.get("elapsed_seconds"),
            "child_pid": synthesis_runtime.get("child_pid"),
            "returncode": synthesis_runtime.get("returncode"),
            "effect_summary": synthesis.get("effect_summary", {}),
            "validation": synthesis.get("customer_package", {}).get("validation", {}),
            "delivery_report": "results/openx_1t_expansion/synthesis/qtail_service_delivery_report.json",
            "synthetic_plan": "results/openx_1t_expansion/synthesis/qtail_service_synthetic_plan.csv",
            "package_zip": "results/openx_1t_expansion/synthesis/qtail_delivery_package.zip",
            "readme": "results/openx_1t_expansion/synthesis/README_QTAIL_DELIVERY.md",
        },
        "handoff_preflight": {
            "status": "pass" if preflight_valid else "wait",
            "claim_boundary": "Engineering handoff rehearsal on a bounded real-shard subset; not the final 1 TiB result.",
            "training_runtime_status": preflight_training_runtime.get("status"),
            "training_returncode": preflight_training_runtime.get("returncode"),
            "steps": int(preflight_training.get("steps", 0)),
            "shards": int(preflight_training.get("shard_count", 0)),
            "record_parse_rate": float(
                preflight_training.get("trajectory_evidence", {}).get(
                    "record_parse_rate", 0.0
                )
            ),
            "cached_rows": int(preflight_cache.get("cached_rows", 0)),
            "fresh_rows": int(preflight_cache.get("fresh_rows", 0)),
            "checkpoint_sha256_match": bool(
                preflight_checkpoint_expected_sha
                and preflight_checkpoint_actual_sha
                == preflight_checkpoint_expected_sha
            ),
            "optimizer_updates": int(
                preflight_optimizer.get("overall_completed_updates", 0)
            ),
            "optimizer_updates_target": int(
                preflight_optimizer.get("overall_target_updates", 0)
            ),
            "resume_checkpoints": sum(
                path.is_file()
                for path in (
                    preflight_root / "training" / "resume_checkpoints" / "source.pt",
                    preflight_root / "training" / "resume_checkpoints" / "qtail.pt",
                )
            ),
            "synthesis_runtime_status": preflight_synthesis_runtime.get("status"),
            "synthesis_returncode": preflight_synthesis_runtime.get("returncode"),
            "validation_valid": bool(
                preflight_synthesis.get("customer_package", {})
                .get("validation", {})
                .get("valid")
            ),
            "winner": preflight_synthesis.get("customer_package", {})
            .get("validation", {})
            .get("winner"),
            "artifacts_passed": sum(preflight_artifacts.values()),
            "artifacts_total": len(preflight_artifacts),
            "training_report": "results/openx_1t_expansion/handoff_preflight/training/openx_demo_training_report.json",
            "cache_usage": "results/openx_1t_expansion/handoff_preflight/training/feature_cache_usage.json",
            "optimizer_progress": "results/openx_1t_expansion/handoff_preflight/training/optimizer_progress.json",
            "delivery_report": "results/openx_1t_expansion/handoff_preflight/synthesis/qtail_service_delivery_report.json",
            "package_zip": "results/openx_1t_expansion/handoff_preflight/synthesis/qtail_delivery_package.zip",
        },
        "pipeline_gates": pipeline_gates,
        "completion": {
            "status": "complete" if completion_passed == len(completion_checks) else "in_progress",
            "passed": completion_passed,
            "total": len(completion_checks),
            "checks": completion_checks,
            "checkpoint_sha256": {
                "expected": checkpoint_expected_sha or None,
                "actual": checkpoint_actual_sha,
                "match": bool(
                    checkpoint_actual_sha
                    and checkpoint_actual_sha == checkpoint_expected_sha
                ),
            },
            "synthesis_artifacts": synthesis_artifacts,
        },
        "history": {
            "sample_count": len(history_samples),
            "path": "results/openx_1t_expansion/status_history.json",
            "recent": history_samples[-12:],
        },
        "runtime": {
            "pipeline": matching_processes("qtail_openx_1t_pipeline.sh"),
            "downloader": matching_processes("qtail_parallel_gcs_download.py"),
            "prewarm_loop": leaf_processes(
                matching_processes("qtail_openx_1t_prewarm_loop.sh")
            ),
            "prewarmer": matching_processes("qtail_openx_feature_prewarm.py"),
            "trainer": matching_processes("qtail_run_openx_training_with_status.py"),
            "synthesizer": matching_processes("qtail_run_openx_synthesis_with_status.py"),
            "uniclash_core_running": bool(
                guard.get("uniclash", {}).get("core_running")
            ),
            "uniclash_tun_enabled": guard.get("uniclash", {}).get("tun_enabled"),
            "guard_status": guard.get("status"),
            "supervision": supervision,
            "web_page": web_page,
            "stage_markers": {
                "download": download_marker_status,
                "training": training_marker_status,
                "synthesis": synthesis_marker_status,
            },
            "final_page_qa": final_page_qa,
        },
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_tib": round(usage.free / 2**40, 3),
            "remaining_download_bytes": remaining_bytes,
            "projected_free_after_download_bytes": projected_free_after_download,
            "projected_free_after_download_tib": round(
                projected_free_after_download / 2**40, 3
            ),
            "reserve_free_bytes": reserve_free_bytes,
            "projected_headroom_after_download_bytes": (
                projected_headroom_after_download
            ),
            "projected_headroom_after_download_gib": round(
                projected_headroom_after_download / 2**30, 3
            ),
            "projected_capacity_passed": projected_headroom_after_download >= 0,
        },
        "claim_boundary": (
            "The trained artifact is an Open X record-informed long-tail allocation "
            "head, not an end-to-end robot policy."
        ),
        "logs": {"pipeline_tail": tail_lines(args.root / "pipeline.log")},
    }
    atomic_json(args.out, payload)
    print(json.dumps({"status": payload["status"], "stage": stage}))


if __name__ == "__main__":
    main()
