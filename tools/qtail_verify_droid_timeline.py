#!/usr/bin/env python3
"""Verify the hash-chained DROID pipeline timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TIMELINE_VERSION = "qtail_droid_pipeline_timeline_v1"
REMOTE_BYTES = 3_700_745_265_151


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def transport_adjudication_valid(timeline_path: Path) -> bool:
    path = timeline_path.with_name(
        "uniclash_transport_guard_adjudication.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    findings = payload.get("findings", [])
    archives = payload.get("preservation", {}).get("archives", [])
    actual_hashes = payload.get("archive_hashes_actual", {})
    return bool(
        payload.get("status") == "adjudicated_transport_epochs_v6"
        and isinstance(findings, list)
        and len(findings) >= 5
        and all(
            finding.get("data_transfer_violation") is False
            for finding in findings
        )
        and isinstance(archives, list)
        and len(archives) >= 5
        and all(
            archive.get("data_transfer_violation") is False
            and archive.get("sha256")
            == actual_hashes.get(archive.get("path"))
            for archive in archives
        )
        and any(
            archive.get("policy_pause") is True for archive in archives
        )
    )


def transport_evidence_scope(
    timeline_path: Path,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    adjudication_path = timeline_path.with_name(
        "uniclash_transport_guard_adjudication.json"
    )
    try:
        adjudication = json.loads(
            adjudication_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unavailable",
            "claim_boundary": (
                "No adjudicated transport epoch index was available. "
                "Timeline transport claims must remain withheld."
            ),
        }

    archives = adjudication.get("preservation", {}).get("archives", [])
    actual_hashes = adjudication.get("archive_hashes_actual", {})
    epochs: list[dict[str, Any]] = []
    coverage_gap_epochs = 0
    for archive in archives if isinstance(archives, list) else []:
        archive_path = Path(str(archive.get("path", "")))
        if (
            not archive_path.is_file()
            or archive.get("sha256") != actual_hashes.get(str(archive_path))
            or file_sha256(archive_path) != archive.get("sha256")
        ):
            continue
        try:
            archived_guard = json.loads(
                archive_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        cumulative = archived_guard.get("cumulative", {})
        first_at = cumulative.get("first_sample_at")
        last_at = cumulative.get("last_sample_at")
        if first_at and last_at:
            epochs.append(
                {
                    "name": archive_path.name,
                    "first_sample_at": first_at,
                    "last_sample_at": last_at,
                    "coverage_gap": archive.get("coverage_gap") is True,
                    "mutable_current": False,
                }
            )
        if archive.get("coverage_gap") is True:
            coverage_gap_epochs += 1

    current_guard_path = timeline_path.with_name(
        "uniclash_transport_guard.json"
    )
    try:
        current_guard = json.loads(
            current_guard_path.read_text(encoding="utf-8")
        )
        cumulative = current_guard.get("cumulative", {})
        if cumulative.get("first_sample_at") and cumulative.get(
            "last_sample_at"
        ):
            epochs.append(
                {
                    "name": current_guard_path.name,
                    "first_sample_at": cumulative["first_sample_at"],
                    "last_sample_at": cumulative["last_sample_at"],
                    "coverage_gap": False,
                    "mutable_current": True,
                }
            )
    except (OSError, json.JSONDecodeError):
        pass

    parsed_epochs: list[tuple[datetime, datetime, dict[str, Any]]] = []
    for epoch in epochs:
        try:
            parsed_epochs.append(
                (
                    datetime.fromisoformat(str(epoch["first_sample_at"])),
                    datetime.fromisoformat(str(epoch["last_sample_at"])),
                    epoch,
                )
            )
        except (TypeError, ValueError):
            continue
    parsed_epochs.sort(key=lambda item: item[0])
    if not parsed_epochs:
        return {
            "status": "unavailable",
            "adjudication": str(adjudication_path),
            "claim_boundary": (
                "No hash-verified transport epoch supplied a valid time range."
            ),
        }

    earliest_guard = parsed_epochs[0][0]
    timeline_started: datetime | None = None
    timeline_start_physical_bytes: int | None = None
    physical_bytes_at_or_before_first_guard: int | None = None
    max_physical_bytes_before_first_guard: int | None = None
    for sample in samples:
        try:
            sample_at = datetime.fromisoformat(str(sample["generated_at"]))
        except (KeyError, TypeError, ValueError):
            continue
        physical_bytes = optional_int(
            sample.get("download", {}).get("physical_bytes")
        )
        if timeline_started is None:
            timeline_started = sample_at
            timeline_start_physical_bytes = physical_bytes
        if sample_at <= earliest_guard and physical_bytes is not None:
            physical_bytes_at_or_before_first_guard = physical_bytes
            max_physical_bytes_before_first_guard = max(
                max_physical_bytes_before_first_guard or 0,
                physical_bytes,
            )

    inter_epoch_gaps: list[dict[str, Any]] = []
    previous_end: datetime | None = None
    previous_name: str | None = None
    for start, end, epoch in parsed_epochs:
        if previous_end is not None and start > previous_end:
            inter_epoch_gaps.append(
                {
                    "after": previous_name,
                    "before": epoch["name"],
                    "seconds": (start - previous_end).total_seconds(),
                }
            )
        if previous_end is None or end > previous_end:
            previous_end = end
            previous_name = epoch["name"]

    pre_guard_seconds = (
        (earliest_guard - timeline_started).total_seconds()
        if timeline_started is not None and earliest_guard > timeline_started
        else 0.0
    )
    pre_guard_net_physical_byte_change = None
    if (
        timeline_start_physical_bytes is not None
        and physical_bytes_at_or_before_first_guard is not None
    ):
        pre_guard_net_physical_byte_change = (
            physical_bytes_at_or_before_first_guard
            - timeline_start_physical_bytes
        )
    return {
        "status": (
            "partial_history_disclosed"
            if pre_guard_seconds > 0
            or coverage_gap_epochs > 0
            or inter_epoch_gaps
            else "complete"
        ),
        "timeline_started_at": (
            timeline_started.isoformat() if timeline_started else None
        ),
        "earliest_guard_sample_at": earliest_guard.isoformat(),
        "pre_guard_duration_seconds": pre_guard_seconds,
        "timeline_start_physical_bytes": timeline_start_physical_bytes,
        "physical_bytes_at_or_before_first_guard": (
            physical_bytes_at_or_before_first_guard
        ),
        "max_physical_bytes_before_first_guard": (
            max_physical_bytes_before_first_guard
        ),
        "pre_guard_net_physical_byte_change": (
            pre_guard_net_physical_byte_change
        ),
        "pre_guard_route_evidence_available": pre_guard_seconds <= 0,
        "hash_verified_archived_epochs": sum(
            item[2].get("mutable_current") is False for item in parsed_epochs
        ),
        "classifier_coverage_gap_epochs": coverage_gap_epochs,
        "inter_epoch_gaps": inter_epoch_gaps,
        "max_inter_epoch_gap_seconds": max(
            (item["seconds"] for item in inter_epoch_gaps),
            default=0.0,
        ),
        "claim_boundary": (
            "This discloses when retained transport-guard evidence begins, "
            "known classifier coverage-gap epochs, and gaps between preserved "
            "epochs. Physical-byte change before the first guard sample is a "
            "filesystem observation, not proof of route or provider usage. "
            "Socket sampling cannot prove the absence of unobserved traffic."
        ),
    }


def verify_timeline(path: Path, require_final: bool) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "timeline": str(path),
            "errors": [f"timeline is unreadable: {error}"],
        }
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "timeline": str(path),
            "errors": ["timeline root must be an object"],
        }

    samples = payload.get("samples", [])
    if payload.get("version") != TIMELINE_VERSION:
        errors.append("timeline version mismatch")
    if payload.get("retention") != "full_pipeline_history":
        errors.append("timeline retention contract mismatch")
    if not isinstance(samples, list) or not samples:
        errors.append("timeline has no samples")
        samples = []
    if int(payload.get("sample_count", -1)) != len(samples):
        errors.append("timeline sample_count mismatch")

    previous_hash: str | None = None
    previous_time: datetime | None = None
    full_samples = 0
    clean_transport_samples = 0
    first_transport_sample_at: str | None = None
    last_transport_sample_at: str | None = None
    previous_transport_time: datetime | None = None
    previous_guard_samples: int | None = None
    max_transport_gap_seconds = 0.0
    guard_sample_resets = 0
    freshness_contract_started = False
    fresh_guard_samples = 0
    adjudicated_policy_pause_samples = 0
    vpn_route_violation_samples = 0
    guard_process_count_anomaly_samples = 0
    adjudication_valid = transport_adjudication_valid(path)
    legacy_physical_byte_decrease_events = 0
    full_completed_object_decrease_events = 0
    full_verified_object_decrease_events = 0
    full_checksum_error_samples = 0
    feature_pass_reset_events = 0
    committed_feature_counter_decrease_events = 0
    previous_legacy_physical_bytes: int | None = None
    previous_completed_objects: int | None = None
    previous_verified_objects: int | None = None
    previous_feature_processed_shards: int | None = None
    for sequence, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, dict):
            errors.append(f"sample {sequence} is not an object")
            continue
        sample = dict(raw_sample)
        observed_hash = sample.pop("sample_sha256", None)
        if sample.get("sequence") != sequence:
            errors.append(f"sample {sequence} sequence mismatch")
        if sample.get("previous_sample_sha256") != previous_hash:
            errors.append(f"sample {sequence} previous hash mismatch")
        expected_hash = canonical_sha256(sample)
        if observed_hash != expected_hash:
            errors.append(f"sample {sequence} hash mismatch")
        try:
            generated_at = datetime.fromisoformat(
                str(sample["generated_at"])
            )
            if previous_time is not None and generated_at < previous_time:
                errors.append(f"sample {sequence} timestamp moved backwards")
            previous_time = generated_at
        except (KeyError, TypeError, ValueError):
            errors.append(f"sample {sequence} timestamp is invalid")
        previous_hash = observed_hash
        if sample.get("kind") == "legacy_download_sample":
            physical_bytes = optional_int(
                sample.get("download", {}).get("physical_bytes")
            )
            if (
                physical_bytes is not None
                and previous_legacy_physical_bytes is not None
                and physical_bytes < previous_legacy_physical_bytes
            ):
                legacy_physical_byte_decrease_events += 1
            if physical_bytes is not None:
                previous_legacy_physical_bytes = physical_bytes
        elif sample.get("kind") == "full_pipeline_sample":
            full_samples += 1
            download = sample.get("download", {})
            checksums = sample.get("object_checksums", {})
            features = sample.get("feature_extraction", {})
            completed_objects = optional_int(
                download.get("completed_objects")
            )
            verified_objects = optional_int(
                checksums.get("verified_objects")
            )
            checksum_errors = optional_int(
                checksums.get("checksum_errors")
            )
            feature_processed_shards = optional_int(
                features.get("processed_shards")
            )
            feature_counter_semantics = features.get(
                "counter_semantics"
            )
            if (
                completed_objects is not None
                and previous_completed_objects is not None
                and completed_objects < previous_completed_objects
            ):
                full_completed_object_decrease_events += 1
            if (
                verified_objects is not None
                and previous_verified_objects is not None
                and verified_objects < previous_verified_objects
            ):
                full_verified_object_decrease_events += 1
            if checksum_errors is not None and checksum_errors > 0:
                full_checksum_error_samples += 1
            if (
                feature_processed_shards is not None
                and previous_feature_processed_shards is not None
                and feature_processed_shards
                < previous_feature_processed_shards
            ):
                if (
                    feature_counter_semantics
                    == "monotonic_committed_prewarm_snapshot_v1"
                ):
                    committed_feature_counter_decrease_events += 1
                    errors.append(
                        "committed feature counter decreased at "
                        f"sample {sequence}"
                    )
                else:
                    feature_pass_reset_events += 1
            if completed_objects is not None:
                previous_completed_objects = completed_objects
            if verified_objects is not None:
                previous_verified_objects = verified_objects
            if feature_processed_shards is not None:
                previous_feature_processed_shards = (
                    feature_processed_shards
                )
            transport = sample.get("transport_isolation", {})
            network_route_clean = bool(
                transport.get("status") in {"passed", "passed_idle"}
                and transport.get("core_running") is True
                and transport.get("tun_enabled") is False
                and int(transport.get("guard_samples", 0)) > 0
                and int(
                    transport.get(
                        "forbidden_socket_observations",
                        -1,
                    )
                )
                == 0
                and int(
                    transport.get("wrong_route_observations", -1)
                )
                == 0
            )
            blocked_samples = int(
                transport.get("blocked_samples", -1)
            )
            if (
                int(
                    transport.get(
                        "forbidden_socket_observations",
                        -1,
                    )
                )
                > 0
                or int(
                    transport.get("wrong_route_observations", -1)
                )
                > 0
            ):
                vpn_route_violation_samples += 1
            policy_pause_adjudicated = bool(
                network_route_clean
                and blocked_samples > 0
                and adjudication_valid
            )
            transport_clean = bool(
                network_route_clean
                and (
                    blocked_samples == 0
                    or policy_pause_adjudicated
                )
            )
            if policy_pause_adjudicated:
                adjudicated_policy_pause_samples += 1
            has_freshness_evidence = "guard_age_seconds" in transport
            guard_process_count = int(
                transport.get("guard_process_count", -1)
            )
            if has_freshness_evidence and guard_process_count != 1:
                guard_process_count_anomaly_samples += 1
            if has_freshness_evidence:
                freshness_contract_started = True
            freshness_clean = bool(
                not freshness_contract_started
                or (
                    has_freshness_evidence
                    and isinstance(
                        transport.get("guard_age_seconds"), (int, float)
                    )
                    and float(transport["guard_age_seconds"]) <= 10.0
                    and bool(transport.get("guard_generated_at"))
                )
            )
            if freshness_clean and has_freshness_evidence:
                fresh_guard_samples += 1
            if not freshness_clean:
                transport_clean = False
                errors.append(
                    f"sample {sequence} guard freshness/process contract failed"
                )
            if transport_clean:
                clean_transport_samples += 1
            else:
                errors.append(
                    f"sample {sequence} transport isolation is not clean"
                )
            if first_transport_sample_at is None:
                first_transport_sample_at = sample.get("generated_at")
            last_transport_sample_at = sample.get("generated_at")
            try:
                transport_time = datetime.fromisoformat(
                    str(sample["generated_at"])
                )
                if previous_transport_time is not None:
                    max_transport_gap_seconds = max(
                        max_transport_gap_seconds,
                        (
                            transport_time - previous_transport_time
                        ).total_seconds(),
                    )
                previous_transport_time = transport_time
            except (KeyError, TypeError, ValueError):
                pass
            guard_samples = int(transport.get("guard_samples", 0))
            if (
                previous_guard_samples is not None
                and guard_samples < previous_guard_samples
            ):
                guard_sample_resets += 1
            previous_guard_samples = guard_samples

    if samples:
        if payload.get("chain_head_sha256") != previous_hash:
            errors.append("timeline chain head mismatch")
        if payload.get("first_generated_at") != samples[0].get(
            "generated_at"
        ):
            errors.append("timeline first timestamp mismatch")
        if payload.get("last_generated_at") != samples[-1].get(
            "generated_at"
        ):
            errors.append("timeline last timestamp mismatch")
    if full_samples < 1:
        errors.append("timeline has no full pipeline samples")

    evidence_scope = transport_evidence_scope(path, samples)
    final_sample = samples[-1] if samples else {}
    if require_final:
        download = final_sample.get("download", {})
        completion = final_sample.get("completion", {})
        transport = final_sample.get("transport_isolation", {})
        checksums = final_sample.get("object_checksums", {})
        features = final_sample.get("feature_extraction", {})
        model = final_sample.get("model_training", {})
        runtime = final_sample.get("runtime", {})
        if payload.get("status") != "in_progress":
            errors.append("final precommit timeline status is not in_progress")
        if final_sample.get("stage") != "final_page_qa":
            errors.append(
                "final precommit timeline sample is not in final_page_qa"
            )
        if int(download.get("physical_bytes", -1)) != REMOTE_BYTES:
            errors.append("final timeline does not bind all official bytes")
        if int(download.get("completed_objects", -1)) != 4_102:
            errors.append("final timeline does not bind 4,102 objects")
        if (
            checksums.get("status") != "complete"
            or int(checksums.get("verified_objects", -1)) != 4_102
            or int(checksums.get("expected_objects", -1)) != 4_102
            or int(checksums.get("checksum_errors", -1)) != 0
            or int(
                checksums.get("stale_or_unexpected_entries", -1)
            )
            != 0
        ):
            errors.append("final timeline checksum closure is incomplete")
        if (
            int(features.get("processed_shards", -1)) != 4_096
            or int(features.get("total_shards", -1)) != 4_096
            or int(
                features.get("record_scan_complete_shards", -1)
            )
            != 4_096
            or int(features.get("records_decoded", -1)) != 187_891
            or features.get("official_record_counts_verified") is not True
            or int(
                features.get("record_count_mismatch_count", -1)
            )
            != 0
            or features.get(
                "unreferenced_cache_excluded_from_training"
            )
            is not True
        ):
            errors.append("final timeline full-record closure is incomplete")
        if model.get("formal_report_status") != "complete":
            errors.append("final timeline formal training is incomplete")
        if (
            int(completion.get("passed_requirements", -1)) != 8
            or int(completion.get("total_requirements", -1)) != 9
        ):
            errors.append(
                "final precommit timeline does not bind sealing 8/9"
            )
        if (
            transport.get("core_running") is not True
            or transport.get("tun_enabled") is not False
            or int(transport.get("blocked_samples", -1)) != 0
            or int(
                transport.get("forbidden_socket_observations", -1)
            )
            != 0
            or int(transport.get("wrong_route_observations", -1)) != 0
        ):
            errors.append("final timeline transport isolation is not clean")
        if (
            runtime.get("healthy") is not True
            or runtime.get("heartbeat_gate_passed") is not True
            or runtime.get("mount_gate_passed") is not True
            or runtime.get("web_gate_passed") is not True
        ):
            errors.append("final timeline runtime health is incomplete")

    return {
        "generated_at": now(),
        "status": "passed" if not errors else "failed",
        "scope": "final_precommit" if require_final else "current",
        "timeline": str(path),
        "timeline_bytes": path.stat().st_size if path.exists() else 0,
        "timeline_sha256": file_sha256(path) if path.exists() else None,
        "version": payload.get("version"),
        "sample_count": len(samples),
        "full_pipeline_samples": full_samples,
        "transport_continuity": {
            "status": (
                (
                    "passed_with_adjudicated_policy_pauses"
                    if adjudicated_policy_pause_samples > 0
                    else "passed"
                )
                if full_samples > 0
                and clean_transport_samples == full_samples
                else "failed"
            ),
            "clean_samples": clean_transport_samples,
            "total_samples": full_samples,
            "first_sample_at": first_transport_sample_at,
            "last_sample_at": last_transport_sample_at,
            "max_gap_seconds": max_transport_gap_seconds,
            "guard_sample_resets": guard_sample_resets,
            "freshness_contract_started": freshness_contract_started,
            "fresh_guard_samples": fresh_guard_samples,
            "adjudication_valid": adjudication_valid,
            "adjudicated_policy_pause_samples": (
                adjudicated_policy_pause_samples
            ),
            "vpn_route_violation_samples": (
                vpn_route_violation_samples
            ),
            "guard_process_count_anomaly_samples": (
                guard_process_count_anomaly_samples
            ),
            "claim_boundary": (
                "This validates every retained full-pipeline timeline sample. "
                "Legacy byte-only samples predate transport fields and are not "
                "silently treated as route observations. Conservatively "
                "blocked samples count separately as adjudicated policy pauses "
                "only when the preserved raw epochs prove zero forbidden "
                "sockets and zero wrong routes. Guard-process count anomalies "
                "remain visible but do not turn a fresh, socket-clean route "
                "sample into a VPN-route violation."
            ),
        },
        "transport_evidence_scope": evidence_scope,
        "data_continuity": {
            "status": (
                "passed"
                if full_completed_object_decrease_events == 0
                and full_verified_object_decrease_events == 0
                and full_checksum_error_samples == 0
                and committed_feature_counter_decrease_events == 0
                else "repair_events_observed"
            ),
            "full_pipeline_samples": full_samples,
            "completed_object_decrease_events": (
                full_completed_object_decrease_events
            ),
            "verified_object_decrease_events": (
                full_verified_object_decrease_events
            ),
            "checksum_error_samples": full_checksum_error_samples,
            "legacy_physical_byte_decrease_events": (
                legacy_physical_byte_decrease_events
            ),
            "feature_pass_reset_events": feature_pass_reset_events,
            "committed_feature_counter_decrease_events": (
                committed_feature_counter_decrease_events
            ),
            "claim_boundary": (
                "Completed-object and official-ledger counts are the "
                "authoritative data-continuity signals. Legacy physical-byte "
                "drops can reflect temporary-part cleanup. Legacy feature-pass "
                "resets reflect a new cache scan starting from zero; counters "
                "tagged monotonic_committed_prewarm_snapshot_v1 are required "
                "never to decrease."
            ),
        },
        "first_generated_at": payload.get("first_generated_at"),
        "last_generated_at": payload.get("last_generated_at"),
        "chain_head_sha256": payload.get("chain_head_sha256"),
        "final_stage": final_sample.get("stage"),
        "final_download_bytes": final_sample.get(
            "download", {}
        ).get("physical_bytes"),
        "final_completed_objects": final_sample.get(
            "download", {}
        ).get("completed_objects"),
        "final_completion": final_sample.get("completion", {}),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()

    result = verify_timeline(args.timeline, args.require_final)
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
