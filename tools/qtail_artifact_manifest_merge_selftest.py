#!/usr/bin/env python3
"""Positive/negative controls for formal artifact-manifest merging."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from qtail_merge_droid_artifact_manifest import (
    HISTORICAL_OPTIONAL_ARTIFACTS,
    MANIFEST_CONTROL_FILENAMES,
    atomic_write_json,
    build_manifest_payload,
    formal_droid_artifact_paths,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rejection(
    *,
    manifest: dict,
    additions: list[Path],
    formal_root: Path,
    expected_message: str,
) -> bool:
    try:
        build_manifest_payload(
            manifest=manifest,
            additions=additions,
            formal_root=formal_root,
        )
    except ValueError as error:
        return expected_message in str(error)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    controls: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="qtail-manifest-selftest-") as raw:
        root = Path(raw)
        result_root = root / "results"
        result_root.mkdir()
        for path in formal_droid_artifact_paths(result_root):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"artifact:{path.name}".encode())
        missing_optional = result_root / HISTORICAL_OPTIONAL_ARTIFACTS[0]
        manifest = {
            "artifacts": [
                {
                    "path": str(missing_optional),
                    "bytes": 1,
                    "sha256": "0" * 64,
                }
            ]
        }
        payload = build_manifest_payload(
            manifest=manifest,
            additions=[],
            formal_root=result_root,
        )
        controls.append(
            {
                "name": "missing_optional_history_is_pruned",
                "passed": (
                    payload["status"] == "complete"
                    and payload["formal_droid_contract"][
                        "all_required_present"
                    ]
                    is True
                    and payload["formal_droid_contract"][
                        "required_artifact_count"
                    ]
                    == len(formal_droid_artifact_paths(result_root))
                    and len(
                        payload["formal_droid_contract"][
                            "required_artifact_paths"
                        ]
                    )
                    == len(formal_droid_artifact_paths(result_root))
                    and set(
                        payload["formal_droid_contract"][
                            "required_artifact_paths"
                        ]
                    )
                    == {
                        str(path.resolve())
                        for path in formal_droid_artifact_paths(result_root)
                    }
                    and str(missing_optional)
                    not in {
                        entry["path"] for entry in payload["artifacts"]
                    }
                ),
            }
        )

        present_optional = result_root / HISTORICAL_OPTIONAL_ARTIFACTS[-1]
        present_optional.write_bytes(b"bounded-history")
        payload_with_history = build_manifest_payload(
            manifest=payload,
            additions=[],
            formal_root=result_root,
        )
        optional_entry = next(
            (
                entry
                for entry in payload_with_history["artifacts"]
                if entry["path"] == str(present_optional.resolve())
            ),
            None,
        )
        controls.append(
            {
                "name": "present_optional_history_is_retained_and_hashed",
                "passed": (
                    optional_entry is not None
                    and optional_entry["bytes"] == len(b"bounded-history")
                    and len(optional_entry["sha256"]) == 64
                    and str(present_optional.resolve())
                    not in payload_with_history["formal_droid_contract"][
                        "required_artifact_paths"
                    ]
                    and payload_with_history["formal_droid_contract"][
                        "required_artifact_count"
                    ]
                    == len(formal_droid_artifact_paths(result_root))
                ),
            }
        )

        control_paths = [
            result_root / name for name in MANIFEST_CONTROL_FILENAMES
        ]
        for path in control_paths:
            path.write_bytes(b"manifest-control")
        manifest_with_control = {
            **payload_with_history,
            "artifacts": [
                *payload_with_history["artifacts"],
                {
                    "path": str(control_paths[0]),
                    "bytes": len(b"manifest-control"),
                    "sha256": "0" * 64,
                },
            ],
        }
        payload_without_controls = build_manifest_payload(
            manifest=manifest_with_control,
            additions=control_paths[1:],
            formal_root=result_root,
        )
        observed_paths = {
            entry["path"] for entry in payload_without_controls["artifacts"]
        }
        controls.append(
            {
                "name": "manifest_control_files_are_excluded",
                "passed": all(
                    str(path.resolve()) not in observed_paths
                    for path in control_paths
                ),
            }
        )

        required_path = payload["formal_droid_contract"][
            "required_artifact_paths"
        ][0]
        missing_required_entry_manifest = {
            **payload,
            "artifacts": [
                entry
                for entry in payload["artifacts"]
                if entry["path"] != required_path
            ],
        }
        controls.append(
            {
                "name": "required_manifest_membership_drift_is_rejected",
                "passed": rejection(
                    manifest=missing_required_entry_manifest,
                    additions=[],
                    formal_root=result_root,
                    expected_message="manifest set drift",
                ),
            }
        )
        required = formal_droid_artifact_paths(result_root)[0]
        required.unlink()
        controls.append(
            {
                "name": "missing_formal_artifact_is_rejected",
                "passed": rejection(
                    manifest={"artifacts": []},
                    additions=[],
                    formal_root=result_root,
                    expected_message="formal artifact is missing",
                ),
            }
        )
        required.write_bytes(f"artifact:{required.name}".encode())

        outside = root / "outside.bin"
        outside.write_bytes(b"outside")
        escaped_required = formal_droid_artifact_paths(result_root)[1]
        escaped_required.unlink()
        escaped_required.symlink_to(outside)
        controls.append(
            {
                "name": "escaped_symlink_is_rejected",
                "passed": rejection(
                    manifest={"artifacts": []},
                    additions=[],
                    formal_root=result_root,
                    expected_message="must not be a symlink",
                ),
            }
        )
        escaped_required.unlink()
        escaped_required.write_bytes(
            f"artifact:{escaped_required.name}".encode()
        )

        controls.append(
            {
                "name": "dotdot_outside_addition_is_rejected",
                "passed": rejection(
                    manifest={"artifacts": []},
                    additions=[result_root / ".." / outside.name],
                    formal_root=result_root,
                    expected_message="escapes formal root",
                ),
            }
        )

        drifted_manifest = {
            "artifacts": payload["artifacts"],
            "formal_droid_contract": {
                **payload["formal_droid_contract"],
                "required_artifact_paths": payload[
                    "formal_droid_contract"
                ]["required_artifact_paths"][:-1],
            },
        }
        controls.append(
            {
                "name": "required_path_set_drift_is_rejected",
                "passed": rejection(
                    manifest=drifted_manifest,
                    additions=[],
                    formal_root=result_root,
                    expected_message="path set drift",
                ),
            }
        )
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
