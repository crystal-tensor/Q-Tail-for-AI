#!/usr/bin/env python3
"""Publish the ORICO orchestration snapshot as one atomic directory swap."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


AT_FDCWD = -2
RENAME_SWAP = 0x00000002
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  \./(.+)$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def existing_relative_paths(snapshot: Path) -> set[Path]:
    manifest = snapshot / "SHA256SUMS"
    if not manifest.is_file():
        raise SystemExit(f"snapshot manifest is missing: {manifest}")
    paths: set[Path] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = MANIFEST_LINE.fullmatch(line)
        if not match:
            raise SystemExit(
                f"invalid SHA256SUMS line {line_number}: {line!r}"
            )
        relative = Path(match.group(2))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(
                f"unsafe snapshot path on line {line_number}: {relative}"
            )
        paths.add(relative)
    return paths


def build_stage(
    repo_root: Path,
    snapshot: Path,
    stage: Path,
) -> tuple[list[Path], str]:
    paths = existing_relative_paths(snapshot)
    publisher = Path(__file__).resolve()
    try:
        paths.add(publisher.relative_to(repo_root.resolve()))
    except ValueError as error:
        raise SystemExit(
            f"publisher is outside repo root: {publisher}"
        ) from error

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    ordered = sorted(paths, key=lambda item: item.as_posix())
    manifest_lines = []
    for relative in ordered:
        source = repo_root / relative
        if not source.is_file():
            raise SystemExit(f"workspace source is missing: {source}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise SystemExit(
                f"staged source mismatch: {relative} "
                f"{source_hash} != {destination_hash}"
            )
        manifest_lines.append(f"{destination_hash}  ./{relative.as_posix()}")
    manifest = stage / "SHA256SUMS"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return ordered, sha256(manifest)


def atomic_swap(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx_np = libc.renameatx_np
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        RENAME_SWAP,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{left} <-> {right}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    job_root = args.job_root.resolve()
    snapshot = (
        args.snapshot_dir.resolve()
        if args.snapshot_dir
        else job_root / "code" / "qtail_orchestration"
    )
    result_root = job_root / "results" / "qtail_droid_full"
    out = (
        args.out.resolve()
        if args.out
        else result_root / "qtail_orchestration_snapshot_sync_audit.json"
    )
    stage = snapshot.parent / f".{snapshot.name}.stage.{os.getpid()}"
    lock_path = result_root / ".progress_refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    manifest_hash = ""
    payload: dict = {}
    swapped = False
    try:
        paths, manifest_hash = build_stage(repo_root, snapshot, stage)
        with lock_path.open("a+") as progress_lock:
            fcntl.flock(progress_lock.fileno(), fcntl.LOCK_EX)
            atomic_swap(stage, snapshot)
            swapped = True
            published_manifest = snapshot / "SHA256SUMS"
            if sha256(published_manifest) != manifest_hash:
                raise SystemExit(
                    "published snapshot manifest changed after swap"
                )
            verified = 0
            for relative in paths:
                source = repo_root / relative
                published = snapshot / relative
                if sha256(source) != sha256(published):
                    raise SystemExit(
                        f"published workspace parity mismatch: {relative}"
                    )
                verified += 1
            payload = {
                "format_version": (
                    "qtail_orchestration_snapshot_sync_v1"
                ),
                "generated_at": now(),
                "status": "passed",
                "repo_root": str(repo_root),
                "snapshot": str(snapshot),
                "manifest": str(published_manifest),
                "manifest_sha256": manifest_hash,
                "file_count": len(paths),
                "verified_file_count": verified,
                "progress_refresh_lock": str(lock_path),
                "progress_refresh_lock_held_during_swap": True,
                "audit_committed_before_progress_lock_release": True,
                "atomic_directory_swap": (
                    "macos_renameatx_np_RENAME_SWAP"
                ),
                "claim_boundary": (
                    "The snapshot was fully staged and source-verified "
                    "before one same-volume atomic directory swap. The "
                    "published tree was reverified and this audit was "
                    "committed while the public progress projection lock "
                    "remained held. This proves local publication "
                    "consistency, not external timestamping or WORM "
                    "storage."
                ),
            }
            atomic_write_json(out, payload)
        shutil.rmtree(stage)
    finally:
        if stage.exists():
            if swapped:
                shutil.rmtree(stage)
            else:
                shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
