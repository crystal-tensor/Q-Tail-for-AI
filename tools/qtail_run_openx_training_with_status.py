#!/usr/bin/env python3
"""Run the existing Open X trainer while publishing a live runtime ledger."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish(
    path: Path,
    *,
    status: str,
    phase: str,
    started_at: str,
    started_monotonic: float,
    steps: int,
    child_pid: int | None,
    command: list[str],
    returncode: int | None = None,
) -> None:
    atomic_json(
        path,
        {
            "format_version": "qtail_openx_training_runtime_v1",
            "generated_at": now(),
            "status": status,
            "phase": phase,
            "started_at": started_at,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 1),
            "steps_target": steps,
            "child_pid": child_pid,
            "returncode": returncode,
            "command": command,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--records-per-shard", type=int, default=4)
    parser.add_argument("--min-record-parse-rate", type=float, default=0.95)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runtime_path = args.out / "training_runtime_status.json"
    command = [
        sys.executable,
        str(args.trainer),
        "--data-dir",
        str(args.data_dir),
        "--out",
        str(args.out),
        "--steps",
        str(args.steps),
        "--records-per-shard",
        str(args.records_per_shard),
        "--min-record-parse-rate",
        str(args.min_record_parse_rate),
        "--max-shards",
        str(args.max_shards),
        "--wait",
        "0",
    ]
    started_at = now()
    started_monotonic = time.monotonic()
    publish(
        runtime_path,
        status="starting",
        phase="launching_record_informed_trainer",
        started_at=started_at,
        started_monotonic=started_monotonic,
        steps=args.steps,
        child_pid=None,
        command=command,
    )
    child = subprocess.Popen(command)
    while child.poll() is None:
        publish(
            runtime_path,
            status="running",
            phase="feature_extraction_and_source_qtail_optimization",
            started_at=started_at,
            started_monotonic=started_monotonic,
            steps=args.steps,
            child_pid=child.pid,
            command=command,
        )
        time.sleep(max(1.0, args.interval_seconds))

    returncode = int(child.returncode or 0)
    publish(
        runtime_path,
        status="complete" if returncode == 0 else "failed",
        phase="trainer_complete" if returncode == 0 else "trainer_failed",
        started_at=started_at,
        started_monotonic=started_monotonic,
        steps=args.steps,
        child_pid=child.pid,
        command=command,
        returncode=returncode,
    )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
