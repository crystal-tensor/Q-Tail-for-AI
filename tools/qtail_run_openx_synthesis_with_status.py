#!/usr/bin/env python3
"""Generate and validate the Open X calibrated long-tail delivery package."""

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
    child_pid: int | None,
    returncode: int | None = None,
) -> None:
    atomic_json(
        path,
        {
            "format_version": "qtail_openx_synthesis_runtime_v1",
            "generated_at": now(),
            "status": status,
            "phase": phase,
            "started_at": started_at,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 1),
            "child_pid": child_pid,
            "returncode": returncode,
        },
    )


def run_phase(
    command: list[str],
    *,
    runtime_path: Path,
    phase: str,
    started_at: str,
    started_monotonic: float,
    interval_seconds: float,
) -> int:
    child = subprocess.Popen(command)
    while child.poll() is None:
        publish(
            runtime_path,
            status="running",
            phase=phase,
            started_at=started_at,
            started_monotonic=started_monotonic,
            child_pid=child.pid,
        )
        time.sleep(max(1.0, interval_seconds))
    return int(child.returncode or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--synthetic-budget", type=float, default=100_000.0)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runtime_path = args.out / "synthesis_runtime_status.json"
    started_at = now()
    started_monotonic = time.monotonic()
    generator_command = [
        sys.executable,
        str(args.generator),
        "--input",
        str(args.input),
        "--out",
        str(args.out),
        "--training-report",
        str(args.training_report),
        "--training-rows",
        str(args.training_rows),
        "--synthetic-budget",
        str(args.synthetic_budget),
        "--top-k",
        str(args.top_k),
    ]
    returncode = run_phase(
        generator_command,
        runtime_path=runtime_path,
        phase="generating_pt_heavy_tail_package",
        started_at=started_at,
        started_monotonic=started_monotonic,
        interval_seconds=args.interval_seconds,
    )
    if returncode == 0:
        validator_command = [
            sys.executable,
            str(args.validator),
            str(args.out / "qtail_data_engine_report.json"),
        ]
        returncode = run_phase(
            validator_command,
            runtime_path=runtime_path,
            phase="validating_same_budget_delivery_package",
            started_at=started_at,
            started_monotonic=started_monotonic,
            interval_seconds=args.interval_seconds,
        )
    publish(
        runtime_path,
        status="complete" if returncode == 0 else "failed",
        phase="synthesis_complete" if returncode == 0 else "synthesis_failed",
        started_at=started_at,
        started_monotonic=started_monotonic,
        child_pid=None,
        returncode=returncode,
    )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
