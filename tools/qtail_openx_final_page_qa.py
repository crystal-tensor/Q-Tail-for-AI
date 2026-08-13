#!/usr/bin/env python3
"""Validate the final Open X page projection and its public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qtail_openx_stage_marker import marker_status


FORMAT_VERSION = "qtail_openx_1t_final_page_qa_v1"
REQUIRED_COMPLETION_IDS = (
    "official_selection_manifest",
    "direct_transport_isolation",
    "cross_restart_supervision",
    "download_and_official_md5",
    "model_training",
    "long_tail_synthesis",
)
REQUIRED_PAGE_TOKENS = (
    "Open X 1 TiB 扩展流水线",
    "官方 MD5 门禁",
    "模型训练",
    "长尾分配计划",
    "原始样本生成器",
    "阶段哈希链",
    "本轮分配流水线完成审计",
)
PUBLIC_PATHS = (
    "/qtail-openx-training",
    "/results/openx_1t_expansion/status.json",
    "/results/openx_1t_expansion/synthesis/qtail_service_delivery_report.json",
    "/results/openx_1t_expansion/synthesis/qtail_service_synthetic_plan.csv",
    "/results/openx_1t_expansion/synthesis/qtail_synthetic_data.csv",
    "/results/openx_1t_expansion/synthesis/qtail_service_model_card.json",
    "/results/openx_1t_expansion/synthesis/qtail_data_engine_report.json",
    "/results/openx_1t_expansion/synthesis/README_QTAIL_DELIVERY.md",
    "/results/openx_1t_expansion/synthesis/qtail_delivery_package.zip",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_status(path: str) -> int | None:
    connection = http.client.HTTPConnection("127.0.0.1", 54655, timeout=3)
    try:
        connection.request("HEAD", path, headers={"Connection": "close"})
        response = connection.getresponse()
        response.read()
        return response.status
    except OSError:
        return None
    finally:
        connection.close()


def current_evidence(root: Path, workspace: Path) -> dict[str, Any]:
    synthesis = marker_status(root, "synthesis")
    if not synthesis.get("valid"):
        raise ValueError(f"synthesis marker invalid: {synthesis.get('error')}")
    status = read_json(root / "status.json")
    checks = {
        str(item.get("id")): bool(item.get("passed"))
        for item in status.get("completion", {}).get("checks", [])
        if isinstance(item, dict)
    }
    missing_checks = [item for item in REQUIRED_COMPLETION_IDS if not checks.get(item)]
    if missing_checks:
        raise ValueError(f"required completion checks not passed: {missing_checks}")

    page_path = workspace / "qtail-openx-training.html"
    page_text = page_path.read_text(encoding="utf-8")
    missing_tokens = [token for token in REQUIRED_PAGE_TOKENS if token not in page_text]
    if missing_tokens:
        raise ValueError(f"page source tokens missing: {missing_tokens}")

    statuses = {path: http_status(path) for path in PUBLIC_PATHS}
    failed_paths = [path for path, code in statuses.items() if code != 200]
    if failed_paths:
        raise ValueError(f"public paths not HTTP 200: {failed_paths}")
    return {
        "page_source_sha256": file_sha256(page_path),
        "synthesis_marker_sha256": file_sha256(
            root / "OPENX_1T_SYNTHESIS_COMPLETE"
        ),
        "required_completion_checks": list(REQUIRED_COMPLETION_IDS),
        "required_page_tokens": list(REQUIRED_PAGE_TOKENS),
        "public_http_status": statuses,
    }


def validate_report(root: Path, workspace: Path) -> dict[str, Any]:
    report_path = root / "final_page_qa.json"
    try:
        report = read_json(report_path)
        evidence = current_evidence(root, workspace)
        valid = bool(
            report.get("format_version") == FORMAT_VERSION
            and report.get("valid") is True
            and report.get("evidence") == evidence
        )
        return {
            "valid": valid,
            "generated_at": report.get("generated_at") if valid else None,
            "error": None if valid else "final page QA evidence mismatch",
            "evidence": evidence,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "generated_at": None, "error": str(error)}


def write_report(root: Path, workspace: Path) -> dict[str, Any]:
    payload = {
        "format_version": FORMAT_VERSION,
        "generated_at": now(),
        "valid": True,
        "claim_boundary": (
            "This proves the local page, status ledger, and final artifacts are "
            "projected over HTTP; it does not extend the model claim boundary."
        ),
        "evidence": current_evidence(root, workspace),
    }
    path = root / "final_page_qa.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--workspace", type=Path, default=Path("/Users/avalok/work/Q-TAIL-MVP")
    )
    args = parser.parse_args()
    if args.action == "write":
        report = write_report(args.root, args.workspace)
        print(json.dumps(report, ensure_ascii=False))
        return
    status = validate_report(args.root, args.workspace)
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status["valid"] else 1)


if __name__ == "__main__":
    main()
