#!/usr/bin/env python3
"""End-to-end validation for the production synthesizer bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    report = json.loads((args.result / "synthesis_report.json").read_text())
    with (args.result / "qtail_synthetic_allocation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if report.get("status") != "complete": errors.append("report_status")
    if sum(int(row["synthetic_count"]) for row in rows) != args.budget: errors.append("budget_closure")
    if abs(sum(float(row["synthetic_share"]) for row in rows) - 1.0) > 1e-9: errors.append("allocation_sum")
    if report.get("tail_share_gain_pp", 0) <= 0: errors.append("tail_share_gain")
    if report.get("materialized_rows") != args.budget: errors.append("materialized_rows")
    payload = {"status": "passed" if not errors else "failed", "errors": errors, "budget": args.budget, "tail_share_gain_pp": report.get("tail_share_gain_pp")}
    (args.result / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
