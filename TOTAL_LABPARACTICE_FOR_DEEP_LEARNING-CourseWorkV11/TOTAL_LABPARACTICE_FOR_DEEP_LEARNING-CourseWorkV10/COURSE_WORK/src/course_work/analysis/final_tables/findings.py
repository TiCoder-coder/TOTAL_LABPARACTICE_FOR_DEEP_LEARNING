# -*- coding: utf-8 -*-
"""Phase 58 - findings, tests, discrepancies, handoffs, summary, signoff."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_findings(root: Path, findings: list[dict]) -> None:
    """Write final_tables_findings.csv."""
    import csv
    fp = root / "artifacts/final_tables/final_tables_findings.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    cols = ["finding_id", "topic", "source_phase", "source_finding_id_if_available",
            "supported_statement", "supporting_table", "caveat", "ready_for_phase59", "status"]
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in findings:
            w.writerow(f)


def write_tests_csv(root: Path, tests: list[dict]) -> None:
    import csv
    fp = root / "artifacts/final_tables/final_tables_tests.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    cols = ["test_id", "scope", "description", "expected", "observed", "critical", "status"]
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in tests:
            w.writerow(t)


def write_discrepancies(root: Path, discrepancies: dict) -> None:
    fp = root / "artifacts/final_tables/final_tables_discrepancies.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(discrepancies, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def write_phase59_handoff(root: Path, handoff: dict) -> None:
    fp = root / "artifacts/final_tables/phase59_final_conclusions_handoff.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(handoff, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def write_summary_json(root: Path, summary: dict) -> None:
    fp = root / "artifacts/final_tables/final_tables_summary.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def write_signoff(root: Path, signoff: dict) -> None:
    fp = root / "artifacts/final_tables/phase_58_signoff.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(signoff, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def write_processing_log(root: Path, log: dict) -> None:
    fp = root / "docs/save_log_in_processing/phase_58_final_tables_log.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def write_report(root: Path, report_md: str) -> None:
    fp = root / "artifacts/final_tables/final_tables_report.md"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(report_md, encoding="utf-8")


def write_readme(root: Path, readme_md: str) -> None:
    fp = root / "artifacts/final_tables/README_FINAL_TABLES.md"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(readme_md, encoding="utf-8")


def write_catalog(root: Path, catalog_md: str) -> None:
    fp = root / "artifacts/final_tables/FINAL_TABLE_CATALOG.md"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(catalog_md, encoding="utf-8")
