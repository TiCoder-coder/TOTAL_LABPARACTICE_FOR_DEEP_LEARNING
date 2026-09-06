"""Phase45 preflight / prelock gate runner.

Runs every acceptance check, classifies FAILs, and emits preflight rows for
``phase45_preflight_audit.csv`` and the structured summary payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consistency import CheckResult, run_acceptance_checks


@dataclass(frozen=True)
class PreflightReport:
    results: tuple[CheckResult, ...]
    any_critical_fail: bool
    rows: tuple[dict[str, Any], ...]


def run_preflight(ctx: dict[str, Any]) -> PreflightReport:
    """Run all acceptance checks with ``ctx`` and produce preflight rows."""
    results = run_acceptance_checks(ctx)
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "check": r.code,
            "description": r.description,
            "status": r.status,
            "severity": r.severity,
            "actual": str(r.actual),
            "expected": str(r.expected),
        })
    any_critical_fail = any(
        r.status == "FAIL" and r.severity == "CRITICAL" for r in results
    )
    return PreflightReport(
        results=tuple(results),
        any_critical_fail=any_critical_fail,
        rows=tuple(rows),
    )


__all__ = ["PreflightReport", "run_preflight"]
