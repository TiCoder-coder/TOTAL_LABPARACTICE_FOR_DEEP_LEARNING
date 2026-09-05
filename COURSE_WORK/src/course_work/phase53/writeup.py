"""Phase 53 — preflight audit, findings, visual notes.

`phase53_preflight_audit.csv` per canonical schema §128.
`attention_heatmap_findings.csv` per canonical schema (qualitative only).
`attention_heatmap_visual_notes.csv` per canonical schema §138 (optional).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .sources import (
    NUM_HEADS,
    NUM_LAYERS,
    REPORT_CASE_RULE,
    SEEDS,
)


PREFLIGHT_CHECKS: list[dict] = [
    {"check": "Phase52 approved", "expected": "PASS or PASS_WITH_WARNING"},
    {"check": "phase53_ready_in_phase52_handoff", "expected": "true"},
    {"check": "Seed42 dense NPZ exists", "expected": "true"},
    {"check": "Seed123 dense NPZ exists", "expected": "true"},
    {"check": "Seed2026 dense NPZ exists", "expected": "true"},
    {"check": "Raw SHA256s match expected", "expected": "PASS"},
    {"check": "dtype float32", "expected": "PASS"},
    {"check": "Same dense case count across seeds", "expected": "PASS"},
    {"check": "Same dense case order across seeds", "expected": "PASS"},
    {"check": "Same layer count", "expected": "PASS"},
    {"check": "Same head count", "expected": "PASS"},
    {"check": "Same lookback", "expected": "PASS"},
    {"check": "Relative position map verified", "expected": "PASS"},
    {"check": "Case metadata available", "expected": "PASS"},
    {"check": "Phase51 shared top5 ranks available", "expected": "PASS"},
    {"check": "Render config frozen before official rendering", "expected": "PASS"},
    {"check": "Output directory clean", "expected": "PASS"},
    {"check": "No new inference/extraction required", "expected": "PASS"},
    {"check": "Synthetic orientation test PASS", "expected": "PASS"},
    {"check": "Real-source orientation test PASS", "expected": "PASS"},
]


def write_preflight_audit(results: list[dict], output_csv: Path) -> None:
    """Write `phase53_preflight_audit.csv`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "check", "expected", "observed", "critical", "status",
        ])
        w.writeheader()
        w.writerows(results)


def write_findings(findings_rows: list[dict], output_csv: Path) -> None:
    """Write `attention_heatmap_findings.csv`.

    Findings are STRICTLY within Phase53 qualitative visualization scope.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not findings_rows:
        return
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(findings_rows[0].keys()))
        w.writeheader()
        w.writerows(findings_rows)


def write_visual_notes(rows: list[dict], output_csv: Path) -> None:
    """Write `attention_heatmap_visual_notes.csv` (optional)."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit("Phase 53 writeup helpers are libraries — import it from orchestrator.")
