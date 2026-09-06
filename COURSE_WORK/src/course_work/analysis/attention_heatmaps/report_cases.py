"""Phase 53 — report-selected case manifest (Phase51 W2 SHARED_WORST ranks 1-5).

Deterministic resolution (NO visual selection, NO replacement after rendering).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .sources import (
    MAX_REPORT_CASES,
    REPORT_CASE_RANK_RANGE,
    REPORT_CASE_RULE,
)


def resolve_report_cases(project_root: Path) -> list[dict]:
    """Resolve the Phase51 W2 SHARED_WORST ranks 1-5 into deterministic case rows.

    Returns a list of dicts with:
      report_order, case_row_idx0, target_id, target_timestamp,
      shared_worst_rank, selection_rule, regime_labels, status
    """
    # Phase 51 frozen casebook
    cb_fp = project_root / "artifacts/worst_error_analysis/worst_shared_top20.csv"
    casebook_idx_fp = project_root / "artifacts/worst_error_analysis/casebook_index.csv"
    dense_order_fp = project_root / "artifacts/attention_extraction/attention_dense_case_order.csv"
    case_meta_fp = project_root / "artifacts/attention_extraction/attention_case_metadata.csv"

    # Find dense case row index per target_id
    target_to_row = {}
    if dense_order_fp.exists():
        with dense_order_fp.open() as f:
            # header: case_row_idx0,target_id,...
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.rstrip().split(",")
                if len(fields) >= 2:
                    target_to_row[fields[1]] = int(fields[0])

    # Find regime labels per target_id from dense case metadata
    target_to_regime: dict[str, str] = {}
    if case_meta_fp.exists():
        with case_meta_fp.open() as f:
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.rstrip().split(",")
                if len(fields) >= 2:
                    target_to_regime[fields[0]] = fields[1] if len(fields) > 1 else ""

    # Map W2 ranks 1-5 from frozen Phase51 worst_shared_top20
    rows = []
    if not cb_fp.exists():
        return rows

    import csv as _csv
    with cb_fp.open() as f:
        r = _csv.DictReader(f)
        for row in r:
            if row.get("selection_family") != "W2_SHARED_WORST":
                continue
            rank = int(row["rank"])
            if REPORT_CASE_RANK_RANGE[0] <= rank <= REPORT_CASE_RANK_RANGE[1]:
                tid = row["target_id"]
                ts = row.get("target_timestamp", "UNKNOWN")
                case_row = target_to_row.get(tid, -1)
                regime = target_to_regime.get(tid, "")
                rows.append({
                    "report_order": rank,
                    "case_row_idx0": case_row,
                    "target_id": tid,
                    "target_timestamp": ts,
                    "shared_worst_rank": rank,
                    "selection_rule": REPORT_CASE_RULE,
                    "regime_labels": regime,
                    "status": "PASS" if case_row >= 0 else "FAIL",
                })

    rows.sort(key=lambda d: d["shared_worst_rank"])
    # Trim to MAX_REPORT_CASES
    rows = rows[:MAX_REPORT_CASES]
    return rows


def write_report_case_manifest(rows: list[dict], output_csv: Path) -> None:
    """Write `attention_heatmap_report_cases.csv`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write empty file with headers
        with output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "report_order", "case_row_idx0", "target_id", "target_timestamp",
                "shared_worst_rank", "selection_rule", "regime_labels", "status",
            ])
            w.writeheader()
        return
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit("Phase 53 report_cases is a library — import it from orchestrator.")
