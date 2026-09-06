"""Generate the worst_case_casebook.md (O51.23).

Renders the casebook in human-readable markdown form. No causal claims;
each case exposes only canonical evidence: case ID, family, rank, target
ID, timestamp, truth, prediction, residual, regime labels, sign consensus,
exact-input reference. Deterministic order.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_casebook_markdown(
    project_root: Path | None = None,
    max_cases: int = 50,
) -> str:
    root = project_root if project_root is not None else get_project_root()
    fp_out = root / PHASE51_DIR_REL / "worst_case_casebook.md"
    cb_fp = root / PHASE51_DIR_REL / "casebook_index.csv"
    um_fp = root / PHASE51_DIR_REL / "casebook_unique_case_master.csv"

    rows = sorted(_load_csv(cb_fp), key=lambda r: (r["selection_family"], r["seed"], int(r["rank"])))
    unique = _load_csv(um_fp) if um_fp.exists() else []

    lines = [
        "# Phase 51 — Worst-Case Casebook (deterministic, descriptive only)",
        "",
        "This document is a deterministic, non-causal render of the frozen "
        "Phase 51-F casebook. Each case exposes only canonical evidence.",
        "",
        "## Casebook summary",
        "",
        f"- Total case memberships: **{len(rows)}**",
        f"- Unique targets: **{len(unique)}**",
        "- Deterministic case ID format: `CASE_{family_short}_{seed}_rank{NNN}_{target_id}`",
        "",
        "## Per-case evidence (deterministic selection)",
        "",
        "| case_id | family | seed | rank | target_id | timestamp | y_true_wh | y_pred_wh | residual_wh | abs_err_wh | regime | sign_consensus | input_window_ref |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    # Show up to max_cases rows
    for r in rows[:max_cases]:
        lineage_short = r["selection_family"].replace("W1_PER_SEED_WORST", "W1").replace("W2_SH_SHARED_ALL", "W2").replace("W3_UNDERPREDICTION_WORST", "W3").replace("W3_SH_SHARED_ALL_UNDER", "W3_SH").replace("W4_OVERPREDICTION_WORST", "W4").replace("W4_SH_SHARED_ALL_OVER", "W4_SH")
        regime = " ".join([
            r.get("R1_TARGET_LEVEL", ""),
            r.get("R2_EXTREME_HIGH", ""),
        ]).strip()
        sign = r.get("cross_seed_consensus_class", "")
        win_ref = r.get("input_window_reference_id", "")
        lines.append(
            f"| {r['case_id']} | {lineage_short} | {r['seed']} | {r['rank']} | "
            f"{r['target_id']} | {r['target_timestamp']} | "
            f"{r['y_true_wh']} | {r['y_pred_wh']} | {r['residual_wh']} | "
            f"{r['absolute_error_wh']} | {regime} | {sign} | {win_ref} |"
        )
    if len(rows) > max_cases:
        lines += [
            "",
            f"_Showing first {max_cases} of {len(rows)} case memberships "
            "(deterministic order). Full list in `casebook_index.csv`._",
            "",
        ]
    content = "\n".join(lines).encode("utf-8")
    atomic_write_bytes(fp_out, content)
    try:
        os.chmod(fp_out, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp_out)
