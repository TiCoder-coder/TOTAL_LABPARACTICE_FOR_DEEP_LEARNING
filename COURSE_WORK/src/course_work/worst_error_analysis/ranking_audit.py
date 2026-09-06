"""Phase 51-C — ranking audit + checksum verification.

Produces a machine-readable audit for each ranking artifact proving:
  - source row count
  - eligible row count
  - requested K
  - emitted row count
  - rank starts at 1 and is contiguous
  - primary ordering correct
  - tie-break correct
  - no duplicate target within (family, seed)
  - sign eligibility correct where applicable
  - source / contract checksums
  - deterministic checksum of result
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from . import contract
from ..utils.artifacts import get_project_root


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_ranks_contiguous(rows: list[dict[str, str]]) -> bool:
    """Verify rank starts at 1 and is contiguous 1..N."""
    ranks = [int(r["rank"]) for r in rows]
    expected = list(range(1, len(ranks) + 1))
    return ranks == expected


def _check_no_duplicate_target(rows: list[dict[str, str]]) -> bool:
    tids = [r["target_id"] for r in rows]
    return len(tids) == len(set(tids))


def _check_primary_ordering_per_seed(
    rows: list[dict[str, str]], abs_field: str = "absolute_error_wh"
) -> bool:
    """Verify absolute_error_wh is monotonically non-increasing."""
    for i in range(1, len(rows)):
        if _to_float(rows[i][abs_field]) > _to_float(rows[i - 1][abs_field]):
            return False
    return True


def _check_shared_ordering(rows: list[dict[str, str]]) -> bool:
    """Verify mean_abs_error_wh is monotonically non-increasing."""
    for i in range(1, len(rows)):
        if _to_float(rows[i]["mean_abs_error_wh"]) > _to_float(rows[i - 1]["mean_abs_error_wh"]):
            return False
    return True


def _check_tie_break_per_seed(rows: list[dict[str, str]]) -> bool:
    """Verify frozen contract tie-break: on equal absolute_error, target_id ASC.

    No target_timestamp tier allowed by frozen contract.
    """
    for i in range(1, len(rows)):
        ae_curr = _to_float(rows[i]["absolute_error_wh"])
        ae_prev = _to_float(rows[i - 1]["absolute_error_wh"])
        if ae_curr == ae_prev:
            if rows[i]["target_id"] < rows[i - 1]["target_id"]:
                return False
    return True


def _check_tie_break_shared(rows: list[dict[str, str]]) -> bool:
    """Verify on equal mean_abs_error, target_id ASC."""
    for i in range(1, len(rows)):
        mae_curr = _to_float(rows[i]["mean_abs_error_wh"])
        mae_prev = _to_float(rows[i - 1]["mean_abs_error_wh"])
        if mae_curr == mae_prev:
            if rows[i]["target_id"] < rows[i - 1]["target_id"]:
                return False
    return True


def _check_signed_eligibility(
    rows: list[dict[str, str]], sign_filter: str
) -> bool:
    """Verify all rows match sign_filter."""
    for r in rows:
        if sign_filter == "residual > 0":
            if _to_float(r["residual_wh"]) <= 0:
                return False
        elif sign_filter == "residual < 0":
            if _to_float(r["residual_wh"]) >= 0:
                return False
    return True


def _check_signed_all_under(rows: list[dict[str, str]]) -> bool:
    """Verify all 3 seeds have residual > 0 for shared all-under."""
    for r in rows:
        if _to_float(r["seed42_residual_wh"]) <= 0:
            return False
        if _to_float(r["seed123_residual_wh"]) <= 0:
            return False
        if _to_float(r["seed2026_residual_wh"]) <= 0:
            return False
    return True


def _check_signed_all_over(rows: list[dict[str, str]]) -> bool:
    """Verify all 3 seeds have residual < 0 for shared all-over."""
    for r in rows:
        if _to_float(r["seed42_residual_wh"]) >= 0:
            return False
        if _to_float(r["seed123_residual_wh"]) >= 0:
            return False
        if _to_float(r["seed2026_residual_wh"]) >= 0:
            return False
    return True


def audit_ranking_artifact(
    csv_path: Path,
    family: str,
    seed: str,
    k: int,
    primary_ordering: str = "abs_desc",
    sign_filter: str | None = None,
    shared_signed_filter: str | None = None,
    contract_sha: str = "",
    source_sha: str = "",
    source_rows: int = 0,
    eligible_rows: int = 0,
) -> dict[str, Any]:
    """Audit a single ranking artifact.

    Note: When `family` is one of the per-seed families (W1/W3/W4), the
    `seed` argument is for labeling only — the audit checks the WHOLE file
    (which contains all 3 seeds × K_PER_SEED rows). For these families,
    set `k` to the total expected row count.
    """
    rows = _load_rows(csv_path)

    # Filter rows by seed for per-seed audits
    if seed in ("42", "123", "2026"):
        seed_rows = [r for r in rows if r.get("seed") == seed]
    else:
        seed_rows = rows

    # For per-seed families, check rank and ordering within each seed
    if seed in ("42", "123", "2026"):
        rank_contiguous = _check_ranks_contiguous(seed_rows)
        primary_correct = _check_primary_ordering_per_seed(seed_rows)
        tie_correct = _check_tie_break_per_seed(seed_rows)
    else:
        rank_contiguous = _check_ranks_contiguous(rows)
        primary_correct = (
            _check_primary_ordering_per_seed(rows) if primary_ordering == "abs_desc"
            else _check_shared_ordering(rows)
        )
        tie_correct = (
            _check_tie_break_per_seed(rows) if primary_ordering == "abs_desc"
            else _check_tie_break_shared(rows)
        )

    checks = {
        "file_exists": csv_path.exists(),
        "rows_emitted_equal_k": len(rows) == k,
        "rows_per_seed_correct": (
            len(seed_rows) == (k // 3) if seed in ("42", "123", "2026")
            else True
        ),
        "rank_starts_at_1_and_contiguous": rank_contiguous,
        "no_duplicate_target_id": _check_no_duplicate_target(seed_rows),
        "primary_ordering_correct": primary_correct,
        "tie_break_correct": tie_correct,
        "sign_eligibility_correct": (
            _check_signed_eligibility(seed_rows, sign_filter)
            if sign_filter in ("residual > 0", "residual < 0") else True
        ),
        "shared_signed_eligibility_correct": (
            _check_signed_all_under(seed_rows) if shared_signed_filter == "all_under"
            else _check_signed_all_over(seed_rows) if shared_signed_filter == "all_over"
            else True
        ),
        "result_sha256": _sha256(csv_path),
    }

    all_pass = all([
        checks["file_exists"],
        checks["rows_emitted_equal_k"],
        checks["rows_per_seed_correct"],
        checks["rank_starts_at_1_and_contiguous"],
        checks["no_duplicate_target_id"],
        checks["primary_ordering_correct"],
        checks["tie_break_correct"],
        checks["sign_eligibility_correct"],
        checks["shared_signed_eligibility_correct"],
    ])

    return {
        "family": family,
        "seed": seed,
        "k": k,
        "rows_emitted": len(rows),
        "rows_per_seed": len(seed_rows),
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "selection_contract_sha256": contract_sha,
        "source_working_table_sha256": source_sha,
        "all_pass": all_pass,
        "checks": checks,
    }
