"""Phase 51-C — deterministic worst-error ranking core (W1, W2).

Deterministic ranking protocol (HUMAN-APPROVED 2026-09-04):
  W1 PER_SEED_WORST:
    sort key: (absolute_error_wh DESC, target_timestamp ASC, target_id ASC)
    K_PER_SEED = 20
  W2 SHARED_WORST:
    score: (|e_42| + |e_123| + |e_2026|) / 3
    sort key: (mean_abs_error_wh DESC, target_id ASC)
    K_SHARED = 20

Tie-break is stable + total. No random ordering.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import contract
from . import sources


def _load_working_table(project_root: Path | None = None) -> list[dict[str, str]]:
    """Load the Phase 51-B frozen working table."""
    root = (
        project_root if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    fp = root / "artifacts/worst_error_analysis/phase51_target_level_working_table.csv"
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _stable_sort_per_seed(
    rows: list[dict[str, str]], abs_error_field: str, descending: bool = True
) -> list[dict[str, str]]:
    """Stable total-order sort following FROZEN Phase 51 contract tie-break.

    Frozen tie-break: target_id ASC (no timestamp tier).
    Primary: absolute_error_wh DESC.
    """
    def _key(r):
        ae = float(r[abs_error_field])
        tid = r["target_id"]
        ae_key = -ae if descending else ae
        return (ae_key, tid)
    return sorted(rows, key=_key)


def _stable_sort_shared(
    rows: list[dict[str, str]], descending: bool = True
) -> list[dict[str, str]]:
    """Stable total-order sort with mean_abs_error DESC, target_id ASC."""
    def _key(r):
        mae = float(r["mean_abs_error_wh"])
        tid = r["target_id"]
        mae_key = -mae if descending else mae
        return (mae_key, tid)
    return sorted(rows, key=_key)


def rank_w1_per_seed_worst(
    project_root: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Execute W1: per-seed Top 20 worst absolute_error_wh.

    Returns dict: seed → list of ranked rows (rank 1..K_PER_SEED).
    Each row contains the per-seed error fields + Phase 50 regime labels.
    """
    rows = _load_working_table(project_root)
    out: dict[str, list[dict[str, str]]] = {}

    for seed in contract.SEEDS:
        abs_field = f"seed{seed}_abs_error_wh"
        y_pred_field = f"seed{seed}_y_pred_wh"
        residual_field = f"seed{seed}_residual_wh"
        squared_field = f"seed{seed}_squared_error_wh2"
        sign_field = f"seed{seed}_residual_sign"

        # Sort all 2961 rows for this seed by absolute_error DESC + tie-break
        sorted_rows = _stable_sort_per_seed(rows, abs_field, descending=True)

        # Take top K_PER_SEED
        top_rows = sorted_rows[: contract.K_ABS_PER_SEED]

        # Assign rank 1..K_PER_SEED
        ranked = []
        for i, r in enumerate(top_rows, start=1):
            new_row = {
                "selection_family": "W1_PER_SEED_WORST",
                "seed": seed,
                "rank": str(i),
                "target_id": r["target_id"],
                "target_timestamp": r["target_timestamp"],
                "y_true_wh": r["y_true_wh"],
                "y_pred_wh": r[y_pred_field],
                "residual_wh": r[residual_field],
                "absolute_error_wh": r[abs_field],
                "squared_error_wh2": r[squared_field],
                "residual_sign": r[sign_field],
                "primary_ranking_metric": "absolute_error_wh",
                "primary_ranking_direction": "DESC",
                "tie_break": "target_id ASC",
            }
            ranked.append(new_row)

        out[seed] = ranked

    return out


def rank_w2_shared_worst(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Execute W2: shared hard cases across all 3 seeds.

    Sort by mean_abs_error_wh DESC, target_id ASC.
    Take Top K_SHARED = 20.
    """
    rows = _load_working_table(project_root)
    sorted_rows = _stable_sort_shared(rows, descending=True)
    top_rows = sorted_rows[: contract.K_SHARED]

    ranked = []
    for i, r in enumerate(top_rows, start=1):
        new_row = {
            "selection_family": "W2_SHARED_WORST",
            "seed": "ALL",
            "rank": str(i),
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": r["y_true_wh"],
            "mean_abs_error_wh": r["mean_abs_error_wh"],
            "seed_abs_error_std_wh": r["seed_abs_error_std_wh"],
            "seed42_abs_error_wh": r["seed42_abs_error_wh"],
            "seed123_abs_error_wh": r["seed123_abs_error_wh"],
            "seed2026_abs_error_wh": r["seed2026_abs_error_wh"],
            "primary_ranking_metric": "mean_abs_error_wh",
            "primary_ranking_direction": "DESC",
            "tie_break": "target_id ASC",
        }
        ranked.append(new_row)

    return ranked
