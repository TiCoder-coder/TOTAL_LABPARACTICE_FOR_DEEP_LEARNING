"""Phase 51-C — signed ranking families (W3, W4, W3_SH, W4_SH).

Residual convention (inherited from Phase 49/50):
  residual = y_true - y_pred
  residual > 0  → UNDERPREDICTION
  residual < 0  → OVERPREDICTION
  residual == 0 → EXACT_ZERO (EXCLUDED from signed rankings)

Deterministic sort keys (FROZEN CONTRACT — DO NOT MODIFY):
  W3 / W4:  (absolute_error_wh DESC, target_id ASC)
  W3_SH / W4_SH: (mean_abs_error_wh DESC, target_id ASC)
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import contract


def _load_working_table(project_root: Path | None = None) -> list[dict[str, str]]:
    root = (
        project_root if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    fp = root / "artifacts/worst_error_analysis/phase51_target_level_working_table.csv"
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _stable_sort_per_seed(
    rows: list[dict[str, str]], abs_error_field: str
) -> list[dict[str, str]]:
    """Per-seed signed sorting: absolute_error DESC, target_id ASC.

    Frozen Phase 51 contract tie-break is target_id ASC only.
    No target_timestamp tier.
    """
    def _key(r):
        return (-float(r[abs_error_field]), r["target_id"])
    return sorted(rows, key=_key)


def _stable_sort_shared(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Shared sorting: mean_abs_error DESC, target_id ASC."""
    def _key(r):
        return (-float(r["mean_abs_error_wh"]), r["target_id"])
    return sorted(rows, key=_key)


def _to_float(x: str) -> float:
    return float(x)


def rank_w3_underprediction_worst(
    project_root: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """W3: per-seed Top K_UNDER_PER_SEED where residual > 0 (UNDERPREDICTION)."""
    rows = _load_working_table(project_root)
    out: dict[str, list[dict[str, str]]] = {}

    for seed in contract.SEEDS:
        residual_field = f"seed{seed}_residual_wh"
        abs_field = f"seed{seed}_abs_error_wh"
        y_pred_field = f"seed{seed}_y_pred_wh"
        squared_field = f"seed{seed}_squared_error_wh2"
        sign_field = f"seed{seed}_residual_sign"

        # Filter: residual > 0 (strictly positive, exclude EXACT_ZERO)
        eligible = [
            r for r in rows if _to_float(r[residual_field]) > 0
        ]
        sorted_rows = _stable_sort_per_seed(eligible, abs_field)
        top_rows = sorted_rows[: contract.K_UNDER_PER_SEED]

        ranked = []
        for i, r in enumerate(top_rows, start=1):
            new_row = {
                "selection_family": "W3_UNDERPREDICTION_WORST",
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
                "filter_condition": "residual > 0",
                "primary_ranking_metric": "absolute_error_wh",
                "primary_ranking_direction": "DESC",
                "tie_break": "target_id ASC",
            }
            ranked.append(new_row)

        out[seed] = ranked

    return out


def rank_w4_overprediction_worst(
    project_root: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """W4: per-seed Top K_OVER_PER_SEED where residual < 0 (OVERPREDICTION)."""
    rows = _load_working_table(project_root)
    out: dict[str, list[dict[str, str]]] = {}

    for seed in contract.SEEDS:
        residual_field = f"seed{seed}_residual_wh"
        abs_field = f"seed{seed}_abs_error_wh"
        y_pred_field = f"seed{seed}_y_pred_wh"
        squared_field = f"seed{seed}_squared_error_wh2"
        sign_field = f"seed{seed}_residual_sign"

        # Filter: residual < 0 (strictly negative, exclude EXACT_ZERO)
        eligible = [
            r for r in rows if _to_float(r[residual_field]) < 0
        ]
        sorted_rows = _stable_sort_per_seed(eligible, abs_field)
        top_rows = sorted_rows[: contract.K_OVER_PER_SEED]

        ranked = []
        for i, r in enumerate(top_rows, start=1):
            new_row = {
                "selection_family": "W4_OVERPREDICTION_WORST",
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
                "filter_condition": "residual < 0",
                "primary_ranking_metric": "absolute_error_wh",
                "primary_ranking_direction": "DESC",
                "tie_break": "target_id ASC",
            }
            ranked.append(new_row)

        out[seed] = ranked

    return out


def rank_w3_shared_all_under(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """W3_SH: shared all-under — residual > 0 for ALL 3 seeds.

    Sort by mean_abs_error_wh DESC, target_id ASC.
    """
    rows = _load_working_table(project_root)
    eligible = [
        r for r in rows
        if _to_float(r["seed42_residual_wh"]) > 0
        and _to_float(r["seed123_residual_wh"]) > 0
        and _to_float(r["seed2026_residual_wh"]) > 0
    ]
    sorted_rows = _stable_sort_shared(eligible)
    top_rows = sorted_rows[: contract.K_SHARED_SIGNED]

    ranked = []
    for i, r in enumerate(top_rows, start=1):
        new_row = {
            "selection_family": "W3_SH_SHARED_ALL_UNDER",
            "seed": "ALL",
            "rank": str(i),
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": r["y_true_wh"],
            "mean_abs_error_wh": r["mean_abs_error_wh"],
            "seed42_residual_wh": r["seed42_residual_wh"],
            "seed123_residual_wh": r["seed123_residual_wh"],
            "seed2026_residual_wh": r["seed2026_residual_wh"],
            "sign_all_under": "true",
            "primary_ranking_metric": "mean_abs_error_wh",
            "primary_ranking_direction": "DESC",
            "tie_break": "target_id ASC",
        }
        ranked.append(new_row)

    return ranked


def rank_w4_shared_all_over(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """W4_SH: shared all-over — residual < 0 for ALL 3 seeds."""
    rows = _load_working_table(project_root)
    eligible = [
        r for r in rows
        if _to_float(r["seed42_residual_wh"]) < 0
        and _to_float(r["seed123_residual_wh"]) < 0
        and _to_float(r["seed2026_residual_wh"]) < 0
    ]
    sorted_rows = _stable_sort_shared(eligible)
    top_rows = sorted_rows[: contract.K_SHARED_SIGNED]

    ranked = []
    for i, r in enumerate(top_rows, start=1):
        new_row = {
            "selection_family": "W4_SH_SHARED_ALL_OVER",
            "seed": "ALL",
            "rank": str(i),
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": r["y_true_wh"],
            "mean_abs_error_wh": r["mean_abs_error_wh"],
            "seed42_residual_wh": r["seed42_residual_wh"],
            "seed123_residual_wh": r["seed123_residual_wh"],
            "seed2026_residual_wh": r["seed2026_residual_wh"],
            "sign_all_over": "true",
            "primary_ranking_metric": "mean_abs_error_wh",
            "primary_ranking_direction": "DESC",
            "tie_break": "target_id ASC",
        }
        ranked.append(new_row)

    return ranked
