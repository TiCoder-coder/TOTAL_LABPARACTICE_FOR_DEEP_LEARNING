"""Phase 51-E — Persistence context on Phase 51-C selected targets.

Joins the canonical Phase 47 Persistence predictions
(artifacts/final_test/predictions/final_test_predictions_persistence.csv,
SHA 7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b)
onto selected targets.

Persistence is descriptive context only. Phase 51 rankings are
Transformer-defined and are NEVER altered by Persistence results.

The canonical global Phase 47 interpretation is preserved verbatim:
  Persistence better on MAE globally.
  Transformer better on RMSE and R² globally.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from . import regime_context
from course_work.utils.artifacts import get_project_root


PERSISTENCE_SHA = "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b"
PERSISTENCE_REL = "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
N_TEST = 2961

PHASE47_GLOBAL_INTERPRETATION = (
    "Persistence better globally on MAE. "
    "Transformer better globally on RMSE and R^2."
)


def _load_persistence(project_root: Path) -> list[dict[str, str]]:
    fp = project_root / PERSISTENCE_REL
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_ranking(project_root: Path, rel: str) -> list[dict[str, str]]:
    from . import regime_context as rc
    fp = project_root / rc.PHASE51_DIR_REL / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)


def build_persistence_context_table(
    project_root: Path,
) -> list[dict[str, Any]]:
    """Per-selected-target Persistence context rows.

    Schema per row:
      selection_family, seed, rank, target_id,
      transformer_abs_error_wh (or mean_abs for shared),
      persistence_y_pred_wh,
      persistence_residual_wh,
      persistence_absolute_error_wh,
      persistence_squared_error_wh2,
    """
    pers = _load_persistence(project_root)
    assert len(pers) == N_TEST
    pers_by_id = {r["target_id"]: r for r in pers}

    families: list[tuple[str, str, str, str]] = [
        # (selection_family, ranking_csv, error_field_in_ranking, seed_or_all)
        ("W1_PER_SEED_WORST", "worst_per_seed_top20.csv",
         "absolute_error_wh", "per_seed"),
        ("W2_SHARED_WORST", "worst_shared_top20.csv",
         "mean_abs_error_wh", "ALL"),
        ("W3_UNDERPREDICTION_WORST", "worst_underprediction_top10.csv",
         "absolute_error_wh", "per_seed"),
        ("W4_OVERPREDICTION_WORST", "worst_overprediction_top10.csv",
         "absolute_error_wh", "per_seed"),
        ("W3_SH_SHARED_ALL_UNDER", "shared_all_under_top10.csv",
         "mean_abs_error_wh", "ALL"),
        ("W4_SH_SHARED_ALL_OVER", "shared_all_over_top10.csv",
         "mean_abs_error_wh", "ALL"),
    ]

    rows: list[dict[str, Any]] = []
    for family, rel, err_field, kind in families:
        ranking = _load_ranking(project_root, rel)
        for r in ranking:
            tid = r["target_id"]
            p = pers_by_id[tid]
            transformer_err = _to_float(r[err_field])
            persistence_err = _to_float(p["absolute_error_wh"])
            rows.append({
                "selection_family": family,
                "seed": r.get("seed", "ALL"),
                "rank": r["rank"],
                "target_id": tid,
                "target_timestamp": r.get("target_timestamp", p["target_timestamp"]),
                "y_true_wh": r.get("y_true_wh", p["y_true_wh"]),
                "transformer_metric_field": err_field,
                "transformer_abs_error_wh": round(transformer_err, 9),
                "persistence_y_pred_wh": p["y_pred_wh"],
                "persistence_residual_wh": p["residual_wh"],
                "persistence_absolute_error_wh": round(persistence_err, 9),
                "persistence_squared_error_wh2": p["squared_error_wh"],
            })

    return rows


def build_persistence_summary(
    project_root: Path,
    context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-family Persistence summary row: mean absolute error on selected."""
    cnt = Counter((r["selection_family"], r["seed"]) for r in context_rows)
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in context_rows:
        by_key.setdefault(
            (r["selection_family"], r["seed"]), []
        ).append(_to_float(r["persistence_absolute_error_wh"]))

    out = []
    for (family, seed), pers_errs in sorted(by_key.items()):
        out.append({
            "selection_family": family,
            "seed": seed,
            "n_selected": cnt[(family, seed)],
            "persistence_mean_abs_error_wh_on_selected": round(
                sum(pers_errs) / len(pers_errs), 9
            ),
            "persistence_min_abs_error_wh_on_selected": round(
                min(pers_errs), 9
            ),
            "persistence_max_abs_error_wh_on_selected": round(
                max(pers_errs), 9
            ),
        })
    return out
