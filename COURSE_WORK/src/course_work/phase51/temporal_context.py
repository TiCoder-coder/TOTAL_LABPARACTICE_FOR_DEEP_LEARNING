"""Phase 51-F — deterministic local temporal context ±6 around frozen selected
Test targets.

This module emits canonical ±6 step context rows from frozen Test
predictions. NO new Test inference is performed. NO interpolation.
NO padding. NO forward-fill. NO silent gap crossing.

For each selected target_id (which carries an integer suffix that
maps directly to a 10-min cadence step within the Test population
16774..19734):
  - left context: target_id - k  for k ∈ [1, 6]
  - center:      target_id
  - right context: target_id + k for k ∈ [1, 6]

A neighbor is AVAILABLE iff:
  - its target_id integer suffix falls within [16774, 19734]
  - its row exists in the frozen final_test_predictions_persistence.csv

Otherwise status = "UNAVAILABLE_BOUNDARY" or "UNAVAILABLE_GAP".

The center target row is always AVAILABLE for every frozen Phase 51-C
selected target.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..utils.artifacts import get_project_root


CONTEXT_RADIUS = 6
CADENCE_MINUTES = 10
TEST_POP_FIRST_ID = 16774
TEST_POP_LAST_ID = 19734
TEST_POP_N = 2961
PERSISTENCE_REL = "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
PERSISTENCE_SHA = "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b"
SEED42_REL = "artifacts/final_test/predictions/final_test_predictions_seed42.csv"
SEED123_REL = "artifacts/final_test/predictions/final_test_predictions_seed123.csv"
SEED2026_REL = "artifacts/final_test/predictions/final_test_predictions_seed2026.csv"


def _load_predictions(project_root: Path, rel: str) -> dict[str, dict[str, str]]:
    fp = project_root / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return {r["target_id"]: r for r in csv.DictReader(fh)}


def _target_id_to_index(target_id: str) -> int:
    return int(target_id.split("_")[1])


def _index_to_target_id(index: int) -> str:
    return f"TGT_{index:08d}"


def build_local_temporal_context(
    project_root: Path,
    selected_target_ids: list[str],
) -> list[dict[str, Any]]:
    """Emit ±6 context rows for each selected target.

    Schema per row:
      case_id (filled in by casebook module if absent)
      selection_family (filled in by caller; default "")
      seed (filled in by caller; default "")
      rank (filled in by caller; default "")
      center_target_id
      relative_step ∈ [-6, +6]
      context_target_id (UNAVAILABLE_BOUNDARY if invalid index)
      context_target_timestamp
      context_y_true_wh (Appliances)
      context_seed42_y_pred_wh (frozen)
      context_seed42_residual_wh
      context_seed42_abs_error_wh
      context_persistence_y_pred_wh
      context_persistence_residual_wh
      context_persistence_abs_error_wh
      is_center (1 if relative_step == 0 else 0)
      continuity_valid (1 if center step, else 1 if neighbor row exists in Test population, else 0)
      availability_status (AVAILABLE / UNAVAILABLE_BOUNDARY / UNAVAILABLE_GAP)
    """
    pers = _load_predictions(project_root, PERSISTENCE_REL)
    seed42 = _load_predictions(project_root, SEED42_REL)
    seed123 = _load_predictions(project_root, SEED123_REL)
    seed2026 = _load_predictions(project_root, SEED2026_REL)

    rows: list[dict[str, Any]] = []
    for tid in selected_target_ids:
        center_idx = _target_id_to_index(tid)
        for k in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
            neighbor_idx = center_idx + k
            neighbor_id = _index_to_target_id(neighbor_idx)
            is_center = 1 if k == 0 else 0

            if not (TEST_POP_FIRST_ID <= neighbor_idx <= TEST_POP_LAST_ID):
                rows.append({
                    "center_target_id": tid,
                    "relative_step": k,
                    "context_target_id": neighbor_id,
                    "context_target_timestamp": "",
                    "context_y_true_wh": "",
                    "context_seed42_y_pred_wh": "",
                    "context_seed42_residual_wh": "",
                    "context_seed42_abs_error_wh": "",
                    "context_seed123_y_pred_wh": "",
                    "context_seed2026_y_pred_wh": "",
                    "context_persistence_y_pred_wh": "",
                    "context_persistence_residual_wh": "",
                    "context_persistence_abs_error_wh": "",
                    "is_center": is_center,
                    "continuity_valid": 0,
                    "availability_status": "UNAVAILABLE_BOUNDARY",
                })
                continue

            neighbor_pers = pers.get(neighbor_id)
            if neighbor_pers is None:
                rows.append({
                    "center_target_id": tid,
                    "relative_step": k,
                    "context_target_id": neighbor_id,
                    "context_target_timestamp": "",
                    "context_y_true_wh": "",
                    "context_seed42_y_pred_wh": "",
                    "context_seed42_residual_wh": "",
                    "context_seed42_abs_error_wh": "",
                    "context_seed123_y_pred_wh": "",
                    "context_seed2026_y_pred_wh": "",
                    "context_persistence_y_pred_wh": "",
                    "context_persistence_residual_wh": "",
                    "context_persistence_abs_error_wh": "",
                    "is_center": is_center,
                    "continuity_valid": 0,
                    "availability_status": "UNAVAILABLE_GAP",
                })
                continue

            neighbor_42 = seed42.get(neighbor_id, {})
            neighbor_123 = seed123.get(neighbor_id, {})
            neighbor_2026 = seed2026.get(neighbor_id, {})

            rows.append({
                "center_target_id": tid,
                "relative_step": k,
                "context_target_id": neighbor_id,
                "context_target_timestamp": neighbor_pers["target_timestamp"],
                "context_y_true_wh": neighbor_pers["y_true_wh"],
                "context_seed42_y_pred_wh": neighbor_42.get("y_pred_wh", ""),
                "context_seed42_residual_wh": neighbor_42.get("residual_wh", ""),
                "context_seed42_abs_error_wh": neighbor_42.get("absolute_error_wh", ""),
                "context_seed123_y_pred_wh": neighbor_123.get("y_pred_wh", ""),
                "context_seed2026_y_pred_wh": neighbor_2026.get("y_pred_wh", ""),
                "context_persistence_y_pred_wh": neighbor_pers["y_pred_wh"],
                "context_persistence_residual_wh": neighbor_pers["residual_wh"],
                "context_persistence_abs_error_wh": neighbor_pers["absolute_error_wh"],
                "is_center": is_center,
                "continuity_valid": 1,
                "availability_status": "AVAILABLE",
            })
    return rows
