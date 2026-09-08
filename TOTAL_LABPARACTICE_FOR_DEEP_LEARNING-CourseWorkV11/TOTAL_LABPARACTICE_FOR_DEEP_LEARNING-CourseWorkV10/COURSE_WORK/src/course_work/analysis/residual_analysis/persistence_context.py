from __future__ import annotations

import csv
import hashlib
from typing import Any

import numpy as np

from course_workutils.artifacts import get_project_root, sha256_file


PERSISTENCE_SOURCE_REL = "artifacts/final_test/predictions/final_test_predictions_persistence.csv"


def load_persistence_rows(
    project_root=None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = project_root if project_root is not None else get_project_root()
    path = root / PERSISTENCE_SOURCE_REL
    if not path.exists():
        raise FileNotFoundError(f"Persistence source not found: {path}")
    observed_sha = sha256_file(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    meta = {
        "source_path": str(path),
        "observed_sha256": observed_sha,
        "n_rows": len(rows),
    }
    return rows, meta


def verify_persistence_against_phase47(
    persistence_rows: list[dict[str, str]],
    phase47_long_table: list[dict[str, Any]],
    persistence_meta: dict[str, Any],
) -> dict[str, Any]:
    """Verify that the persistence bundle:
    - matches its declared checksum (caller responsibility)
    - has N=2961
    - shares target_ids with Phase49-B long table
    - shares y_true with Phase49-B long table
    - residual_wh = y_true_wh - y_pred_wh (computed by us, not asserted)
    """
    seed42_phase49 = sorted(
        [r for r in phase47_long_table if r["seed"] == "42"],
        key=lambda r: r["target_id"],
    )
    seed42_target_ids = [r["target_id"] for r in seed42_phase49]
    seed42_y_true = [float(r["y_true_wh"]) for r in seed42_phase49]
    seed42_timestamps = [r["target_timestamp"] for r in seed42_phase49]

    persistence_sorted = sorted(persistence_rows, key=lambda r: r["target_id"])

    n_match = len(persistence_sorted) == 2961
    target_id_match = (
        len(persistence_sorted) == len(seed42_target_ids)
        and [r["target_id"] for r in persistence_sorted] == seed42_target_ids
    )
    y_true_match = (
        len(persistence_sorted) == len(seed42_y_true)
        and all(
            float(persistence_sorted[i]["y_true_wh"]) == seed42_y_true[i]
            for i in range(len(persistence_sorted))
        )
    )
    timestamp_match = (
        len(persistence_sorted) == len(seed42_timestamps)
        and [r["target_timestamp"] for r in persistence_sorted] == seed42_timestamps
    )

    return {
        "N_match": n_match,
        "N_value": len(persistence_sorted),
        "target_id_match": target_id_match,
        "y_true_match": y_true_match,
        "timestamp_match": timestamp_match,
        "observed_sha256": persistence_meta["observed_sha256"],
        "status": "PASS"
        if n_match and target_id_match and y_true_match and timestamp_match
        else "FAIL",
    }


def compute_persistence_residual_context(
    persistence_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute descriptive baseline residual context for Persistence.

    residual = y_true - y_pred (computed here; never trust upstream stored value)
    """
    residuals = np.asarray(
        [float(r["y_true_wh"]) - float(r["y_pred_wh"]) for r in persistence_rows],
        dtype=np.float64,
    )
    abs_residuals = np.abs(residuals)
    n = residuals.size
    signs: list[str] = []
    for r in residuals:
        if r > 0.0:
            signs.append("UNDERPREDICTION")
        elif r < 0.0:
            signs.append("OVERPREDICTION")
        else:
            signs.append("EXACT")

    return {
        "N": int(n),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "std_residual": float(np.std(residuals, ddof=1)),
        "mae": float(np.mean(abs_residuals)),
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "underprediction_count": sum(1 for s in signs if s == "UNDERPREDICTION"),
        "overprediction_count": sum(1 for s in signs if s == "OVERPREDICTION"),
        "exact_count": sum(1 for s in signs if s == "EXACT"),
        "underprediction_fraction": sum(1 for s in signs if s == "UNDERPREDICTION") / n,
        "overprediction_fraction": sum(1 for s in signs if s == "OVERPREDICTION") / n,
        "exact_fraction": sum(1 for s in signs if s == "EXACT") / n,
        "residual_convention": "y_true - y_pred",
        "scope": "BASELINE_RESIDUAL_CONTEXT",
        "purpose": "NOT a model ranking",
    }
