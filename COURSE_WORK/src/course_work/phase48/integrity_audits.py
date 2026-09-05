"""Phase 48-D5/D6/D7 — Negative-value audit + saturation audit + persistence baseline context.

Strictly descriptive. No clipping. No invented thresholds.
Persistence bundle is read-only and verified against the canonical checksums.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from course_work.phase48.contract import OUTPUT_DIR
from course_work.phase48.writers import write_csv


SEED_COLUMNS = [("SEED42", "y_pred_seed42"),
                ("SEED123", "y_pred_seed123"),
                ("SEED2026", "y_pred_seed2026")]


# ---------------------------------------------------------------------------
# 5. Negative prediction audit
# ---------------------------------------------------------------------------

def compute_negative_audit(wide_rows: list[dict]) -> list[dict]:
    """Per plan §109. No clipping. Denominator = total N per seed."""
    rows: list[dict] = []
    for seed_label, col in SEED_COLUMNS:
        ys = np.array([float(r[col]) for r in wide_rows], dtype=float)
        n = int(ys.size)
        neg_mask = ys < 0
        neg_count = int(neg_mask.sum())
        neg_frac = (neg_count / n) if n > 0 else 0.0
        if neg_count > 0:
            min_pred = float(np.min(ys))
            first_neg_idx = int(np.argmax(neg_mask))
            first_neg_ts = wide_rows[first_neg_idx]["target_timestamp"]
        else:
            min_pred = float(np.min(ys))
            first_neg_ts = ""
        rows.append({
            "seed": seed_label,
            "negative_count": neg_count,
            "negative_fraction": float(neg_frac),
            "minimum_prediction": min_pred,
            "first_negative_timestamp": first_neg_ts,
            "clipping_applied": False,
            "denominator": n,
            "status": "PASS",
        })
    return rows


def write_negative_audit(wide_rows: list[dict]) -> Path:
    rows = compute_negative_audit(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_negative_value_audit.csv",
        ["seed", "negative_count", "negative_fraction", "minimum_prediction",
         "first_negative_timestamp", "clipping_applied", "denominator", "status"],
        rows,
    )


# ---------------------------------------------------------------------------
# 6. Saturation audit (descriptive only, no invented thresholds)
# ---------------------------------------------------------------------------

def compute_saturation_audit(wide_rows: list[dict]) -> list[dict]:
    """Per plan §110. Conservative structural checks only:
    - unique_prediction_count == 1 → all identical (suspect saturation)
    - fraction_at_exact_min == 1.0 → all predictions equal global min
    - fraction_at_exact_max == 1.0 → all predictions equal global max
    No data-dependent quantile thresholds invented after seeing results.
    """
    rows: list[dict] = []
    for seed_label, col in SEED_COLUMNS:
        ys = np.array([float(r[col]) for r in wide_rows], dtype=float)
        n = int(ys.size)
        unique_count = int(len(np.unique(ys)))
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        frac_at_min = float(np.sum(ys == y_min) / n) if n > 0 else 0.0
        frac_at_max = float(np.sum(ys == y_max) / n) if n > 0 else 0.0
        duplicate_rate = float((n - unique_count) / n) if n > 0 else 0.0

        # Structural suspicion rules only — no data-derived thresholds
        if unique_count == 1:
            suspect = True
            reason = "all_predictions_identical_single_unique_value"
        elif frac_at_min == 1.0:
            suspect = True
            reason = "all_predictions_equal_global_min"
        elif frac_at_max == 1.0:
            suspect = True
            reason = "all_predictions_equal_global_max"
        else:
            suspect = False
            reason = "no_structural_saturation_signal"

        rows.append({
            "seed": seed_label,
            "unique_prediction_count": unique_count,
            "fraction_at_exact_min": frac_at_min,
            "fraction_at_exact_max": frac_at_max,
            "duplicate_rate": duplicate_rate,
            "suspected_saturation": bool(suspect),
            "reason": reason,
            "global_min": y_min,
            "global_max": y_max,
            "status": "PASS",
        })
    return rows


def write_saturation_audit(wide_rows: list[dict]) -> Path:
    rows = compute_saturation_audit(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_saturation_audit.csv",
        ["seed", "unique_prediction_count", "fraction_at_exact_min",
         "fraction_at_exact_max", "duplicate_rate", "suspected_saturation",
         "reason", "global_min", "global_max", "status"],
        rows,
    )


# ---------------------------------------------------------------------------
# 7. Persistence baseline context (read-only, verified against canonical)
# ---------------------------------------------------------------------------

def _load_persistence_bundle(bundle_path: Path) -> tuple[list[dict], str]:
    """Load persistence bundle. Verify sha256 against canonical registry.

    Returns (rows, sha256_hex).
    """
    expected = json.loads((bundle_path.parent.parent / "prediction_checksums.json").read_text())
    expected_sha = expected["predictions"]["persistence"]["sha256"]
    observed_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if expected_sha != observed_sha:
        raise RuntimeError(
            f"Persistence bundle checksum mismatch: expected {expected_sha[:16]}, "
            f"observed {observed_sha[:16]}. Phase 48-D will not load an unverified bundle."
        )
    with bundle_path.open() as fh:
        return list(csv.DictReader(fh)), observed_sha


def _verify_persistence_population_match(pers_rows: list[dict], wide_rows: list[dict]) -> None:
    """target_ids and target_timestamps must match exactly."""
    if len(pers_rows) != len(wide_rows):
        raise RuntimeError(
            f"Persistence row count {len(pers_rows)} != wide table row count {len(wide_rows)}."
        )
    for i, (pr, wr) in enumerate(zip(pers_rows, wide_rows)):
        if pr["target_id"] != wr["target_id"] or pr["target_timestamp"] != wr["target_timestamp"]:
            raise RuntimeError(
                f"Persistence population mismatch at index {i}: "
                f"pers=({pr['target_id']},{pr['target_timestamp']}) vs wide=({wr['target_id']},{wr['target_timestamp']})."
            )
    # Also verify y_true matches the wide table y_true
    for i, (pr, wr) in enumerate(zip(pers_rows, wide_rows)):
        if float(pr["y_true_wh"]) != float(wr["y_true_wh"]):
            raise RuntimeError(
                f"Persistence y_true mismatch at index {i}: pers={pr['y_true_wh']} vs wide={wr['y_true_wh']}."
            )


def compute_baseline_context(wide_rows: list[dict],
                             predictions_dir: Path,
                             lstm_eligibility: dict) -> list[dict]:
    """Per plan §111. Persistence descriptive context only. LSTM: NOT_EVALUATED_BY_PROTOCOL."""
    pers_rows, pers_sha = _load_persistence_bundle(predictions_dir / "final_test_predictions_persistence.csv")
    _verify_persistence_population_match(pers_rows, wide_rows)

    pers_pred = np.array([float(r["y_pred_wh"]) for r in pers_rows], dtype=float)
    pers_true = np.array([float(r["y_true_wh"]) for r in pers_rows], dtype=float)
    pers_diffs = np.diff(pers_pred)
    change_mean_abs = float(np.mean(np.abs(pers_diffs))) if pers_diffs.size > 0 else 0.0

    rows: list[dict] = [
        {
            "model_id": "PERSISTENCE",
            "prediction_bundle_available": True,
            "common_population_verified": True,
            "population_sha256": "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87",
            "bundle_sha256": pers_sha,
            "n": int(pers_pred.size),
            "prediction_mean": float(np.mean(pers_pred)),
            "prediction_std": float(np.std(pers_pred, ddof=1)),
            "prediction_min": float(np.min(pers_pred)),
            "prediction_max": float(np.max(pers_pred)),
            "change_mean_abs_delta": change_mean_abs,
            "interpretation_label": "PERSISTENCE_ONE_STEP_BEHIND_NAIVE_BASELINE",
            "phase47_final_comparison_verdict": "Transformer_better_RMSE_R2_Persistence_better_MAE",
            "notes": "Persistence uses y_hat = y_{t-1}. No scaler. Descriptive context only.",
            "status": "PASS",
        },
        {
            "model_id": "LSTM_TUNED_DEV",
            "prediction_bundle_available": False,
            "common_population_verified": False,
            "population_sha256": "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87",
            "bundle_sha256": "",
            "n": 0,
            "prediction_mean": None,
            "prediction_std": None,
            "prediction_min": None,
            "prediction_max": None,
            "change_mean_abs_delta": None,
            "interpretation_label": lstm_eligibility.get("eligibility_status", "NOT_ELIGIBLE_CONFIG_MISMATCH"),
            "phase47_final_comparison_verdict": "not_compared_phase48_ineligible",
            "notes": (f"LSTM uses lookback L36 (not L72). Per Phase47 plan §61, §143, §149, and "
                      f"final_test_lstm_eligibility.json. Status={lstm_eligibility.get('eligibility_status')}."),
            "status": "PASS",
        },
    ]
    return rows


def write_baseline_context(wide_rows: list[dict],
                           predictions_dir: Path = Path("artifacts/final_test/predictions"),
                           ftest_dir: Path = Path("artifacts/final_test"),
                           ) -> Path:
    lstm_elig_path = ftest_dir / "final_test_lstm_eligibility.json"
    lstm_elig = json.loads(lstm_elig_path.read_text()) if lstm_elig_path.exists() else {}
    rows = compute_baseline_context(wide_rows, predictions_dir, lstm_elig)
    return write_csv(
        OUTPUT_DIR / "prediction_baseline_context.csv",
        ["model_id", "prediction_bundle_available", "common_population_verified",
         "population_sha256", "bundle_sha256", "n",
         "prediction_mean", "prediction_std", "prediction_min", "prediction_max",
         "change_mean_abs_delta", "interpretation_label",
         "phase47_final_comparison_verdict", "notes", "status"],
        rows,
    )
