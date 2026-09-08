"""Phase 50 — Train-only threshold derivation (Stage A; NO Test access)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from . import contract
from . import sources
from . import train_reference


def _deterministic_sha256_of_json(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True, indent=2).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def derive_thresholds(ref: dict) -> dict:
    """Derive Train-only thresholds using numpy.quantile(method='linear').

    Returns dict with provenance + Q25_y, Q75_y, Q90_y, Q90_abs_delta.
    """
    # Frozen contract: Train-only + numpy + linear + float64
    contract.assert_train_only("REGIME_REFERENCE_TRAIN-v1")
    contract.assert_no_test_threshold("REGIME_REFERENCE_TRAIN-v1")
    contract.assert_no_best_seed(False)
    contract.assert_no_ensemble(False)
    contract.assert_no_3n_iid_pooling(False)
    contract.assert_no_cartesian_mining(False)
    contract.assert_no_worst_error_ranking(False)
    contract.assert_no_attention_analysis(False)
    contract.assert_decile_not_phase50_regime(False)

    # 1. Q25_y, Q75_y, Q90_y on raw Appliances Wh (Train only)
    y_values = np.array(
        [ref["raw_appliances"][tid] for tid in ref["target_ids"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(y_values)):
        raise ValueError(
            "Train y_values contain non-finite entries; cannot derive thresholds."
        )

    q25 = float(np.quantile(y_values, 0.25, method="linear"))
    q75 = float(np.quantile(y_values, 0.75, method="linear"))
    q90 = float(np.quantile(y_values, 0.90, method="linear"))

    # 2. Q90_abs_delta over gap-safe Train delta pairs (no split crossing)
    deltas: list[float] = []
    classified_count = 0
    for tid in ref["target_ids"]:
        d, status = train_reference.compute_train_delta_y(ref, tid)
        if status == "CLASSIFIED":
            deltas.append(abs(float(d)))
            classified_count += 1
    abs_delta = np.array(deltas, dtype=np.float64)
    if abs_delta.size == 0:
        raise ValueError("No classified Train delta pairs; Q90_abs_delta undefined.")
    q90_abs_delta = float(np.quantile(abs_delta, 0.90, method="linear"))

    # 3. Degeneracy checks
    degenerate_target_level = bool(q25 >= q75)
    degenerate_change_magnitude = bool(q90_abs_delta <= 0)

    if degenerate_target_level:
        # Per plan §13: mark TARGET_LEVEL_THRESHOLD_DEGENERATE; do NOT perturb
        status = "TARGET_LEVEL_THRESHOLD_DEGENERATE"
    elif degenerate_change_magnitude:
        status = "CHANGE_MAGNITUDE_THRESHOLD_DEGENERATE"
    else:
        status = "PASS"

    thresholds = {
        "version": "REGIME_THRESHOLDS-v1",
        "source_reference": "REGIME_REFERENCE_TRAIN-v1",
        "target_ids_sha256": ref["target_ids_sha256"],
        "target_unit": "Wh",
        "quantile_method": contract.QUANTILE_METHOD,
        "quantile_library": contract.QUANTILE_LIBRARY,
        "quantile_function": contract.QUANTILE_FUNCTION,
        "quantile_dtype": contract.QUANTILE_DTYPE,
        "n_train_reference": ref["n"],
        "n_train_classified_delta_pairs": int(classified_count),
        "target_level": {"Q25": q25, "Q75": q75},
        "extreme_high": {"Q90": q90},
        "change_magnitude": {
            "Q90_abs_delta": q90_abs_delta,
            "valid_delta_pair_count": int(classified_count),
        },
        "time_of_day": {
            "blocks": [
                {"label": label, "start_hour": s, "end_hour": e}
                for (label, s, e) in contract.TOD_BLOCKS
            ]
        },
        "day_type": {
            "weekday_labels": ["DAY_WEEKDAY"],
            "weekend_labels": ["DAY_WEEKEND"],
        },
        "created_before_test_error_join": True,
        "test_values_used": False,
        "validation_values_used": False,
        "degenerate_target_level": degenerate_target_level,
        "degenerate_change_magnitude": degenerate_change_magnitude,
        "status": status,
    }

    return thresholds


def freeze_thresholds(thresholds: dict, project_root: Path | None = None) -> str:
    """Write regime_thresholds_train_only.json + regime_threshold_fingerprint.json.

    Returns sha256 of the threshold file.
    """
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = out_dir / "regime_thresholds_train_only.json"
    threshold_path.write_text(
        json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8"
    )
    threshold_sha = hashlib.sha256(threshold_path.read_bytes()).hexdigest()

    fingerprint = {
        "version": "REGIME_THRESHOLD_FINGERPRINT-v1",
        "threshold_file_sha256": threshold_sha,
        "train_target_ids_sha256": thresholds["target_ids_sha256"],
        "quantile_method": thresholds["quantile_method"],
        "source_code_version_optional": "phase50-b/v1",
        "status": thresholds["status"],
    }
    fp_path = out_dir / "regime_threshold_fingerprint.json"
    fp_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    return threshold_sha
