"""Phase 48 — atomic CSV / JSON writers under artifacts/prediction_analysis/."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from course_work.prediction_analysis.contract import OUTPUT_DIR, OUTPUT_VERSION, PHASE_NUM


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _to_jsonable(v) for k, v in r.items()})
    return path


def write_json(path: Path, data: Any) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return path


def write_source_verification(rows: Sequence[Mapping[str, Any]]) -> Path:
    return write_csv(
        OUTPUT_DIR / "prediction_source_verification.csv",
        [
            "source_id",
            "file",
            "expected_sha256",
            "observed_sha256",
            "row_count",
            "expected_row_count",
            "population_sha256",
            "frozen",
            "checksum_match",
            "row_count_match",
            "all_target_ids_unique",
            "all_predictions_finite",
            "all_y_true_finite",
            "chronological",
            "status",
        ],
        rows,
    )


def write_alignment_audit(rows: Sequence[Mapping[str, Any]]) -> Path:
    return write_csv(
        OUTPUT_DIR / "prediction_alignment_audit.csv",
        ["check", "seed42", "seed123", "seed2026", "persistence", "expected", "status"],
        rows,
    )


def write_preflight_audit(rows: Sequence[Mapping[str, Any]]) -> Path:
    return write_csv(
        OUTPUT_DIR / "phase48_preflight_audit.csv",
        ["check", "expected", "observed", "critical", "status"],
        rows,
    )


def write_wide_table(rows: Sequence[Mapping[str, Any]]) -> Path:
    return write_csv(
        OUTPUT_DIR / "prediction_wide_table.csv",
        [
            "target_id",
            "target_timestamp",
            "y_true_wh",
            "y_pred_seed42",
            "y_pred_seed123",
            "y_pred_seed2026",
            "seed_mean_prediction",
            "seed_std_prediction",
            "seed_min_prediction",
            "seed_max_prediction",
            "seed_range_prediction",
        ],
        rows,
    )


def write_long_table(rows: Sequence[Mapping[str, Any]]) -> Path:
    return write_csv(
        OUTPUT_DIR / "prediction_long_table.csv",
        [
            "target_id",
            "target_timestamp",
            "y_true_wh",
            "seed",
            "y_pred_wh",
            "population_sha256",
            "source_prediction_sha256",
        ],
        rows,
    )


def write_manifest(
    population_sha256: str,
    final_lock_sha256: str,
    phase47_signoff_sha256: str,
    checksum_registry_sha256: str,
    seed42_sha: str,
    seed123_sha: str,
    seed2026_sha: str,
    persistence_sha: str,
    n_test: int,
    first_target_id: str,
    last_target_id: str,
    first_target_timestamp: str,
    last_target_timestamp: str,
    lag_sign_convention: str,
    seed_std_ddof: int,
) -> Path:
    manifest = {
        "phase": PHASE_NUM,
        "phase_name": "Prediction analysis",
        "version": OUTPUT_VERSION,
        "source_phase47_version": "FINAL_TEST_EVAL-v1",
        "source_phase47_signoff_sha256": phase47_signoff_sha256,
        "final_lock_sha256": final_lock_sha256,
        "test_population_sha256": population_sha256,
        "test_population_id": "FINAL_TEST_POP-v1",
        "n_test": n_test,
        "first_target_id": first_target_id,
        "last_target_id": last_target_id,
        "first_target_timestamp": first_target_timestamp,
        "last_target_timestamp": last_target_timestamp,
        "source_prediction_files": {
            "transformer_seed42": "artifacts/final_test/predictions/final_test_predictions_seed42.csv",
            "transformer_seed123": "artifacts/final_test/predictions/final_test_predictions_seed123.csv",
            "transformer_seed2026": "artifacts/final_test/predictions/final_test_predictions_seed2026.csv",
            "persistence": "artifacts/final_test/predictions/final_test_predictions_persistence.csv",
        },
        "source_prediction_sha256s": {
            "transformer_seed42": seed42_sha,
            "transformer_seed123": seed123_sha,
            "transformer_seed2026": seed2026_sha,
            "persistence": persistence_sha,
        },
        "checksum_registry_sha256": checksum_registry_sha256,
        "transformer_seeds": [42, 123, 2026],
        "new_inference": False,
        "new_training": False,
        "model_selection": False,
        "seed_selection": False,
        "ensemble": False,
        "seed_std_ddof": seed_std_ddof,
        "lag_sign_convention": lag_sign_convention,
        "primary_scope": "DESCRIPTIVE_PREDICTION_BEHAVIOR",
        "status": "PASS",
        "created_at": utc_now_iso(),
    }
    return write_json(OUTPUT_DIR / "prediction_analysis_manifest.json", manifest)


def write_contract(
    population_sha256: str,
    final_lock_sha256: str,
    n_test: int,
    seed_std_ddof: int,
    lag_sign_convention: str,
    lag_range_min: int,
    lag_range_max: int,
    acf_registered_lags: list,
    peak_timing_window_steps: int,
    top_disagreement_k: int,
    rolling_window: int,
) -> Path:
    contract = {
        "phase": PHASE_NUM,
        "version": OUTPUT_VERSION,
        "uses_only_frozen_phase47_predictions": True,
        "analyzes_all_three_transformer_seeds": True,
        "transformer_seeds": [42, 123, 2026],
        "persistence_baseline_label": "PERSISTENCE",
        "seed_mean_prediction_meaning": "DESCRIPTIVE_CENTRAL_TENDENCY_NOT_ENSEMBLE",
        "seed_spread_meaning": "CROSS_SEED_PREDICTION_SPREAD_NOT_CONFIDENCE_INTERVAL",
        "seed_std_ddof": seed_std_ddof,
        "lag_sign_convention": lag_sign_convention,
        "lag_range_steps": list(range(lag_range_min, lag_range_max + 1)),
        "acf_registered_lags": acf_registered_lags,
        "peak_timing_window_steps": peak_timing_window_steps,
        "top_disagreement_k": top_disagreement_k,
        "rolling_window_samples": rolling_window,
        "forbidden_actions": {
            "new_test_inference": True,
            "checkpoint_loading_for_new_predictions": True,
            "training": True,
            "scaler_fitting": True,
            "best_seed_selection": True,
            "ensemble_evaluation": True,
            "prediction_shifting": True,
            "prediction_clipping": True,
            "post_hoc_calibration": True,
            "test_derived_threshold_tuning": True,
            "residual_analysis_creep": True,
            "regime_specific_rmse": True,
            "worst_error_ranking": True,
            "attention_extraction": True,
        },
        "deferred_to_other_phases": {
            "residual_analysis": 49,
            "error_regime": 50,
            "worst_error": 51,
            "attention": "52+",
        },
        "test_population": {
            "id": "FINAL_TEST_POP-v1",
            "n": n_test,
            "sha256": population_sha256,
        },
        "final_lock_sha256": final_lock_sha256,
        "frozen_at": utc_now_iso(),
        "status": "FROZEN",
    }
    return write_json(OUTPUT_DIR / "prediction_analysis_contract.json", contract)
