"""Build final V2 reporting analyses from frozen Step 17 predictions only.

This module never loads a dataset, model, checkpoint, or scaler.  It validates
the frozen Step 17 prediction checksums before deriving presentation-only
Phase 48-51 summaries, then creates concise Phase 58-59 reporting summaries.
Historical V1 artifacts are read only and are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


BENCHMARK_ROOT = Path("artifacts/model_improvement_v2/post_hoc_v2_benchmark")
OUTPUT_ROOT = Path("artifacts/model_improvement_v2/final_reporting_analysis")
EXPECTED_POPULATION_FINGERPRINT = (
    "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
)
EXPECTED_COUNT = 2961
PREDICTION_FILES = {
    "V2 Seed 42": "seed_42.csv",
    "V2 Seed 123": "seed_123.csv",
    "V2 Seed 2026": "seed_2026.csv",
    "V2 Equal-weight Ensemble": "ensemble.csv",
    "Persistence": "persistence.csv",
}
LINEAGE = {
    "project": "MODEL_IMPROVEMENT_V2",
    "benchmark": "POST_HOC_V2_BENCHMARK",
    "analysis": "REPORTING_ANALYSIS_FROM_FROZEN_PREDICTIONS",
    "final_policy": "V2 Equal-weight Ensemble",
    "training_executed": False,
    "inference_executed": False,
    "raw_test_source_opened": False,
    "best_seed_selection": False,
}


class V2ReportingError(RuntimeError):
    """Raised when frozen evidence or reporting lineage is invalid."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V2ReportingError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(value))
    temporary.replace(path)


def _population_fingerprint(target_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(str(value) for value in target_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_prediction(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    required = {"target_id", "target_timestamp", "y_true_wh", "y_pred_wh", "residual_wh"}
    if len(raw) != EXPECTED_COUNT or not raw or not required.issubset(raw[0]):
        raise V2ReportingError(f"Invalid frozen prediction schema/count: {path}")
    rows: list[dict[str, Any]] = []
    for item in raw:
        y_true = float(item["y_true_wh"])
        y_pred = float(item["y_pred_wh"])
        residual = float(item["residual_wh"])
        if not all(math.isfinite(v) for v in (y_true, y_pred, residual)):
            raise V2ReportingError(f"Non-finite frozen prediction value: {path}")
        if abs((y_true - y_pred) - residual) > 1e-9:
            raise V2ReportingError(f"Residual identity mismatch: {path}:{item['target_id']}")
        rows.append({
            "target_id": item["target_id"],
            "target_timestamp": item["target_timestamp"],
            "y_true_wh": y_true,
            "y_pred_wh": y_pred,
            "residual_wh": residual,
        })
    return rows


def _metric(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise V2ReportingError("Metric vectors are empty or misaligned")
    residual = [a - b for a, b in zip(y_true, y_pred)]
    n = len(residual)
    sse = sum(value * value for value in residual)
    sae = sum(abs(value) for value in residual)
    mean_y = sum(y_true) / n
    ss_tot = sum((value - mean_y) ** 2 for value in y_true)
    return {
        "n": n,
        "rmse_wh": math.sqrt(sse / n),
        "mae_wh": sae / n,
        "r2": None if ss_tot == 0 else 1.0 - sse / ss_tot,
        "mean_residual_wh": sum(residual) / n,
        "sse": sse,
        "sae": sae,
    }


def _sample_std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_index(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    label_by_id = {
        "V2_FINAL_SEED_42": "V2 Seed 42",
        "V2_FINAL_SEED_123": "V2 Seed 123",
        "V2_FINAL_SEED_2026": "V2 Seed 2026",
        "V2_FINAL_EQUAL_WEIGHT_ENSEMBLE": "V2 Equal-weight Ensemble",
        "PERSISTENCE_LAST_VALUE": "Persistence",
    }
    result = {}
    for row in metrics.get("metrics", []):
        if isinstance(row, dict) and row.get("model_id") in label_by_id:
            result[label_by_id[row["model_id"]]] = row
    if set(result) != set(PREDICTION_FILES):
        raise V2ReportingError("Step 17 metric model set is incomplete")
    return result


def _verify_sources(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    benchmark = root / BENCHMARK_ROOT
    manifest_path = benchmark / "step17_benchmark_manifest.json"
    addendum_path = benchmark / "step17_mape_addendum_manifest.json"
    metrics_path = benchmark / "step17_metrics_with_mape.json"
    closure_path = root / "artifacts/model_improvement_v2/model_improvement_v2_final_closure.json"
    lock_path = root / "artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json"
    manifest = _read_json(manifest_path)
    addendum = _read_json(addendum_path)
    metrics = _read_json(metrics_path)
    closure = _read_json(closure_path)
    lock = _read_json(lock_path)
    if manifest.get("status") != "PASS" or addendum.get("status") != "PASS":
        raise V2ReportingError("Step 17 manifest/addendum is not PASS")
    if addendum.get("metrics_sha256") != _sha256(metrics_path):
        raise V2ReportingError("Step 17 MAPE metrics checksum mismatch")
    if addendum.get("source_manifest_sha256") != _sha256(manifest_path):
        raise V2ReportingError("Step 17 source manifest checksum mismatch")
    if closure.get("status") != "COMPLETE" or lock.get("lock_status") != "LOCKED":
        raise V2ReportingError("Final closure/lock is not complete and locked")
    if closure.get("final_policy", {}).get("best_seed_selection") is not False:
        raise V2ReportingError("Best-seed selection is forbidden")

    expected_sha = manifest.get("prediction_sha256", {})
    addendum_sha = addendum.get("source_prediction_sha256", {})
    loaded: dict[str, list[dict[str, Any]]] = {}
    canonical_ids: list[str] | None = None
    canonical_timestamps: list[str] | None = None
    canonical_truth: list[float] | None = None
    source_sha: dict[str, str] = {}
    for label, filename in PREDICTION_FILES.items():
        path = benchmark / "predictions" / filename
        relative = path.relative_to(root).as_posix()
        actual = _sha256(path)
        if expected_sha.get(relative) != actual or addendum_sha.get(relative) != actual:
            raise V2ReportingError(f"Frozen prediction checksum mismatch: {relative}")
        rows = _load_prediction(path)
        ids = [row["target_id"] for row in rows]
        timestamps = [row["target_timestamp"] for row in rows]
        truth = [row["y_true_wh"] for row in rows]
        if canonical_ids is None:
            canonical_ids, canonical_timestamps, canonical_truth = ids, timestamps, truth
        elif ids != canonical_ids or timestamps != canonical_timestamps or truth != canonical_truth:
            raise V2ReportingError(f"Frozen prediction population mismatch: {relative}")
        loaded[label] = rows
        source_sha[relative] = actual
    assert canonical_ids is not None
    fingerprint = _population_fingerprint(canonical_ids)
    if fingerprint != EXPECTED_POPULATION_FINGERPRINT:
        raise V2ReportingError("Step 17 ordered target population fingerprint mismatch")
    return loaded, {
        "manifest_sha256": _sha256(manifest_path),
        "mape_addendum_manifest_sha256": _sha256(addendum_path),
        "metrics_with_mape_sha256": _sha256(metrics_path),
        "final_closure_sha256": _sha256(closure_path),
        "final_lock_sha256": _sha256(lock_path),
        "prediction_sha256": source_sha,
        "population_fingerprint": fingerprint,
        "metrics": metrics,
        "closure": closure,
        "lock": lock,
    }


def _base(schema: str, source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": schema,
        "status": "PASS",
        "lineage": dict(LINEAGE),
        "source_population_fingerprint": source["population_fingerprint"],
        "source_prediction_sha256": source["prediction_sha256"],
        "sample_count": EXPECTED_COUNT,
    }


def _verified_train_thresholds(root: Path, source: Mapping[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    threshold_path = root / "artifacts/error_by_regime/regime_thresholds_train_only.json"
    fingerprint_path = root / "artifacts/error_by_regime/test_regime_assignment_fingerprint.json"
    thresholds = _read_json(threshold_path)
    historical_fingerprint = _read_json(fingerprint_path)
    if (
        thresholds.get("status") != "PASS"
        or thresholds.get("test_values_used") is not False
        or thresholds.get("validation_values_used") is not False
        or historical_fingerprint.get("status") != "PASS"
        or historical_fingerprint.get("test_population_sha256") != source["population_fingerprint"]
        or historical_fingerprint.get("n_rows") != EXPECTED_COUNT
        or historical_fingerprint.get("threshold_sha256") != _sha256(threshold_path)
    ):
        raise V2ReportingError("Phase 50 Train-only threshold/population lineage mismatch")
    return thresholds, threshold_path, fingerprint_path


def _phase48(root: Path, data: Mapping[str, list[dict[str, Any]]], source: Mapping[str, Any]) -> dict[str, Any]:
    metric_by_label = _metric_index(source["metrics"])
    truth = [row["y_true_wh"] for row in data["V2 Equal-weight Ensemble"]]
    persistence = [row["y_pred_wh"] for row in data["Persistence"]]
    actual_change = [a - b for a, b in zip(truth, persistence)]
    thresholds, threshold_path, _ = _verified_train_thresholds(root, source)
    peak_threshold = float(thresholds["extreme_high"]["Q90"])
    peak_indices = [i for i, value in enumerate(truth) if value >= peak_threshold]
    rows = []
    for label in ("V2 Equal-weight Ensemble", "Persistence"):
        prediction = [row["y_pred_wh"] for row in data[label]]
        predicted_change = [a - b for a, b in zip(prediction, persistence)]
        peak_metric = _metric(
            [truth[i] for i in peak_indices], [prediction[i] for i in peak_indices]
        )
        registered = metric_by_label[label]
        rows.append({
            "model": label,
            "rmse_wh": registered["rmse_wh"],
            "mae_wh": registered["mae_wh"],
            "mape_pct": registered["mape_pct"],
            "r2": registered["r2"],
            "prediction_mean_wh": statistics.fmean(prediction),
            "prediction_std_wh": _sample_std(prediction),
            "prediction_min_wh": min(prediction),
            "prediction_max_wh": max(prediction),
            "mean_abs_predicted_change_wh": statistics.fmean(abs(v) for v in predicted_change),
            "mean_abs_actual_change_wh": statistics.fmean(abs(v) for v in actual_change),
            "train_q90_peak_rmse_wh": peak_metric["rmse_wh"],
        })
    result = _base("MODEL_IMPROVEMENT_V2_PHASE48_REPORTING-v1", source)
    result.update({
        "analysis": "PREDICTION_DISTRIBUTION_CHANGE_PEAK_PERSISTENCE_COMPARISON",
        "peak_definition": {
            "rule": "y_true_wh >= TRAIN_Q90_Y",
            "threshold_wh": peak_threshold,
            "threshold_source": threshold_path.relative_to(root).as_posix(),
            "threshold_sha256": _sha256(threshold_path),
            "count": len(peak_indices),
            "test_derived_threshold": False,
        },
        "rows": rows,
    })
    return result


def _phase49(data: Mapping[str, list[dict[str, Any]]], source: Mapping[str, Any]) -> dict[str, Any]:
    metric_by_label = _metric_index(source["metrics"])
    rows = []
    for label in ("V2 Equal-weight Ensemble", "Persistence"):
        residual = [row["residual_wh"] for row in data[label]]
        registered = metric_by_label[label]
        n = len(residual)
        rows.append({
            "model": label,
            "rmse_wh": registered["rmse_wh"],
            "mae_wh": registered["mae_wh"],
            "mape_pct": registered["mape_pct"],
            "r2": registered["r2"],
            "mean_residual_wh": statistics.fmean(residual),
            "residual_std_wh": _sample_std(residual),
            "residual_q05_wh": _quantile(residual, 0.05),
            "residual_median_wh": _quantile(residual, 0.5),
            "residual_q95_wh": _quantile(residual, 0.95),
            "underprediction_fraction": sum(value > 0 for value in residual) / n,
            "overprediction_fraction": sum(value < 0 for value in residual) / n,
            "exact_fraction": sum(value == 0 for value in residual) / n,
        })
    result = _base("MODEL_IMPROVEMENT_V2_PHASE49_REPORTING-v1", source)
    result.update({"analysis": "RESIDUAL_BIAS_AND_ERROR_DISTRIBUTION", "residual_convention": "y_true_wh - y_pred_wh", "rows": rows})
    return result


def _regime_labels(row: Mapping[str, Any], persistence_row: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, str]:
    y_true = float(row["y_true_wh"])
    delta = y_true - float(persistence_row["y_pred_wh"])
    timestamp = datetime.strptime(str(row["target_timestamp"]), "%Y-%m-%d %H:%M:%S")
    q25 = float(thresholds["target_level"]["Q25"])
    q75 = float(thresholds["target_level"]["Q75"])
    q90 = float(thresholds["extreme_high"]["Q90"])
    q90_delta = float(thresholds["change_magnitude"]["Q90_abs_delta"])
    if y_true < q25:
        level = "TL_LOW"
    elif y_true < q75:
        level = "TL_MID"
    else:
        level = "TL_HIGH"
    if timestamp.hour < 6:
        tod = "TOD_NIGHT"
    elif timestamp.hour < 12:
        tod = "TOD_MORNING"
    elif timestamp.hour < 18:
        tod = "TOD_AFTERNOON"
    else:
        tod = "TOD_EVENING"
    return {
        "R1_TARGET_LEVEL": level,
        "R2_EXTREME_HIGH": "EXTREME_HIGH" if y_true >= q90 else "NON_EXTREME",
        "R3_CHANGE_MAGNITUDE": "CHANGE_RAPID" if abs(delta) >= q90_delta else "CHANGE_NORMAL",
        "R4_CHANGE_DIRECTION": "DIR_UP" if delta > 0 else "DIR_DOWN" if delta < 0 else "DIR_FLAT",
        "R5_TIME_OF_DAY": tod,
        "R6_DAY_TYPE": "DAY_WEEKDAY" if timestamp.weekday() < 5 else "DAY_WEEKEND",
    }


def _phase50(root: Path, data: Mapping[str, list[dict[str, Any]]], source: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    thresholds, threshold_path, fingerprint_path = _verified_train_thresholds(root, source)
    threshold_sha = _sha256(threshold_path)

    ensemble = data["V2 Equal-weight Ensemble"]
    persistence = data["Persistence"]
    assignments: list[dict[str, str]] = []
    for row, baseline in zip(ensemble, persistence):
        labels = _regime_labels(row, baseline, thresholds)
        assignments.append({"target_id": row["target_id"], **labels})
    result_rows = []
    for family in assignments[0]:
        if family == "target_id":
            continue
        labels = sorted({row[family] for row in assignments})
        for label in labels:
            indices = [i for i, row in enumerate(assignments) if row[family] == label]
            truth = [ensemble[i]["y_true_wh"] for i in indices]
            final_pred = [ensemble[i]["y_pred_wh"] for i in indices]
            baseline_pred = [persistence[i]["y_pred_wh"] for i in indices]
            final_metric = _metric(truth, final_pred)
            baseline_metric = _metric(truth, baseline_pred)
            result_rows.append({
                "regime_family": family,
                "regime_label": label,
                "n": len(indices),
                "ensemble_rmse_wh": final_metric["rmse_wh"],
                "ensemble_mae_wh": final_metric["mae_wh"],
                "ensemble_r2": final_metric["r2"],
                "persistence_rmse_wh": baseline_metric["rmse_wh"],
                "persistence_mae_wh": baseline_metric["mae_wh"],
                "persistence_r2": baseline_metric["r2"],
                "ensemble_minus_persistence_rmse_wh": final_metric["rmse_wh"] - baseline_metric["rmse_wh"],
            })
    result = _base("MODEL_IMPROVEMENT_V2_PHASE50_REPORTING-v1", source)
    result.update({
        "analysis": "ERROR_BY_TRAIN_DEFINED_REGIME",
        "assignment_mode": "REBUILT_FROM_FROZEN_PREDICTION_COLUMNS_USING_VERIFIED_TRAIN_THRESHOLDS",
        "historical_assignment_csv_reused": False,
        "threshold_path": threshold_path.relative_to(root).as_posix(),
        "threshold_sha256": threshold_sha,
        "historical_assignment_population_fingerprint_path": fingerprint_path.relative_to(root).as_posix(),
        "population_equality": "PASS",
        "rows": result_rows,
    })
    return result, assignments


def _phase51(data: Mapping[str, list[dict[str, Any]]], assignments: Sequence[Mapping[str, str]], source: Mapping[str, Any]) -> dict[str, Any]:
    ensemble = data["V2 Equal-weight Ensemble"]
    persistence = data["Persistence"]
    ranked = sorted(
        range(len(ensemble)),
        key=lambda i: (-abs(ensemble[i]["residual_wh"]), ensemble[i]["target_id"]),
    )[:20]
    rows = []
    for rank, index in enumerate(ranked, start=1):
        row = ensemble[index]
        baseline = persistence[index]
        rows.append({
            "rank": rank,
            "target_id": row["target_id"],
            "target_timestamp": row["target_timestamp"],
            "y_true_wh": row["y_true_wh"],
            "ensemble_prediction_wh": row["y_pred_wh"],
            "residual_wh": row["residual_wh"],
            "absolute_error_wh": abs(row["residual_wh"]),
            "persistence_prediction_wh": baseline["y_pred_wh"],
            "persistence_absolute_error_wh": abs(baseline["residual_wh"]),
            **{key: value for key, value in assignments[index].items() if key != "target_id"},
        })
    all_abs = [abs(row["residual_wh"]) for row in ensemble]
    all_sse = [row["residual_wh"] ** 2 for row in ensemble]
    result = _base("MODEL_IMPROVEMENT_V2_PHASE51_REPORTING-v1", source)
    result.update({
        "analysis": "FINAL_POLICY_WORST_ERROR_RANKING_AND_CASE_SUMMARY",
        "ranking_contract": {"metric": "absolute_error_wh", "direction": "DESC", "tie_break": "target_id ASC", "k": 20},
        "historical_v1_context_reused": False,
        "context_source": "FROZEN_STEP17_PREDICTION_ROW_PLUS_PHASE50_V2_DERIVED_REGIMES",
        "top20_sae_share": sum(row["absolute_error_wh"] for row in rows) / sum(all_abs),
        "top20_sse_share": sum(row["residual_wh"] ** 2 for row in rows) / sum(all_sse),
        "rows": rows,
    })
    return result


def _phase58(source: Mapping[str, Any], phase48: Mapping[str, Any], phase49: Mapping[str, Any], phase50: Mapping[str, Any], phase51: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metric_index(source["metrics"])
    lock = source["lock"]
    development = lock["development_metrics"]
    result = _base("MODEL_IMPROVEMENT_V2_PHASE58_REPORTING-v1", source)
    result.update({
        "analysis": "V2_FINAL_SUMMARY_REDUCED_SCOPE",
        "attention_scope": "Historical V1 interpretability evidence only; not attributed to V2",
        "phase48_51_status": [phase48["status"], phase49["status"], phase50["status"], phase51["status"]],
        "rows": [
            {
                "evidence": "V2 development selection",
                "model": lock["final_prediction_policy"]["policy_id"],
                "rmse_wh": development["pooled_rmse_wh"],
                "mae_wh": development["pooled_mae_wh"],
                "mape_pct": None,
                "r2": development["pooled_r2"],
                "decision": "FINAL_LOCKED_POLICY",
            },
            {
                "evidence": "POST_HOC_V2_BENCHMARK",
                "model": "V2 Equal-weight Ensemble",
                "rmse_wh": metrics["V2 Equal-weight Ensemble"]["rmse_wh"],
                "mae_wh": metrics["V2 Equal-weight Ensemble"]["mae_wh"],
                "mape_pct": metrics["V2 Equal-weight Ensemble"]["mape_pct"],
                "r2": metrics["V2 Equal-weight Ensemble"]["r2"],
                "decision": "FINAL_LOCKED_POLICY",
            },
            {
                "evidence": "POST_HOC_V2_BENCHMARK",
                "model": "Persistence",
                "rmse_wh": metrics["Persistence"]["rmse_wh"],
                "mae_wh": metrics["Persistence"]["mae_wh"],
                "mape_pct": metrics["Persistence"]["mape_pct"],
                "r2": metrics["Persistence"]["r2"],
                "decision": "BASELINE",
            },
        ],
    })
    return result


def _phase59(source: Mapping[str, Any], phase58: Mapping[str, Any]) -> dict[str, Any]:
    closure = source["closure"]
    ensemble = _metric_index(source["metrics"])["V2 Equal-weight Ensemble"]
    persistence = _metric_index(source["metrics"])["Persistence"]
    result = _base("MODEL_IMPROVEMENT_V2_PHASE59_REPORTING-v1", source)
    result.update({
        "analysis": "MODEL_IMPROVEMENT_V2_FINAL_CONCLUSIONS",
        "final_policy": closure["final_policy"],
        "post_test_retuning": closure["governance"]["post_test_retuning"],
        "attention_conclusion_for_v2": "NOT_CLAIMED",
        "phase58_status": phase58["status"],
        "rows": [
            {"conclusion": "Final model policy", "result": closure["final_policy"]["policy_id"], "status": "LOCKED"},
            {"conclusion": "Post-hoc RMSE Wh", "result": ensemble["rmse_wh"], "status": "POST_HOC_V2_BENCHMARK"},
            {"conclusion": "Post-hoc MAE Wh", "result": ensemble["mae_wh"], "status": "POST_HOC_V2_BENCHMARK"},
            {"conclusion": "Post-hoc MAPE %", "result": ensemble["mape_pct"], "status": "POST_HOC_V2_BENCHMARK"},
            {"conclusion": "Post-hoc R²", "result": ensemble["r2"], "status": "POST_HOC_V2_BENCHMARK"},
            {"conclusion": "RMSE vs Persistence Wh", "result": ensemble["rmse_wh"] - persistence["rmse_wh"], "status": "LOWER_IS_BETTER"},
            {"conclusion": "MAE vs Persistence Wh", "result": ensemble["mae_wh"] - persistence["mae_wh"], "status": "LOWER_IS_BETTER"},
            {"conclusion": "V2 attention behavior", "result": "NOT_EVALUATED", "status": "NO_V2_TEST_ATTENTION_TENSORS"},
            {"conclusion": "Post-Test retuning", "result": False, "status": "PASS"},
        ],
    })
    return result


def build_reporting_artifacts(root: Path | None = None) -> dict[str, Any]:
    """Validate frozen sources and atomically write the V2 reporting package."""
    root = (root or project_root()).resolve()
    data, source = _verify_sources(root)
    phase48 = _phase48(root, data, source)
    phase49 = _phase49(data, source)
    phase50, assignments = _phase50(root, data, source)
    phase51 = _phase51(data, assignments, source)
    phase58 = _phase58(source, phase48, phase49, phase50, phase51)
    phase59 = _phase59(source, phase58)
    documents = {
        "phase48_prediction_analysis.json": phase48,
        "phase49_residual_analysis.json": phase49,
        "phase50_error_by_regime.json": phase50,
        "phase51_worst_error_analysis.json": phase51,
        "phase58_final_summary.json": phase58,
        "phase59_final_conclusions.json": phase59,
    }
    output_dir = root / OUTPUT_ROOT
    for filename, document in documents.items():
        _atomic_json(output_dir / filename, document)
    output_sha = {filename: _sha256(output_dir / filename) for filename in documents}
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_FINAL_REPORTING_MANIFEST-v1",
        "status": "PASS",
        "lineage": dict(LINEAGE),
        "output_root": OUTPUT_ROOT.as_posix(),
        "source_population_fingerprint": source["population_fingerprint"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_mape_addendum_manifest_sha256": source["mape_addendum_manifest_sha256"],
        "source_metrics_with_mape_sha256": source["metrics_with_mape_sha256"],
        "source_final_lock_sha256": source["final_lock_sha256"],
        "source_final_closure_sha256": source["final_closure_sha256"],
        "source_prediction_sha256": source["prediction_sha256"],
        "output_sha256": output_sha,
        "phase52_57_policy": "V1_HISTORICAL_ATTENTION_NOT_RECOMPUTED_FOR_FINAL_V2",
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root())
    args = parser.parse_args(argv)
    manifest = build_reporting_artifacts(args.project_root)
    print(json.dumps({
        "status": manifest["status"],
        "output_root": manifest["output_root"],
        "training_executed": False,
        "inference_executed": False,
        "raw_test_source_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
