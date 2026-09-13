"""Human-gated MAPE addendum over frozen Step 17 prediction artifacts.

Preflight validates metadata and checksums only.  The evaluation mode reads
Test-derived prediction CSVs, but never opens the raw Test source, loads a
checkpoint, or runs model inference.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from course_work.model_improvement_v2.step17_post_hoc_benchmark import (
    BENCHMARK_LABEL,
    EXPECTED_TEST_COUNT,
    EXPECTED_TEST_POPULATION_FINGERPRINT,
    FINAL_SEEDS,
    OUTPUT_ROOT,
    Step17Error,
    _atomic_json,
    _metrics,
    _population_fingerprint,
    _read_json,
    project_root,
)
from course_work.utils.artifacts import sha256_file


EXPERIMENT_ID = "STEP17_MAPE_ADDENDUM"
ADDENDUM_LABEL = "POST_HOC_V2_BENCHMARK_MAPE_ADDENDUM"
STEP17_MANIFEST = OUTPUT_ROOT / "step17_benchmark_manifest.json"
STEP17_METRICS = OUTPUT_ROOT / "step17_metrics.json"
OUTPUT_METRICS = OUTPUT_ROOT / "step17_metrics_with_mape.json"
OUTPUT_MANIFEST = OUTPUT_ROOT / "step17_mape_addendum_manifest.json"
OUTPUT_SIGNOFF = OUTPUT_ROOT / "step17_mape_addendum_signoff.json"
ACCESS_EVENT = OUTPUT_ROOT / "step17_mape_test_derived_access_event.json"

SOURCE_FILES = {
    "V2_FINAL_SEED_42": "predictions/seed_42.csv",
    "V2_FINAL_SEED_123": "predictions/seed_123.csv",
    "V2_FINAL_SEED_2026": "predictions/seed_2026.csv",
    "V2_FINAL_EQUAL_WEIGHT_ENSEMBLE": "predictions/ensemble.csv",
    "PERSISTENCE_LAST_VALUE": "predictions/persistence.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_index(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = document.get("metrics")
    if not isinstance(records, list):
        raise Step17Error("Step 17 metrics list is missing")
    indexed = {str(row.get("model_id")): dict(row) for row in records if isinstance(row, dict)}
    if set(indexed) != set(SOURCE_FILES):
        raise Step17Error("Step 17 metric model set drift")
    return indexed


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    manifest_path = root / STEP17_MANIFEST
    metrics_path = root / STEP17_METRICS
    manifest = _read_json(manifest_path)
    metrics = _read_json(metrics_path)
    if manifest.get("benchmark_label") != BENCHMARK_LABEL or manifest.get("status") != "PASS":
        raise Step17Error("Canonical Step 17 manifest is not PASS")
    if sha256_file(metrics_path) != manifest.get("metrics_sha256"):
        raise Step17Error("Canonical Step 17 metrics checksum mismatch")
    _metric_index(metrics)
    predictions = manifest.get("prediction_sha256")
    if not isinstance(predictions, dict):
        raise Step17Error("Canonical prediction checksum ledger missing")
    verified: list[dict[str, str]] = []
    for model_id, relative in SOURCE_FILES.items():
        path = root / OUTPUT_ROOT / relative
        key = str(path.relative_to(root))
        expected = predictions.get(key)
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise Step17Error(f"Frozen prediction checksum mismatch: {model_id}")
        verified.append({"model_id": model_id, "path": key, "sha256": expected})
    return {
        "experiment": EXPERIMENT_ID,
        "status": "PASS",
        "benchmark_label": ADDENDUM_LABEL,
        "source_benchmark_label": BENCHMARK_LABEL,
        "prediction_sources_verified": verified,
        "expected_rows_per_source": EXPECTED_TEST_COUNT,
        "expected_population_fingerprint": EXPECTED_TEST_POPULATION_FINGERPRINT,
        "raw_test_source_opened": False,
        "checkpoint_payloads_loaded": 0,
        "training_executed": False,
        "inference_executed": False,
        "test_derived_prediction_rows_read": 0,
        "evaluation_authorized": False,
    }


def _read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["target_id", "target_timestamp", "y_true_wh", "y_pred_wh"])
    if len(frame) != EXPECTED_TEST_COUNT:
        raise Step17Error(f"Unexpected prediction row count: {path}")
    if frame["target_id"].astype(str).duplicated().any():
        raise Step17Error(f"Duplicate target IDs: {path}")
    for column in ("y_true_wh", "y_pred_wh"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise Step17Error(f"Non-finite {column}: {path}")
    return frame


def run_evaluation(root: Path | None = None) -> int:
    root = root or project_root()
    preflight = run_preflight(root)
    for output in (OUTPUT_METRICS, OUTPUT_MANIFEST, OUTPUT_SIGNOFF, ACCESS_EVENT):
        if (root / output).exists():
            raise Step17Error(f"MAPE addendum already exists; refusing overwrite: {output}")

    original_metrics = _read_json(root / STEP17_METRICS)
    original_index = _metric_index(original_metrics)
    source_manifest = _read_json(root / STEP17_MANIFEST)
    source_checksums = source_manifest["prediction_sha256"]
    frames: dict[str, pd.DataFrame] = {}
    canonical_ids: list[str] | None = None
    canonical_timestamps: list[str] | None = None
    canonical_truth: np.ndarray | None = None
    for model_id, relative in SOURCE_FILES.items():
        path = root / OUTPUT_ROOT / relative
        frame = _read_prediction(path)
        ids = frame["target_id"].astype(str).tolist()
        timestamps = frame["target_timestamp"].astype(str).tolist()
        truth = frame["y_true_wh"].to_numpy(dtype=np.float64)
        if canonical_ids is None:
            canonical_ids, canonical_timestamps, canonical_truth = ids, timestamps, truth
        elif ids != canonical_ids or timestamps != canonical_timestamps or not np.array_equal(truth, canonical_truth):
            raise Step17Error(f"Prediction population mismatch: {model_id}")
        frames[model_id] = frame

    assert canonical_ids is not None and canonical_truth is not None
    if _population_fingerprint(canonical_ids) != EXPECTED_TEST_POPULATION_FINGERPRINT:
        raise Step17Error("Frozen prediction population fingerprint mismatch")

    rows: list[dict[str, Any]] = []
    for model_id, frame in frames.items():
        computed = _metrics(canonical_truth, frame["y_pred_wh"].to_numpy(dtype=np.float64))
        prior = original_index[model_id]
        for key in ("sample_count", "rmse_wh", "mae_wh", "r2"):
            if not np.isclose(float(computed[key]), float(prior[key]), rtol=0.0, atol=1e-12):
                raise Step17Error(f"Frozen Step 17 metric mismatch for {model_id}: {key}")
        if computed["mape_status"] != "DEFINED":
            raise Step17Error(f"MAPE undefined for frozen source: {model_id}")
        rows.append({**prior, "mape_pct": computed["mape_pct"], "mape_status": computed["mape_status"]})

    _atomic_json(root / ACCESS_EVENT, {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_MAPE_ACCESS_EVENT-v1",
        "benchmark_label": ADDENDUM_LABEL,
        "access_scope": "FROZEN_TEST_DERIVED_PREDICTION_ARTIFACTS_ONLY",
        "raw_test_source_opened": False,
        "checkpoint_payloads_loaded": 0,
        "inference_executed": False,
        "created_at": _utc_now(),
    })
    payload = {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_METRICS_WITH_MAPE-v1",
        "benchmark_label": ADDENDUM_LABEL,
        "source_benchmark_label": BENCHMARK_LABEL,
        "source_metrics_sha256": sha256_file(root / STEP17_METRICS),
        "population_fingerprint": EXPECTED_TEST_POPULATION_FINGERPRINT,
        "metrics": rows,
        "best_seed_selected": False,
        "post_test_retuning": False,
        "training_executed": False,
        "inference_executed": False,
        "raw_test_source_opened": False,
        "test_status": "READ_FROZEN_POST_HOC_V2_PREDICTIONS_FOR_MAPE",
    }
    _atomic_json(root / OUTPUT_METRICS, payload)
    _atomic_json(root / OUTPUT_MANIFEST, {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_MAPE_MANIFEST-v1",
        "benchmark_label": ADDENDUM_LABEL,
        "source_manifest_sha256": sha256_file(root / STEP17_MANIFEST),
        "source_prediction_sha256": source_checksums,
        "metrics_sha256": sha256_file(root / OUTPUT_METRICS),
        "access_event_sha256": sha256_file(root / ACCESS_EVENT),
        "status": "PASS",
    })
    _atomic_json(root / OUTPUT_SIGNOFF, {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_MAPE_SIGNOFF-v1",
        "benchmark_label": ADDENDUM_LABEL,
        "status": "PASS",
        "manifest_sha256": sha256_file(root / OUTPUT_MANIFEST),
        "human_interpretation_required": True,
        "unbiased_unseen_test_claim": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-gated MAPE addendum over frozen Step 17 predictions")
    parser.add_argument("--experiment", required=True, choices=[EXPERIMENT_ID])
    parser.add_argument("--mode", required=True, choices=["preflight", "evaluate"])
    parser.add_argument("--authorize-test-derived-evidence-access", action="store_true")
    parser.add_argument("--acknowledge-post-hoc-not-unseen-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flags = args.authorize_test_derived_evidence_access or args.acknowledge_post_hoc_not_unseen_test
    if args.mode == "preflight":
        if flags:
            print("ERROR: authorization flags are forbidden in preflight", flush=True)
            return 2
        try:
            print(json.dumps(run_preflight(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            return 1
    if not (args.authorize_test_derived_evidence_access and args.acknowledge_post_hoc_not_unseen_test):
        print("REFUSED: evaluation requires explicit Human authorization and post-hoc acknowledgement", flush=True)
        return 3
    try:
        return run_evaluation()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
