from pathlib import Path
from typing import Any

from course_work.metric_addendum.compute import compute_validation_mape_records, summarize_validation_mape
from course_work.metric_addendum.contract import ADDENDUM_VERSION, ARTIFACT_ROOT, PROCESS_LOG_PATH, build_addendum_contract
from course_work.metric_addendum.sources import inspect_final_test_sources
from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
    sha256_file,
)


VALIDATION_FIELDS = [
    "run_id",
    "model_id",
    "model_family",
    "split_id",
    "n_samples",
    "mape_pct",
    "mape_status",
    "zero_target_count",
    "target_min_wh",
    "target_max_wh",
    "target_unit",
    "direction",
    "primary_selection",
    "metric_version",
    "metric_contract_fingerprint",
    "population_fingerprint",
    "prediction_path",
    "prediction_sha256",
    "metrics_path",
    "metrics_sha256",
]
TEST_SOURCE_FIELDS = [
    "source_id",
    "path",
    "expected_sha256",
    "actual_sha256",
    "expected_rows",
    "actual_rows",
    "status",
]


def _write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    atomic_write_bytes(path, csv_text(fieldnames, rows).encode("utf-8"))


def materialize_mape_addendum(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    output_root = root / ARTIFACT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    contract = build_addendum_contract()
    validation_records = compute_validation_mape_records(root)
    validation_summary = summarize_validation_mape(validation_records)
    test_sources = inspect_final_test_sources(root)
    test_ready = bool(test_sources) and all(row["status"] == "PASS" for row in test_sources)
    test_status = {
        "artifact_version": ADDENDUM_VERSION,
        "status": "READY_FOR_DERIVATION" if test_ready else "BLOCKED_SOURCE_UNAVAILABLE",
        "prediction_sources_expected": len(test_sources),
        "prediction_sources_valid": sum(row["status"] == "PASS" for row in test_sources),
        "test_inference_executed": False,
        "test_mape_computed": False,
        "reason": "FROZEN_PREDICTION_BUNDLES_AVAILABLE" if test_ready else "FROZEN_PREDICTION_BUNDLES_MISSING_OR_INVALID",
    }
    discrepancies = [
        {
            "code": "FINAL_TEST_PREDICTION_SOURCE_UNAVAILABLE",
            "severity": "BLOCKING_TEST_EXTENSION",
            "detail": row["path"],
        }
        for row in test_sources
        if row["status"] != "PASS"
    ]
    paths = {
        "contract": output_root / "mape_metric_contract.json",
        "validation_results": output_root / "validation_mape_by_run.csv",
        "validation_summary": output_root / "validation_mape_summary.json",
        "validation_source_audit": output_root / "validation_source_audit.csv",
        "final_test_source_audit": output_root / "final_test_source_audit.csv",
        "final_test_status": output_root / "final_test_mape_status.json",
        "discrepancies": output_root / "mape_discrepancies.json",
    }
    _write_json(paths["contract"], contract)
    _write_csv(paths["validation_results"], VALIDATION_FIELDS, validation_records)
    _write_json(paths["validation_summary"], {"artifact_version": ADDENDUM_VERSION, **validation_summary})
    _write_csv(paths["validation_source_audit"], VALIDATION_FIELDS, validation_records)
    _write_csv(paths["final_test_source_audit"], TEST_SOURCE_FIELDS, test_sources)
    _write_json(paths["final_test_status"], test_status)
    _write_json(paths["discrepancies"], {"artifact_version": ADDENDUM_VERSION, "records": discrepancies})
    output_checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in paths.values()
    }
    signoff = {
        "artifact_version": ADDENDUM_VERSION,
        "status": "PASS_WITH_BLOCKED_TEST_EXTENSION" if validation_records and not test_ready else "PASS",
        "validation_status": "PASS" if validation_records else "BLOCKED_SOURCE_UNAVAILABLE",
        "test_status": test_status["status"],
        "selection_metric_unchanged": "rmse_wh",
        "training_executed": False,
        "test_inference_executed": False,
        "frozen_artifacts_modified": False,
        "output_checksums": output_checksums,
        "discrepancy_count": len(discrepancies),
    }
    signoff_path = output_root / "mape_addendum_signoff.json"
    _write_json(signoff_path, signoff)
    log = {
        "artifact_version": ADDENDUM_VERSION,
        "status": signoff["status"],
        "validation_summary": validation_summary,
        "test_status": test_status,
        "output_paths": [path.relative_to(root).as_posix() for path in [*paths.values(), signoff_path]],
        "output_checksums": {
            **output_checksums,
            signoff_path.relative_to(root).as_posix(): sha256_file(signoff_path),
        },
    }
    _write_json(root / PROCESS_LOG_PATH, log)
    return log
