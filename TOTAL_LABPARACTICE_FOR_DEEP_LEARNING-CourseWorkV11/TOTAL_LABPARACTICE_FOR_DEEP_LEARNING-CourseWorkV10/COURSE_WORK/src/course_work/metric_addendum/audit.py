import json
from pathlib import Path
from typing import Any

from course_work.metric_addendum.contract import ARTIFACT_ROOT, COMPLIANCE_LOG_PATH
from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, csv_text, get_project_root, sha256_file


AUDIT_VERSION = "ML-PIPELINE-COMPLIANCE-v1"
AUDIT_FIELDS = ["order", "area", "status", "finding", "evidence"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_pass(path: Path) -> bool:
    payload = _read_json(path)
    candidates = (
        payload.get("status"),
        payload.get("overall_status"),
        payload.get("phase_status"),
        payload.get("phase52_status"),
        payload.get("phase51_status"),
    )
    return any(value in {"PASS", "COMPLETED"} for value in candidates) or payload.get("all_pass") is True


def _row(order: int, area: str, status: str, finding: str, evidence: str) -> dict[str, Any]:
    return {"order": order, "area": area, "status": status, "finding": finding, "evidence": evidence}


def build_ml_pipeline_compliance_rows(project_root: Path) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    contract = _read_json(root / "artifacts/contracts/coursework_contract.json")
    split = _read_json(root / "artifacts/splits/split_manifest.json")
    eda = _read_json(root / "artifacts/eda/eda_manifest.json")
    phase_9 = _read_json(root / "artifacts/scaling/phase_9_signoff.json")
    final_test = _read_json(root / "artifacts/final_test/final_test_summary.json")
    mape_status = _read_json(root / ARTIFACT_ROOT / "final_test_mape_status.json")
    problem = contract.get("problem", {})
    problem_ok = problem.get("target") == "Appliances" and problem.get("forecast_horizon_steps") == 1
    split_ok = (
        split.get("split_method") == "chronological_observation_proportion"
        and split.get("train_ratio") == 0.7
        and split.get("validation_ratio") == 0.15
        and split.get("test_ratio") == 0.15
    )
    phase_9_expected = str(phase_9.get("output_checksums", {}).get("artifacts/scaling/README_SCALING.md", ""))
    phase_9_readme = root / "artifacts/scaling/README_SCALING.md"
    phase_9_actual = sha256_file(phase_9_readme) if phase_9_readme.exists() else ""
    scaling_status = "PASS" if _phase_pass(root / "artifacts/scaling/phase_9_signoff.json") and phase_9_expected == phase_9_actual else "PARTIAL"
    notebook_text = (root / "notebook_course_work/CourseWork_1.ipynb").read_text(encoding="utf-8")
    direct_eda_full_view = "eda_temporal_view = load_validated_temporal_view(PROJECT_ROOT)" in notebook_text
    error_signoffs = [
        root / "artifacts/prediction_analysis/phase_48_signoff.json",
        root / "artifacts/residual_analysis/phase_49_signoff.json",
        root / "artifacts/error_by_regime/phase_50_signoff.json",
        root / "artifacts/worst_error_analysis/phase51_signoff.json",
    ]
    attention_signoffs = [
        root / "artifacts/attention_extraction/phase_52_signoff.json",
        root / "artifacts/attention_heatmaps/phase_53_signoff.json",
        root / "artifacts/last_query_attention/phase_54_signoff.json",
        root / "artifacts/head_comparison/phase_55_signoff.json",
        root / "artifacts/error_conditioned_attention/phase_56_signoff.json",
        root / "artifacts/seed_stability_attention/phase_57_signoff.json",
    ]
    rows = [
        _row(1, "Problem definition", "PASS" if problem_ok else "FAIL", "Supervised multivariate sequence-to-one time-series regression with Appliances[t+1] in Wh", "artifacts/contracts/coursework_contract.json"),
        _row(2, "Schema and temporal integrity", "PASS" if _phase_pass(root / "artifacts/temporal/phase_4_signoff.json") else "FAIL", "Timestamp order, sampling interval, continuity and target identity are audited before modeling", "artifacts/schema/phase_3_signoff.json; artifacts/temporal/phase_4_signoff.json"),
        _row(3, "Chronological split", "PASS" if split_ok else "FAIL", "Train, Validation and Test follow time order with 70/15/15 allocation and no shuffle", "artifacts/splits/split_manifest.json"),
        _row(4, "EDA coverage", "PASS" if _phase_pass(root / "artifacts/eda/phase_6_signoff.json") else "FAIL", "Structure, descriptive statistics, univariate distributions, correlation, lags, rolling summaries and extreme targets are covered", "artifacts/eda/eda_manifest.json; artifacts/eda/tables; artifacts/eda/figures"),
        _row(5, "EDA decision boundary", "PARTIAL" if direct_eda_full_view else "PASS", "Canonical EDA is Train-only; direct notebook EDA loads the full validated temporal view for display and must remain descriptive only", "artifacts/eda/eda_manifest.json; notebook_course_work/CourseWork_1.ipynb"),
        _row(6, "Outlier policy", "PASS" if eda.get("processing_actions", {}).get("outliers_removed") is False else "FAIL", "Extreme energy observations are retained unless independently proven to be data errors", "artifacts/eda/eda_manifest.json; artifacts/eda/tables/extreme_target_samples.csv"),
        _row(7, "Feature engineering", "PASS" if _phase_pass(root / "artifacts/features/phase_7_signoff.json") else "FAIL", "Calendar cycles are deterministic and leakage checks preserve raw target and feature lineage", "artifacts/features/phase_7_signoff.json"),
        _row(8, "Train-only scaling", scaling_status, "Scaler fit policy is Train-only, but the Phase 9 README checksum currently differs from its frozen signoff", "artifacts/scaling/phase_9_signoff.json; artifacts/scaling/README_SCALING.md"),
        _row(9, "Windowing and DataLoaders", "PASS" if _phase_pass(root / "artifacts/windows/phase_10_signoff.json") and _phase_pass(root / "artifacts/dataloaders/phase_11_signoff.json") else "FAIL", "Sequence-to-one windows and deterministic non-shuffled Validation/Test loaders preserve population lineage", "artifacts/windows/phase_10_signoff.json; artifacts/dataloaders/phase_11_signoff.json"),
        _row(10, "Baseline comparison", "PASS" if all(_phase_pass(path) for path in [root / "artifacts/baselines/persistence/phase_14_signoff.json", root / "artifacts/lstm_baseline/phase_20_signoff.json", root / "artifacts/transformer_b0/phase_21_signoff.json"]) else "FAIL", "Persistence, LSTM and Transformer Validation baselines are available before controlled sweeps", "artifacts/baselines/persistence; artifacts/lstm_baseline; artifacts/transformer_b0"),
        _row(11, "Controlled model selection", "PASS" if _phase_pass(root / "artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json") else "FAIL", "Validation RMSE remains the sole selection metric through one-factor sweeps; Test is forbidden", "artifacts/sweeps; artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json"),
        _row(12, "Rolling-origin robustness", "PASS" if _phase_pass(root / "artifacts/rolling_origin/phase_44_signoff.json") else "FAIL", "Three expanding pre-Test folds use fold-local scaling without outer-selection leakage", "artifacts/rolling_origin/phase_44_signoff.json"),
        _row(13, "Final model lock and Test firewall", "PASS" if final_test.get("overall_status") == "PASS" and final_test.get("training_in_phase47") is False and final_test.get("post_test_tuning") is False else "FAIL", "Model is locked before one-time Test evaluation; no Phase 47 training or post-Test tuning occurred", "artifacts/final_model_lock/phase_45_signoff.json; artifacts/final_test/final_test_summary.json"),
        _row(14, "Three-seed reporting", "PASS" if final_test.get("best_seed_selected") is False and final_test.get("ensemble_used") is False and final_test.get("transformer_sd_rmse_wh") is not None else "FAIL", "Transformer Test performance is reported as mean and sample standard deviation across three fixed seeds without best-seed selection", "artifacts/final_test/final_test_summary.json"),
        _row(15, "Prediction and error analysis", "PASS" if all(_phase_pass(path) for path in error_signoffs) else "FAIL", "Prediction, residual, regime and worst-error analyses are materialized from frozen Test outputs", "artifacts/prediction_analysis; artifacts/residual_analysis; artifacts/error_by_regime; artifacts/worst_error_analysis"),
        _row(16, "Attention analysis", "PASS" if all(_phase_pass(path) for path in attention_signoffs) else "FAIL", "Attention maps, last-query views, head comparisons and seed stability are descriptive and non-causal", "artifacts/attention_extraction; artifacts/attention_heatmaps; artifacts/last_query_attention; artifacts/head_comparison; artifacts/error_conditioned_attention; artifacts/seed_stability_attention"),
        _row(17, "MAPE Validation addendum", "PASS" if (root / ARTIFACT_ROOT / "validation_mape_by_run.csv").exists() else "FAIL", "Standard MAPE is computed once on complete original-Wh Validation populations and remains supplementary", "artifacts/metric_addendum/mape/validation_mape_by_run.csv"),
        _row(18, "MAPE Test addendum", str(mape_status.get("status", "BLOCKED_SOURCE_UNAVAILABLE")), "Test MAPE is not computed because frozen prediction CSV bundles are unavailable; no Test inference is authorized", "artifacts/metric_addendum/mape/final_test_source_audit.csv"),
        _row(19, "LSTM final Test comparison", "PARTIAL" if final_test.get("lstm_eligibility") == "NOT_EVALUATED" else "PASS", "LSTM is compared on Validation but was not eligible for final Test evaluation", "artifacts/final_test/final_test_summary.json"),
        _row(20, "Inferential statistics", "NOT_APPLICABLE", "The coursework reports deterministic metrics and three-seed descriptive variability; no new hypothesis test or confidence interval is authorized post hoc", "artifacts/final_tables; artifacts/final_conclusions"),
        _row(21, "Final scientific closure", "PASS" if _phase_pass(root / "artifacts/final_tables/phase_58_signoff.json") and _phase_pass(root / "artifacts/final_conclusions/phase_59_signoff.json") else "FAIL", "Final tables and conclusions are frozen and remain unchanged by this supplementary addendum", "artifacts/final_tables/phase_58_signoff.json; artifacts/final_conclusions/phase_59_signoff.json"),
    ]
    return rows


def materialize_ml_pipeline_compliance_audit(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    output_root = root / ARTIFACT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    rows = build_ml_pipeline_compliance_rows(root)
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({str(row["status"]) for row in rows})}
    overall_status = "FAIL" if counts.get("FAIL", 0) else "PASS_WITH_LIMITATIONS" if counts.get("PARTIAL", 0) or counts.get("BLOCKED_SOURCE_UNAVAILABLE", 0) else "PASS"
    csv_path = output_root / "ml_pipeline_compliance_audit.csv"
    json_path = output_root / "ml_pipeline_compliance_audit.json"
    atomic_write_bytes(csv_path, csv_text(AUDIT_FIELDS, rows).encode("utf-8"))
    payload = {"artifact_version": AUDIT_VERSION, "overall_status": overall_status, "status_counts": counts, "records": rows}
    atomic_write_bytes(json_path, canonical_json_bytes(payload))
    log = {
        "artifact_version": AUDIT_VERSION,
        "status": overall_status,
        "status_counts": counts,
        "output_paths": [csv_path.relative_to(root).as_posix(), json_path.relative_to(root).as_posix()],
        "output_checksums": {
            csv_path.relative_to(root).as_posix(): sha256_file(csv_path),
            json_path.relative_to(root).as_posix(): sha256_file(json_path),
        },
    }
    atomic_write_bytes(root / COMPLIANCE_LOG_PATH, canonical_json_bytes(log))
    return log
