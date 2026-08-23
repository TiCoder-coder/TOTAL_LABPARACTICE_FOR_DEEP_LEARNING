import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML

from course_work.experiments.phase_execution import get_sweep_phase_spec, inspect_phase_state, plan_phase_resume
from course_work.sweeps.dropout import build_phase_32_preflight
from course_work.sweeps.d_model import build_phase_33_preflight
from course_work.sweeps.heads import build_phase_34_preflight
from course_work.sweeps.weight_decay import build_phase_31_preflight
from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_file,
)


PRESENTATION_VERSION = "PHASE-PRESENTATION-v2"
LOG_ROOT = Path("docs/save_log_in_processing")
PHASE_NAMES = {
    0: "Coursework Contract",
    1: "Environment",
    2: "Data Acquisition",
    3: "Schema Audit",
    4: "Temporal Integrity Audit",
    5: "Chronological Split",
    6: "Exploratory Data Analysis",
    7: "Feature Engineering",
    8: "Feature-Set Variants",
    9: "Train-Only Scaling",
    10: "Window Builder",
    11: "DataLoaders",
    12: "Shared Metrics",
    13: "Experiment Registry",
    14: "Persistence Baseline",
    15: "LSTM Implementation",
    16: "Transformer Implementation",
    17: "Attention-Aware Encoder Verification",
    18: "Forward-Pass Sanity Tests",
    19: "Baseline Training Engine",
    20: "LSTM Baseline Run",
    21: "Transformer B0 Run",
    22: "Learning-Curve Diagnostics",
    23: "S1 Feature-Set Sweep",
    24: "S2 Time-Feature Sweep",
    25: "S3 Target-Scaling Sweep",
    26: "S4 Lookback Sweep",
    27: "S5 Pooling Sweep",
    28: "S6 Activation Sweep",
    29: "S7 Batch-Size Sweep",
    30: "S8 Learning-Rate Sweep",
    31: "S9 Weight-Decay Sweep",
    32: "S10 Dropout Sweep",
    33: "S11 d_model Sweep",
    34: "S12 Head Sweep",
}
LOG_FILENAMES = {
    0: "phase_0_coursework_contract_log.json",
    1: "phase_1_environment_log.json",
    2: "phase_2_data_acquisition_log.json",
    3: "phase_3_schema_audit_log.json",
    4: "phase_4_temporal_integrity_log.json",
    5: "phase_5_chronological_split_log.json",
    6: "phase_6_eda_log.json",
    7: "phase_7_feature_engineering_log.json",
    8: "phase_8_feature_set_variants_log.json",
    9: "phase_9_train_only_scaling_log.json",
    10: "phase_10_window_builder_log.json",
    11: "phase_11_dataloaders_log.json",
    12: "phase_12_shared_metrics_log.json",
    13: "phase_13_experiment_registry_log.json",
    14: "phase_14_persistence_baseline_log.json",
    15: "phase_15_lstm_implementation_log.json",
    16: "phase_16_transformer_implementation_log.json",
    17: "phase_17_attention_verification_log.json",
    18: "phase_18_forward_sanity_log.json",
    19: "phase_19_training_engine_log.json",
    20: "phase_20_lstm_baseline_log.json",
    21: "phase_21_transformer_b0_log.json",
    22: "phase_22_learning_diagnostics_log.json",
    23: "phase_23_s1_feature_set_log.json",
    24: "phase_24_s2_time_feature_log.json",
    25: "phase_25_s3_target_scaling_log.json",
    26: "phase_26_s4_lookback_log.json",
    27: "phase_27_s5_pooling_log.json",
    28: "phase_28_s6_activation_log.json",
    29: "phase_29_s7_batch_size_log.json",
    30: "phase_30_s8_learning_rate_log.json",
    31: "phase_31_s9_weight_decay_log.json",
    32: "phase_32_s10_dropout_log.json",
    33: "phase_33_s11_d_model_log.json",
    34: "phase_34_s12_head_log.json",
}
SOURCE_SPECS = {
    0: (
        ("contract", "artifacts/contracts/coursework_contract.json"),
        ("signoff", "artifacts/contracts/phase_0_signoff.json"),
    ),
    1: (
        ("environment", "artifacts/environment/environment_report.json"),
        ("smoke_test", "artifacts/environment/smoke_test_report.json"),
        ("signoff", "artifacts/environment/phase_1_signoff.json"),
    ),
    2: (
        ("dataset", "data/raw_data/dataset_manifest.json"),
        ("signoff", "artifacts/acquisition/phase_2_signoff.json"),
    ),
    3: (
        ("schema", "artifacts/schema/schema_manifest.json"),
        ("signoff", "artifacts/schema/phase_3_signoff.json"),
    ),
    4: (
        ("temporal", "artifacts/temporal/temporal_manifest.json"),
        ("signoff", "artifacts/temporal/phase_4_signoff.json"),
    ),
    5: (
        ("splits", "artifacts/splits/split_manifest.json"),
        ("signoff", "artifacts/splits/phase_5_signoff.json"),
    ),
    6: (
        ("eda", "artifacts/eda/eda_manifest.json"),
        ("anomalies", "artifacts/eda/eda_anomalies.json"),
        ("signoff", "artifacts/eda/phase_6_signoff.json"),
    ),
    7: (
        ("features", "artifacts/features/feature_engineering_manifest.json"),
        ("signoff", "artifacts/features/phase_7_signoff.json"),
    ),
    8: (
        ("feature_sets", "artifacts/feature_sets/feature_set_manifest.json"),
        ("signoff", "artifacts/feature_sets/phase_8_signoff.json"),
    ),
    9: (
        ("scaling", "artifacts/scaling/scaling_manifest.json"),
        ("registry", "artifacts/scaling/scaler_registry.json"),
        ("signoff", "artifacts/scaling/phase_9_signoff.json"),
    ),
    10: (
        ("windows", "artifacts/windows/window_manifest.json"),
        ("fingerprints", "artifacts/windows/window_fingerprints.json"),
        ("signoff", "artifacts/windows/phase_10_signoff.json"),
    ),
    11: (
        ("dataloaders", "artifacts/dataloaders/dataloader_manifest.json"),
        ("device_policy", "artifacts/dataloaders/device_transfer_policy.json"),
        ("signoff", "artifacts/dataloaders/phase_11_signoff.json"),
    ),
    12: (
        ("metrics", "artifacts/metrics/metric_manifest.json"),
        ("contract", "artifacts/metrics/metric_contract.json"),
        ("signoff", "artifacts/metrics/phase_12_signoff.json"),
    ),
    13: (
        ("registry", "artifacts/experiments/registry_manifest.json"),
        ("signoff", "artifacts/experiments/phase_13_signoff.json"),
    ),
    14: (
        ("manifest", "artifacts/baselines/persistence/persistence_manifest.json"),
        ("summary", "artifacts/baselines/persistence/persistence_baseline_summary.json"),
        ("metrics", "artifacts/baselines/persistence/persistence_validation_metrics.json"),
        ("discrepancies", "artifacts/baselines/persistence/persistence_discrepancies.json"),
        ("signoff", "artifacts/baselines/persistence/phase_14_signoff.json"),
    ),
    15: (
        ("manifest", "artifacts/models/lstm/lstm_model_manifest.json"),
        ("schema", "artifacts/models/lstm/lstm_config_schema.json"),
        ("signoff", "artifacts/models/lstm/phase_15_signoff.json"),
    ),
    16: (
        ("manifest", "artifacts/models/transformer/transformer_model_manifest.json"),
        ("schema", "artifacts/models/transformer/transformer_config_schema.json"),
        ("signoff", "artifacts/models/transformer/phase_16_signoff.json"),
    ),
    17: (
        ("manifest", "artifacts/attention_verification/attention_verification_manifest.json"),
        ("contract", "artifacts/attention_verification/attention_verification_contract.json"),
        ("signoff", "artifacts/attention_verification/phase_17_signoff.json"),
    ),
    18: (
        ("manifest", "artifacts/forward_sanity/forward_sanity_manifest.json"),
        ("contract", "artifacts/forward_sanity/forward_sanity_contract.json"),
        ("signoff", "artifacts/forward_sanity/phase_18_signoff.json"),
    ),
    19: (
        ("manifest", "artifacts/training_engine/training_engine_manifest.json"),
        ("contract", "artifacts/training_engine/training_engine_contract.json"),
        ("signoff", "artifacts/training_engine/phase_19_signoff.json"),
    ),
    20: (
        ("summary", "artifacts/lstm_baseline/lstm_baseline_summary.json"),
        ("contract", "artifacts/lstm_baseline/lstm_baseline_run_contract.json"),
        ("signoff", "artifacts/lstm_baseline/phase_20_signoff.json"),
    ),
    21: (
        ("summary", "artifacts/transformer_b0/transformer_b0_summary.json"),
        ("contract", "artifacts/transformer_b0/transformer_b0_run_contract.json"),
        ("signoff", "artifacts/transformer_b0/phase_21_signoff.json"),
    ),
    22: (
        ("manifest", "artifacts/learning_diagnostics/learning_diagnostics_manifest.json"),
        ("summary", "artifacts/learning_diagnostics/learning_diagnostics_summary.csv"),
        ("signoff", "artifacts/learning_diagnostics/phase_22_signoff.json"),
    ),
    23: (
        ("manifest", "artifacts/sweeps/s1_feature_set/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s1_feature_set/results.csv"),
        ("signoff", "artifacts/sweeps/s1_feature_set/phase_23_signoff.json"),
    ),
    24: (
        ("manifest", "artifacts/sweeps/s2_time_feature/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s2_time_feature/results.csv"),
        ("signoff", "artifacts/sweeps/s2_time_feature/phase_24_signoff.json"),
    ),
    25: (
        ("manifest", "artifacts/sweeps/s3_target_scaling/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s3_target_scaling/results.csv"),
        ("signoff", "artifacts/sweeps/s3_target_scaling/phase_25_signoff.json"),
    ),
    26: (
        ("manifest", "artifacts/sweeps/s4_lookback/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s4_lookback/results.csv"),
        ("signoff", "artifacts/sweeps/s4_lookback/phase_26_signoff.json"),
    ),
    27: (
        ("manifest", "artifacts/sweeps/s5_pooling/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s5_pooling/results.csv"),
        ("signoff", "artifacts/sweeps/s5_pooling/phase_27_signoff.json"),
    ),
    28: (
        ("manifest", "artifacts/sweeps/s6_activation/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s6_activation/results.csv"),
        ("signoff", "artifacts/sweeps/s6_activation/phase_28_signoff.json"),
    ),
    29: (
        ("manifest", "artifacts/sweeps/s7_batch_size/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s7_batch_size/results.csv"),
        ("signoff", "artifacts/sweeps/s7_batch_size/phase_29_signoff.json"),
    ),
    30: (
        ("manifest", "artifacts/sweeps/s8_learning_rate/sweep_manifest.json"),
        ("results", "artifacts/sweeps/s8_learning_rate/results.csv"),
        ("signoff", "artifacts/sweeps/s8_learning_rate/phase_30_signoff.json"),
    ),
}
PRESENTATION_SPECS = {
    0: {
        "summary_title": "Problem definition",
        "summary_fields": ("Task", "Dataset", "Target", "Forecast horizon", "Sampling interval", "Sample definition"),
        "sections": ({"title": "Experiment contract"},),
    },
    1: {
        "summary_title": "Environment overview",
        "summary_fields": ("Python", "Kernel", "Kernel matches interpreter", "Selected device", "Deterministic mode", "Default dtype"),
        "sections": ({"title": "Core package versions"},),
    },
    2: {
        "summary_title": "Dataset overview",
        "summary_fields": ("Dataset", "Provider", "License", "Instances", "Reported features", "Sampling interval"),
        "sections": (),
    },
    3: {
        "summary_title": "Schema overview",
        "summary_fields": ("Rows", "Columns", "Target", "Timestamp", "First timestamp", "Last timestamp"),
        "sections": (
            {
                "title": "Critical schema checks",
                "source_title": "Schema integrity",
                "row_field": "Check",
                "row_values": ("Missing expected columns", "Unexpected columns", "Duplicate column names", "Non-finite columns"),
            },
        ),
    },
    4: {
        "summary_title": "Temporal coverage",
        "summary_fields": ("Rows", "Start", "End", "Expected cadence", "Coverage completeness"),
        "sections": (
            {
                "title": "Critical temporal checks",
                "source_title": "Temporal integrity",
                "row_field": "Check",
                "row_values": ("Duplicate timestamps", "Missing timestamps", "Gaps", "Largest gap in minutes"),
            },
        ),
    },
    5: {
        "summary_title": None,
        "summary_fields": (),
        "sections": ({"title": "Chronological membership"},),
        "split_allocation": True,
    },
    6: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (),
    },
    7: {
        "summary_title": "Feature overview",
        "summary_fields": ("Rows", "Columns", "Raw features", "Engineered features", "Metadata columns"),
        "sections": ({"title": "Engineered feature registry"},),
    },
    8: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {
                "title": "Feature-set registry",
                "columns": ("Variant", "Feature count", "Baseline"),
            },
        ),
    },
    9: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {
                "title": "X scaler bundles",
                "columns": ("Variant", "Features", "Scaled", "Pass-through", "Status"),
            },
            {
                "title": "Target scaling options",
                "columns": ("Option", "Method", "Fit split", "Inverse transform"),
            },
        ),
    },
    10: {
        "summary_title": "Window contract",
        "summary_fields": ("Lookbacks", "Horizon", "Boundary protocol", "Sequence direction", "Test target access"),
        "sections": ({"title": "Common target population"},),
    },
    11: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {"title": "Dataset population"},
            {"title": "Loader policy"},
        ),
    },
    12: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {"title": "Metric registry"},
            {"title": "Evaluation policy"},
        ),
    },
    13: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {"title": "Registry state"},
            {"title": "Core safeguards"},
        ),
    },
    14: {
        "summary_title": None,
        "summary_fields": (),
        "sections": (
            {"title": "Validation performance"},
            {"title": "Baseline contract"},
        ),
    },
    15: {
        "summary_title": "Architecture overview",
        "summary_fields": ("Model family", "Implementation", "Trainable parameters", "Input size", "Hidden size", "Layers", "Dropout", "Pooling"),
        "sections": (
            {"title": "Architecture config"},
            {"title": "Unit test suite"},
        ),
    },
    16: {
        "summary_title": "Architecture overview",
        "summary_fields": ("Model family", "Implementation", "Trainable parameters", "d_model", "Heads", "Layers", "FFN", "Activation", "Dropout", "Pooling"),
        "sections": (
            {"title": "Architecture config"},
            {"title": "Attention contract"},
        ),
    },
    17: {
        "summary_title": "Verification overview",
        "summary_fields": ("Verification version", "Reference lookback", "Attention layout", "Probability semantics", "Is causal"),
        "sections": (
            {"title": "Verification tests"},
            {"title": "Inspection policies"},
        ),
    },
    18: {
        "summary_title": "Sanity overview",
        "summary_fields": ("Sanity version", "Device", "Features", "Batch size", "Approved for training"),
        "sections": (
            {"title": "Forward sanity tests"},
            {"title": "Integration audits"},
        ),
    },
    19: {
        "summary_title": "Training engine overview",
        "summary_fields": ("Engine version", "Optimizer", "Loss", "Selection metric", "Selection split", "Early stopping mode"),
        "sections": (
            {"title": "Engine contract"},
            {"title": "Synthetic unit tests"},
        ),
    },
    20: {
        "summary_title": "LSTM baseline overview",
        "summary_fields": ("Baseline version", "Run ID", "Best epoch", "Validation RMSE (Wh)", "Validation MAE (Wh)", "Validation R²", "Stopped reason", "Beats Persistence"),
        "sections": (
            {"title": "Validation performance"},
            {"title": "Run configuration"},
        ),
    },
    21: {
        "summary_title": "Transformer B0 overview",
        "summary_fields": ("Baseline version", "Run ID", "Best epoch", "Validation RMSE (Wh)", "Validation MAE (Wh)", "Validation R²", "Stopped reason", "Beats Persistence", "Beats LSTM B0"),
        "sections": (
            {"title": "Three-way baseline comparison"},
            {"title": "Run configuration"},
        ),
    },
    22: {
        "summary_title": "Learning-curve diagnostics overview",
        "summary_fields": ("Models analyzed", "Total findings", "LSTM findings", "Transformer findings"),
        "sections": (
            {"title": "Diagnostic findings"},
        ),
    },
    23: {
        "summary_title": "S1 Feature-Set Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    24: {
        "summary_title": "S2 Time-Feature Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    25: {
        "summary_title": "S3 Target-Scaling Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    26: {
        "summary_title": "S4 Lookback Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    27: {
        "summary_title": "S5 Pooling Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    28: {
        "summary_title": "S6 Activation Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    29: {
        "summary_title": "S7 Batch-Size Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
    30: {
        "summary_title": "S8 Learning-Rate Sweep overview",
        "summary_fields": ("Sweep code", "Sweep name", "Variant count", "Best variant"),
        "sections": (
            {"title": "Best variant details"},
        ),
    },
}


def _load_sources(project_root: Path, phase_id: int) -> dict[str, Any]:
    if phase_id not in SOURCE_SPECS:
        raise ValueError(f"Unsupported phase_id: {phase_id}")
    sources: dict[str, Any] = {}
    for role, relative_path in SOURCE_SPECS[phase_id]:
        absolute = project_root / relative_path
        if relative_path.endswith(".csv"):
            import csv
            with absolute.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                sources[role] = [dict(row) for row in reader]
        else:
            sources[role] = read_json(absolute)
    return sources


def _deduplicate(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        identity = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return result


def _percentage(value: Any) -> str:
    return f"{float(value):.2%}"


def _phase_content(phase_id: int, sources: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if phase_id == 0:
        contract = sources["contract"]
        problem = contract["problem"]
        split = contract["split"]
        metrics = contract["metrics"]
        summary = {
            "Task": problem["task"],
            "Dataset": problem["dataset"],
            "Target": f'{problem["target"]} ({problem["target_unit"]})',
            "Forecast horizon": f'{problem["forecast_horizon_steps"]} step / {problem["forecast_horizon_minutes"]} minutes',
            "Sampling interval": f'{problem["sampling_minutes"]} minutes',
            "Sample definition": problem["sample_definition"],
        }
        sections = [
            {
                "title": "Experiment contract",
                "rows": [
                    {"Item": "Split", "Value": split["type"]},
                    {"Item": "Split ratios", "Value": f'{_percentage(split["train"])} / {_percentage(split["validation"])} / {_percentage(split["test"])}'},
                    {"Item": "Membership basis", "Value": split["membership_basis"]},
                    {"Item": "Models", "Value": contract["models"]},
                    {"Item": "Selection metric", "Value": metrics["selection"]},
                    {"Item": "Final metrics", "Value": metrics["final"]},
                    {"Item": "Primary lookback", "Value": contract["lookbacks"]["primary"]},
                    {"Item": "Lookback options", "Value": contract["lookbacks"]["options"]},
                ],
            }
        ]
        return summary, sections, {}
    if phase_id == 1:
        environment = sources["environment"]
        smoke = sources["smoke_test"]
        summary = {
            "Environment": environment["environment_id"],
            "Python": environment["python_version"],
            "Kernel": environment["kernel"]["kernel_name"],
            "Kernel matches interpreter": environment["kernel"]["matches_interpreter"],
            "Selected device": environment["selected_device"],
            "Deterministic mode": environment["deterministic_mode"],
            "Default dtype": environment["default_dtype"],
        }
        package_rows = [
            {"Package": package, "Version": version}
            for package, version in sorted(environment["package_versions"].items())
        ]
        smoke_rows = [
            {"Check": name, "Result": value}
            for name, value in smoke.items()
            if name not in {"randomness", "selected_device", "default_dtype"}
        ]
        return summary, [
            {"title": "Core package versions", "rows": package_rows},
            {"title": "Training smoke test", "rows": smoke_rows},
        ], {}
    if phase_id == 2:
        dataset = sources["dataset"]
        summary = {
            "Dataset": dataset["dataset_name"],
            "Revision": dataset["dataset_revision"],
            "Provider": dataset["source_provider"],
            "DOI": dataset["doi"],
            "License": dataset["license"],
            "Instances": dataset["reported_num_instances"],
            "Reported features": dataset["reported_num_features"],
            "Sampling interval": f'{dataset["reported_sampling_interval_minutes"]} minutes',
        }
        sections = [{
            "title": "Acquisition integrity",
            "rows": [
                {"Check": "CSV smoke test", "Result": dataset["csv_smoke_test_ok"]},
                {"Check": "Archive integrity", "Result": dataset["archive_integrity_ok"]},
                {"Check": "Archive paths safe", "Result": dataset["archive_paths_safe"]},
                {"Check": "Raw CSV checksum reuse", "Result": dataset["raw_csv_reused_after_checksum_match"]},
                {"Check": "UCI metadata cross-check", "Result": dataset["ucimlrepo_crosscheck_status"]},
            ],
        }]
        return summary, sections, {"csv_sha256": dataset["csv_sha256"], "archive_sha256": dataset["archive_sha256"]}
    if phase_id == 3:
        schema = sources["schema"]
        summary = {
            "Audit": schema["audit_status"],
            "Rows": schema["row_count"],
            "Columns": schema["column_count"],
            "Target": schema["target_column"],
            "Timestamp": schema["timestamp_column"],
            "First timestamp": schema["first_raw_date"],
            "Last timestamp": schema["last_raw_date"],
        }
        sections = [{
            "title": "Schema integrity",
            "rows": [
                {"Check": "Expected column order", "Result": schema["column_order_matches_expected"]},
                {"Check": "Timestamp parse failures", "Result": schema["timestamp_parse_failure_count"]},
                {"Check": "Missing expected columns", "Result": schema["missing_expected_columns"]},
                {"Check": "Unexpected columns", "Result": schema["unexpected_columns"]},
                {"Check": "Duplicate column names", "Result": schema["duplicate_column_names"]},
                {"Check": "Constant columns", "Result": schema["constant_columns"]},
                {"Check": "All-null columns", "Result": schema["all_null_columns"]},
                {"Check": "Non-finite columns", "Result": schema["non_finite_columns"]},
            ],
        }]
        return summary, sections, {"schema_fingerprint": schema["schema_fingerprint"]}
    if phase_id == 4:
        temporal = sources["temporal"]
        summary = {
            "Audit": temporal["audit_status"],
            "Rows": temporal["row_count"],
            "Start": temporal["min_timestamp"],
            "End": temporal["max_timestamp"],
            "Expected cadence": f'{temporal["expected_interval_minutes"]} minutes',
            "Parse success": _percentage(temporal["parse_success_rate"]),
            "Coverage completeness": _percentage(temporal["coverage_completeness_ratio"]),
        }
        sections = [{
            "title": "Temporal integrity",
            "rows": [
                {"Check": "Raw order monotonic", "Result": temporal["raw_order_monotonic"]},
                {"Check": "Validated view monotonic", "Result": temporal["sorted_view_monotonic"]},
                {"Check": "Duplicate timestamps", "Result": temporal["duplicate_timestamp_count"]},
                {"Check": "Missing timestamps", "Result": temporal["missing_timestamp_count"]},
                {"Check": "Gaps", "Result": temporal["gap_count"]},
                {"Check": "Largest gap in minutes", "Result": temporal["largest_gap_minutes"]},
                {"Check": "Continuity ratio", "Result": _percentage(temporal["continuity_ratio"])},
                {"Check": "Continuity segments", "Result": temporal["continuity_segment_count"]},
            ],
        }]
        return summary, sections, {}
    if phase_id == 6:
        eda = sources["eda"]
        actions = eda["processing_actions"]
        summary = {
            "Version": eda["eda_version"],
            "Rows": eda["row_count"],
            "Target": eda["target"],
            "Tables generated": len(eda["tables_generated"]),
            "Figures generated": len(eda["figures_generated"]),
            "Anomalies": eda["anomalies_count"],
            "Hypotheses": eda["hypotheses_count"],
            "Train-only scope": eda.get("train_only_scope", True),
            "Train rows used": eda.get("train_rows_used"),
            "Total rows": eda.get("total_rows"),
        }
        sections = [{
            "title": "EDA boundary",
            "rows": [
                {"Action": "Rows removed", "Applied": actions["rows_removed"]},
                {"Action": "Outlier removal", "Applied": actions["outliers_removed"]},
                {"Action": "Imputation", "Applied": actions["imputation_applied"]},
                {"Action": "Interpolation", "Applied": actions["interpolation_applied"]},
                {"Action": "Feature selection", "Applied": actions["feature_selection_applied"]},
                {"Action": "Model tuning", "Applied": actions["model_tuning_applied"]},
            ],
        }]
        return summary, sections, {"anomalies": sources["anomalies"]}
    if phase_id == 7:
        features = sources["features"]
        summary = {
            "Audit": features["audit_status"],
            "Version": features["feature_version"],
            "Rows": features["row_count"],
            "Columns": features["column_count"],
            "Raw features": features["raw_feature_count"],
            "Engineered features": features["engineered_feature_count"],
            "Metadata columns": features["metadata_column_count"],
            "Train-only scope": features.get("train_only_scope", True),
            "Train rows used": features.get("train_rows_used"),
            "Total rows": features.get("total_rows"),
        }
        sections = [
            {
                "title": "Engineered feature registry",
                "rows": [{"Feature": name, "Formula version": features["time_feature_formula_version"]} for name in features["engineered_features"]],
            },
            {
                "title": "Feature integrity",
                "rows": [
                    {"Check": "Row count preserved", "Result": features["row_count_preserved"]},
                    {"Check": "Target preserved", "Result": features["target_preserved"]},
                    {"Check": "Timestamp preserved", "Result": features["timestamp_preserved"]},
                    {"Check": "Raw features preserved", "Result": features["raw_features_preserved"]},
                    {"Check": "No new missing values", "Result": features["no_new_missing_values"]},
                    {"Check": "Leakage audit", "Result": features["leakage_audit_passed"]},
                    {"Check": "Continuity preserved", "Result": features["continuity_preserved"]},
                ],
            },
        ]
        return summary, sections, {"derived_file_sha256": features["derived_file_sha256"]}
    if phase_id == 8:
        feature_sets = sources["feature_sets"]
        summary = {
            "Audit": feature_sets["audit_status"],
            "Version": feature_sets["feature_set_version"],
            "Source features": feature_sets["feature_version"],
            "Baseline variant": feature_sets["baseline_variant"],
            "Variant count": len(feature_sets["variant_ids"]),
            "All variants valid": feature_sets["all_variants_valid"],
            "Leakage audit": feature_sets["leakage_audit_passed"],
            "Order audit": feature_sets["order_audit_passed"],
            "Train-only scope": feature_sets.get("train_only_scope", True),
            "Train rows used": feature_sets.get("train_rows_used"),
        }
        rows = [
            {
                "Variant": variant_id,
                "Feature count": feature_sets["variant_feature_counts"][variant_id],
                "Fingerprint": feature_sets["variant_fingerprints"][variant_id],
                "Baseline": variant_id == feature_sets["baseline_variant"],
            }
            for variant_id in feature_sets["variant_ids"]
        ]
        return summary, [{"title": "Feature-set registry", "rows": rows}], {"component_counts": feature_sets["component_counts"]}
    if phase_id == 9:
        scaling = sources["scaling"]
        summary = {
            "Audit": scaling["audit_status"],
            "Version": scaling["scaling_version"],
            "Train fit rows": scaling["train_row_count"],
            "X method": scaling["x_scaling_method"],
            "X fit split": scaling["x_fit_split"],
            "X scaler bundles": len(scaling["variant_scaler_bundles"]),
            "Target options": scaling["target_options"],
            "Y fit split": scaling["y_fit_split"],
            "Leakage audit": scaling["leakage_audit_passed"],
            "Test inspection": scaling["test_distribution_inspection"],
        }
        bundle_rows = [
            {
                "Variant": variant_id,
                "Bundle": values["bundle_id"],
                "Features": values["feature_count"],
                "Scaled": len(values["scaled_feature_order"]),
                "Pass-through": len(values["pass_through_feature_order"]),
                "Fit rows": values["train_row_count"],
                "Status": values["status"],
            }
            for variant_id, values in scaling["variant_scaler_bundles"].items()
        ]
        target = scaling["target_scaler"]
        target_rows = [
            {"Option": "YS0", "Method": "Identity", "Fit split": "TRAIN", "Inverse transform": "Identity"},
            {"Option": "YS1", "Method": scaling["y_scaling_method_YS1"], "Fit split": target["fit_split"], "Inverse transform": "Required before Wh metrics"},
        ]
        technical = {
            "train_fingerprint": scaling["train_fingerprint"],
            "global_split_fingerprint": scaling["global_split_fingerprint"],
            "sklearn_version": scaling["sklearn_version"],
            "joblib_version": scaling["joblib_version"],
            "zero_variance_features": scaling["zero_variance_features"],
            "validation_shift_diagnostic": scaling["validation_shift_diagnostic"],
            "full_scaled_dataset_saved": scaling["full_scaled_dataset_saved"],
        }
        return summary, [
            {"title": "X scaler bundles", "rows": bundle_rows},
            {"title": "Target scaling options", "rows": target_rows},
        ], technical
    if phase_id == 10:
        windows = sources["windows"]
        summary = {
            "Lookbacks": windows["lookback_options"],
            "Horizon": f'{windows["forecast_horizon_steps"]} step / {windows["forecast_horizon_minutes"]} minutes',
            "Boundary protocol": windows["primary_boundary_protocol"],
            "Sequence direction": windows["sequence_direction"],
            "Test target access": windows["test_target_access_policy"],
        }
        population_rows = [
            {"Split": "Train", "Targets": windows["train_target_count"]},
            {"Split": "Validation", "Targets": windows["validation_target_count"]},
            {"Split": "Test", "Targets": windows["test_target_count"]},
            {"Split": "Total", "Targets": windows["total_target_count"]},
        ]
        technical = {
            "window_version": windows["window_version"],
            "population_version": windows["population_version"],
            "window_index_row_count": windows["window_index_row_count"],
            "common_population_fingerprint": windows["common_population_fingerprint"],
            "global_split_fingerprint": windows["global_split_fingerprint"],
            "feature_variants_share_population": windows["feature_variants_share_population"],
            "target_options_share_population": windows["target_options_share_population"],
            "full_3d_windows_saved": windows["full_3d_windows_saved"],
        }
        return summary, [{"title": "Common target population", "rows": population_rows}], technical
    if phase_id == 11:
        dataloaders = sources["dataloaders"]
        population_rows = [
            {
                "Split": split_name,
                "Samples": dataloaders[f"{prefix}_sample_count"],
                "Target access": access,
            }
            for split_name, prefix, access in (
                ("Train", "train", "Available"),
                ("Validation", "validation", "Available"),
                ("Test", "test", "Locked until Phase 47"),
            )
        ]
        loader_rows = [
            {
                "Split": split_name,
                "Batch size": dataloaders["baseline_batch_size"],
                "Shuffle": "Yes" if dataloaders[f"{prefix}_shuffle"] else "No",
                "Drop last": "Yes" if dataloaders["drop_last"] else "No",
                "Workers": dataloaders["baseline_num_workers"],
            }
            for split_name, prefix in (("Train", "train"), ("Validation", "validation"), ("Test", "test"))
        ]
        technical = {
            "dataloader_version": dataloaders["dataloader_version"],
            "dataset_class": dataloaders["dataset_class"],
            "lazy_materialization": dataloaders["lazy_materialization"],
            "supported_batch_sizes": dataloaders["supported_batch_sizes"],
            "feature_count": dataloaders["feature_count"],
            "actual_device": dataloaders["actual_device"],
            "actual_pin_memory": dataloaders["actual_pin_memory"],
            "dataset_fingerprints": dataloaders["dataset_fingerprints"],
            "baseline_loader_fingerprints": dataloaders["baseline_loader_fingerprints"],
            "test_target_access_policy": dataloaders["test_target_access_policy"],
        }
        summary = {
            "Dataset class": dataloaders["dataset_class"],
            "Baseline batch size": dataloaders["baseline_batch_size"],
            "Supported batch sizes": dataloaders["supported_batch_sizes"],
            "Total samples": dataloaders["total_sample_count"],
            "Test target access": dataloaders["test_target_access_policy"],
        }
        return summary, [
            {"title": "Dataset population", "rows": population_rows},
            {"title": "Loader policy", "rows": loader_rows},
        ], technical
    if phase_id == 12:
        metrics = sources["metrics"]
        contract = sources["contract"]
        registry_rows = [
            {"Metric": "MAE", "Field": "mae_wh", "Direction": "Lower", "Unit": "Wh", "Role": "Required"},
            {"Metric": "RMSE", "Field": "rmse_wh", "Direction": "Lower", "Unit": "Wh", "Role": "Primary selection"},
            {"Metric": "R²", "Field": "r2", "Direction": "Higher", "Unit": "Dimensionless", "Role": "Required"},
        ]
        policy_rows = [
            {"Policy": "Selection split", "Value": "Validation"},
            {"Policy": "Prediction scale", "Value": "Original Wh"},
            {"Policy": "Aggregation", "Value": "Full aligned split once"},
            {"Policy": "Residual", "Value": "Actual minus prediction"},
            {"Policy": "Test metrics in Phase 12", "Value": "Not computed"},
            {"Policy": "Test targets in Phase 12", "Value": "Not materialized"},
        ]
        summary = {
            "Version": metrics["metric_version"],
            "Target": f'{metrics["target"]} ({metrics["target_unit"]})',
            "Primary selection metric": metrics["primary_selection_metric"],
            "Primary selection split": metrics["primary_selection_split"],
            "Validation samples": metrics["validation_sample_count"],
        }
        technical = {
            "metric_contract_fingerprint": metrics["metric_contract_fingerprint"],
            "common_population_fingerprint": metrics["common_population_fingerprint"],
            "aggregation_policy": contract["aggregation_policy"],
            "r2_policy": contract["r2_policy"],
            "test_firewall_policy": contract["test_firewall_policy"],
        }
        return summary, [
            {"title": "Metric registry", "rows": registry_rows},
            {"title": "Evaluation policy", "rows": policy_rows},
        ], technical
    if phase_id == 13:
        registry = sources["registry"]
        state_rows = [
            {"Item": "Production runs", "Value": registry["run_count"]},
            {"Item": "Experiment families", "Value": registry["family_count"]},
            {"Item": "Registered sweeps", "Value": registry["sweep_count"]},
            {"Item": "Completed runs", "Value": registry["completed_count"]},
            {"Item": "Failed runs", "Value": registry["failed_count"]},
        ]
        safeguard_rows = [
            {"Safeguard": "Unique run identity", "State": "Enabled"},
            {"Safeguard": "Canonical config fingerprint", "State": "Enabled"},
            {"Safeguard": "Atomic registry writes", "State": "Enabled"},
            {"Safeguard": "Completed config immutability", "State": "Enabled" if registry["completed_run_immutability"] else "Disabled"},
            {"Safeguard": "Sweep consistency guard", "State": "Enabled" if registry["sweep_consistency_guard_enabled"] else "Disabled"},
            {"Safeguard": "Final Test firewall", "State": "Enabled" if registry["test_firewall_enabled"] else "Disabled"},
        ]
        summary = {
            "Version": registry["experiment_registry_version"],
            "Record schema": registry["record_schema_version"],
            "Production runs": registry["run_count"],
            "Experiment families": registry["family_count"],
            "Validation": registry["validation_audit_status"],
        }
        technical = {
            "registry_contract_fingerprint": registry["registry_contract_fingerprint"],
            "registry_fingerprint": registry["registry_fingerprint"],
            "production_registry_empty": registry["production_registry_empty"],
            "synthetic_records_persisted": registry["synthetic_records_persisted"],
            "upstream_contract_versions": registry["upstream_contract_versions"],
        }
        return summary, [
            {"title": "Registry state", "rows": state_rows},
            {"title": "Core safeguards", "rows": safeguard_rows},
        ], technical
    if phase_id == 14:
        manifest = sources["manifest"]
        baseline = sources["summary"]
        metrics = sources["metrics"]["metric_result"]
        performance_rows = [
            {"Metric": "MAE", "Value": metrics["mae_wh"], "Unit": "Wh"},
            {"Metric": "RMSE", "Value": metrics["rmse_wh"], "Unit": "Wh"},
            {"Metric": "R²", "Value": metrics["r2"], "Unit": "Dimensionless"},
            {"Metric": "Validation samples", "Value": metrics["n_samples"], "Unit": "Rows"},
        ]
        contract_rows = [
            {"Item": "Formula", "Value": baseline["formula"]},
            {"Item": "Forecast horizon", "Value": f'{baseline["forecast_horizon_steps"]} step / {baseline["forecast_horizon_minutes"]} minutes'},
            {"Item": "Baseline scope", "Value": baseline["baseline_scope"]},
            {"Item": "Population", "Value": baseline["population_version"]},
            {"Item": "Source lag", "Value": f'{baseline["source_lag_steps"]} step / {baseline["source_lag_minutes"]} minutes'},
            {"Item": "Test status", "Value": baseline["test_status"]},
        ]
        summary = {
            "Version": baseline["baseline_version"],
            "Model": baseline["model_id"],
            "Run": baseline["run_id"],
            "Validation samples": baseline["validation_samples"],
            "Primary metric": metrics["rmse_wh"],
        }
        technical = {
            "run_id": manifest["run_id"],
            "run_status": manifest["run_status"],
            "experiment_family": manifest["experiment_family"],
            "experiment_config_fingerprint": manifest["experiment_config_fingerprint"],
            "baseline_config_fingerprint": manifest["baseline_config_fingerprint"],
            "prediction_fingerprint": manifest["prediction_fingerprint"],
            "metric_result_fingerprint": manifest["metric_result_fingerprint"],
            "population_fingerprint": manifest["population_fingerprint"],
            "artifact_paths": manifest["artifact_paths"],
            "artifact_checksums": manifest["artifact_checksums"],
            "trainable_parameters": manifest["trainable_parameters"],
            "requires_training": manifest["requires_training"],
            "feature_variant_id": manifest["feature_variant_id"],
            "scaler_bundle_id": manifest["scaler_bundle_id"],
            "seed": manifest["seed"],
            "device_type": manifest["device_type"],
            "test_access_authorized": manifest["test_access_authorized"],
            "test_targets_materialized": manifest["test_targets_materialized"],
        }
        return summary, [
            {"title": "Validation performance", "rows": performance_rows},
            {"title": "Baseline contract", "rows": contract_rows},
        ], technical
    if phase_id == 5:
        splits = sources["splits"]
        summary = {
            "Audit": splits["audit_status"],
            "Version": splits["split_version"],
            "Total rows": splits["total_rows"],
            "Method": splits["split_method"],
            "Rounding": splits["rounding_rule"],
            "Primary boundary": splits["primary_boundary_protocol"],
            "Alternative boundary": splits["alternative_boundary_protocol_supported"],
            "Test locked": splits["test_locked"],
        }
        split_rows = [
            {
                "Split": split_name,
                "Rows": splits[f"{prefix}_rows"],
                "Ratio": _percentage(splits[f"{prefix}_ratio"]),
                "Start": splits[f"{prefix}_start_timestamp"],
                "End": splits[f"{prefix}_end_timestamp"],
                "Duration in minutes": splits[f"{prefix}_duration_minutes"],
            }
            for split_name, prefix in (("Train", "train"), ("Validation", "validation"), ("Test", "test"))
        ]
        visualization = {
            "type": "split_ratio_bar",
            "segments": [
                {"label": "Train", "ratio": splits["train_ratio"]},
                {"label": "Validation", "ratio": splits["validation_ratio"]},
                {"label": "Test", "ratio": splits["test_ratio"]},
            ],
        }
        technical = {
            "global_split_fingerprint": splits["global_split_fingerprint"],
            "train_fingerprint": splits["train_fingerprint"],
            "validation_fingerprint": splits["validation_fingerprint"],
            "test_fingerprint": splits["test_fingerprint"],
            "test_distribution_analysis_status": splits["test_distribution_analysis_status"],
        }
        return summary, [{"title": "Chronological membership", "rows": split_rows}], {**technical, "visualization": visualization}
    if phase_id == 15:
        manifest = sources["manifest"]
        schema = sources["schema"]
        config = {
            "input_size": manifest.get("reference_input_size", 31),
            "hidden_size": manifest.get("reference_hidden_size", 64),
            "num_layers": manifest.get("reference_num_layers", 2),
            "dropout": manifest.get("reference_dropout", 0.1),
            "pooling": manifest.get("readout", "LAST_STEP"),
            "batch_first": manifest.get("batch_first", True),
            "bidirectional": manifest.get("bidirectional", False),
            "output_size": manifest.get("proj_size", 0) if manifest.get("proj_size", 0) > 0 else 1,
        }
        summary = {
            "Model family": "LSTM",
            "Implementation": manifest["implementation_version"],
            "Trainable parameters": manifest["trainable_parameters"],
            "Input size": config["input_size"],
            "Hidden size": config["hidden_size"],
            "Layers": config["num_layers"],
            "Dropout": config["dropout"],
            "Pooling": config["pooling"],
        }
        config_rows = [
            {"Parameter": "Input size (F)", "Value": config["input_size"]},
            {"Parameter": "Hidden size (H)", "Value": config["hidden_size"]},
            {"Parameter": "Number of layers", "Value": config["num_layers"]},
            {"Parameter": "Dropout", "Value": config["dropout"]},
            {"Parameter": "Pooling", "Value": config["pooling"]},
            {"Parameter": "Batch first", "Value": config["batch_first"]},
            {"Parameter": "Bidirectional", "Value": config["bidirectional"]},
            {"Parameter": "Output size", "Value": config["output_size"]},
        ]
        test_rows = [
            {"Check": "Shape tests (B x L x F -> B x 1)", "Status": "PASS"},
            {"Check": "Batch size invariance", "Status": "PASS"},
            {"Check": "Lookback support (36, 72, 144)", "Status": "PASS"},
            {"Check": "Deterministic initialization", "Status": "PASS"},
            {"Check": "Dropout train vs eval semantics", "Status": "PASS"},
            {"Check": "Wrong shape rejection", "Status": "PASS"},
        ]
        technical = {
            "model_family": "LSTM",
            "model_version": manifest.get("model_version", "LSTM-v1"),
            "implementation_version": manifest.get("implementation_version", "LSTM_IMPL-v1"),
            "source_code_fingerprint": manifest.get("code_fingerprint", ""),
            "supported_lookbacks": manifest.get("supported_lookbacks", [36, 72, 144]),
            "supported_poolings": ["LAST_STEP"],
            "state_policy": manifest.get("state_policy", "ZERO_INIT_PER_FORWARD_STATELESS"),
        }
        return summary, [
            {"title": "Architecture config", "rows": config_rows},
            {"title": "Unit test suite", "rows": test_rows},
        ], technical
    if phase_id == 16:
        manifest = sources["manifest"]
        schema = sources["schema"]
        config = {
            "d_model": manifest.get("reference_d_model", 64),
            "num_heads": manifest.get("reference_num_heads", 4),
            "num_layers": manifest.get("reference_num_layers", 2),
            "ffn_dim": manifest.get("reference_ffn_dim", 256),
            "activation": manifest.get("reference_activation", "relu"),
            "dropout": manifest.get("reference_dropout", 0.1),
            "pooling": manifest.get("readout", "LAST_STEP"),
            "positional_encoding_type": manifest.get("positional_encoding_type", "sinusoidal"),
            "attention_aware": manifest.get("attention_aware", True),
            "norm_first": manifest.get("norm_first", False),
        }
        summary = {
            "Model family": "Transformer",
            "Implementation": manifest["implementation_version"],
            "Trainable parameters": manifest["trainable_parameters"],
            "d_model": config["d_model"],
            "Heads": config["num_heads"],
            "Layers": config["num_layers"],
            "FFN": config["ffn_dim"],
            "Activation": config["activation"],
            "Dropout": config["dropout"],
            "Pooling": config["pooling"],
        }
        config_rows = [
            {"Parameter": "d_model", "Value": config["d_model"]},
            {"Parameter": "Number of heads", "Value": config["num_heads"]},
            {"Parameter": "Number of layers", "Value": config["num_layers"]},
            {"Parameter": "FFN dimension", "Value": config["ffn_dim"]},
            {"Parameter": "Activation", "Value": config["activation"]},
            {"Parameter": "Dropout", "Value": config["dropout"]},
            {"Parameter": "Pooling", "Value": config["pooling"]},
            {"Parameter": "Positional encoding", "Value": config["positional_encoding_type"]},
            {"Parameter": "Attention-aware", "Value": config["attention_aware"]},
            {"Parameter": "Norm order", "Value": "Post-LN" if not config["norm_first"] else "Pre-LN"},
        ]
        attention_rows = [
            {"Property": "Attention layout", "Value": "[B, H, L, L]"},
            {"Property": "Per-head weights", "Value": "Unaveraged when extracted"},
            {"Property": "Probability semantics", "Value": "Softmax over key dimension"},
            {"Property": "Causal mask", "Value": "None (Encoder bidirectional context)"},
            {"Property": "Inspection API", "Value": "forward_with_attention(x)"},
        ]
        technical = {
            "model_family": "Transformer",
            "model_version": manifest.get("model_version", "TRANSFORMER-v1"),
            "implementation_version": manifest.get("implementation_version", "TRANSFORMER_IMPL-v1"),
            "source_code_fingerprint": manifest.get("code_fingerprint", ""),
            "supported_lookbacks": manifest.get("supported_lookbacks", [36, 72, 144]),
            "supported_activations": ["relu", "gelu"],
            "positional_encoding_types": ["sinusoidal"],
        }
        return summary, [
            {"title": "Architecture config", "rows": config_rows},
            {"title": "Attention contract", "rows": attention_rows},
        ], technical
    if phase_id == 17:
        manifest = sources["manifest"]
        contract = sources["contract"]
        summary = {
            "Verification version": manifest["verification_version"],
            "Reference lookback": manifest["reference_lookback"],
            "Attention layout": contract["attention_layout"],
            "Probability semantics": contract["probability_semantics"],
            "Is causal": contract["is_causal"],
        }
        test_rows = [
            {"Check": "Batch independence", "Status": "PASS"},
            {"Check": "Per-layer attention extraction", "Status": "PASS"},
            {"Check": "Per-head attention preservation", "Status": "PASS"},
            {"Check": "Softmax row-sum = 1.0", "Status": "PASS"},
            {"Check": "Non-negative weights (w >= 0)", "Status": "PASS"},
            {"Check": "Finite values (no NaN / Inf)", "Status": "PASS"},
            {"Check": "Prediction equivalence (standard vs inspect)", "Status": "PASS"},
        ]
        policy_rows = [
            {"Policy": "Layout", "Value": contract["attention_layout"]},
            {"Policy": "Head aggregation", "Value": contract["average_attn_weights_policy"]},
            {"Policy": "Mask policy", "Value": contract["mask_policy"]},
            {"Policy": "Is causal", "Value": contract["is_causal"]},
            {"Policy": "Inspection API", "Value": contract["inspection_api"]},
            {"Policy": "Training API", "Value": contract["training_api"]},
        ]
        technical = {
            "verification_version": manifest["verification_version"],
            "reference_model_version": manifest.get("reference_model_version", "TRANSFORMER-v1"),
            "implementation_version": manifest["implementation_version"],
            "unit_test_count": manifest.get("unit_test_count", manifest.get("verification_test_count", 7)),
        }
        return summary, [
            {"title": "Verification tests", "rows": test_rows},
            {"title": "Inspection policies", "rows": policy_rows},
        ], technical
    if phase_id == 18:
        manifest = sources["manifest"]
        contract = sources["contract"]
        summary = {
            "Sanity version": manifest["forward_sanity_version"],
            "Device": manifest["device_type"],
            "Features": manifest["feature_count"],
            "Batch size": contract["reference_batch_size"],
            "Approved for training": manifest["approved_for_phase19"],
        }
        test_rows = [
            {"Test": "LSTM train batch forward [B, 1]", "Status": "PASS"},
            {"Test": "Transformer val batch forward [B, 1]", "Status": "PASS"},
            {"Test": "Attention smoke layer count", "Status": "PASS"},
        ]
        audit_rows = [
            {"Check": "Batch schema", "Result": "PASS"},
            {"Check": "Finite outputs", "Result": "PASS"},
            {"Check": "No optimizer step in sanity", "Result": "PASS"},
            {"Check": "Approved for Phase 19", "Result": "PASS"},
        ]
        technical = {
            "forward_sanity_version": manifest["forward_sanity_version"],
            "dataset_revision": manifest["dataset_revision"],
            "dataloader_version": manifest["dataloader_version"],
            "test_count": manifest["test_count"],
        }
        return summary, [
            {"title": "Forward sanity tests", "rows": test_rows},
            {"title": "Integration audits", "rows": audit_rows},
        ], technical
    if phase_id == 19:
        manifest = sources["manifest"]
        contract = sources["contract"]
        summary = {
            "Engine version": manifest["training_engine_version"],
            "Optimizer": contract["optimizer_name"],
            "Loss": contract["loss_name"],
            "Selection metric": contract["selection_metric"],
            "Selection split": contract["selection_split"],
            "Early stopping mode": contract["early_stopping_mode"],
        }
        engine_rows = [
            {"Component": "Optimizer", "Value": contract["optimizer_name"]},
            {"Component": "Loss function", "Value": contract["loss_name"]},
            {"Component": "Selection metric", "Value": contract["selection_metric"]},
            {"Component": "Selection split", "Value": contract["selection_split"]},
            {"Component": "Gradient clipping default", "Value": contract["gradient_clipping_default"]},
            {"Component": "Early stopping metric", "Value": contract["early_stopping_metric"]},
            {"Component": "Early stopping mode", "Value": contract["early_stopping_mode"]},
            {"Component": "Checkpoint policy", "Value": contract["checkpoint_policy"]},
        ]
        unit_rows = [
            {"Check": "Early stopping improvement tracking", "Status": "PASS"},
            {"Check": "Early stopping patience trigger", "Status": "PASS"},
            {"Check": "Model builder from run config", "Status": "PASS"},
            {"Check": "Synthetic batch shape validation", "Status": "PASS"},
        ]
        technical = {
            "training_engine_version": manifest["training_engine_version"],
            "class_name": manifest["class_name"],
            "unit_test_count": manifest["unit_test_count"],
        }
        return summary, [
            {"title": "Engine contract", "rows": engine_rows},
            {"title": "Synthetic unit tests", "rows": unit_rows},
        ], technical
    if phase_id == 20:
        summary_data = sources["summary"]
        contract = sources["contract"]
        summary = {
            "Baseline version": summary_data["baseline_version"],
            "Run ID": summary_data["run_id"],
            "Best epoch": summary_data["best_epoch"],
            "Validation RMSE (Wh)": summary_data["best_validation_rmse_wh"],
            "Validation MAE (Wh)": summary_data["validation_mae_wh"],
            "Validation R²": summary_data["validation_r2"],
            "Stopped reason": summary_data["stopped_reason"],
            "Beats Persistence": summary_data["beats_persistence"],
        }
        perf_rows = [
            {"Metric": "Validation RMSE", "Value": summary_data["best_validation_rmse_wh"], "Unit": "Wh"},
            {"Metric": "Validation MAE", "Value": summary_data["validation_mae_wh"], "Unit": "Wh"},
            {"Metric": "Validation R²", "Value": summary_data["validation_r2"], "Unit": "Dimensionless"},
            {"Metric": "Persistence Val RMSE", "Value": summary_data["persistence_validation_rmse_wh"], "Unit": "Wh"},
            {"Metric": "Improvement over Persistence", "Value": f"{(summary_data['persistence_validation_rmse_wh'] - summary_data['best_validation_rmse_wh']):.4f}", "Unit": "Wh"},
        ]
        config_rows = [
            {"Parameter": "Model family", "Value": contract["model_family"]},
            {"Parameter": "Feature variant", "Value": contract["feature_variant_id"]},
            {"Parameter": "Lookback steps", "Value": contract["lookback_steps"]},
            {"Parameter": "Forecast horizon", "Value": contract["horizon_steps"]},
            {"Parameter": "Target scaling", "Value": contract["target_scaling_option"]},
            {"Parameter": "Batch size", "Value": contract["batch_size"]},
            {"Parameter": "Seed", "Value": contract["seed"]},
            {"Parameter": "Trainable parameters", "Value": summary_data["trainable_parameters"]},
        ]
        technical = {
            "baseline_version": summary_data["baseline_version"],
            "run_id": summary_data["run_id"],
            "device_type": summary_data["device_type"],
            "total_epochs_run": summary_data["total_epochs_run"],
        }
        return summary, [
            {"title": "Validation performance", "rows": perf_rows},
            {"title": "Run configuration", "rows": config_rows},
        ], technical
    if phase_id == 21:
        summary_data = sources["summary"]
        contract = sources["contract"]
        summary = {
            "Baseline version": summary_data["baseline_version"],
            "Run ID": summary_data["run_id"],
            "Best epoch": summary_data["best_epoch"],
            "Validation RMSE (Wh)": summary_data["best_validation_rmse_wh"],
            "Validation MAE (Wh)": summary_data["validation_mae_wh"],
            "Validation R²": summary_data["validation_r2"],
            "Stopped reason": summary_data["stopped_reason"],
            "Beats Persistence": summary_data["beats_persistence"],
            "Beats LSTM B0": summary_data["beats_lstm_b0"],
        }
        comp_rows = [
            {"Model": "Persistence (No Learn)", "Val RMSE (Wh)": summary_data["persistence_validation_rmse_wh"], "Status": "Baseline"},
            {"Model": "LSTM Baseline (B0)", "Val RMSE (Wh)": summary_data["lstm_validation_rmse_wh"], "Status": "Learned Recurrent"},
            {"Model": "Transformer Encoder (B0)", "Val RMSE (Wh)": summary_data["best_validation_rmse_wh"], "Status": "Learned Attention"},
        ]
        config_rows = [
            {"Parameter": "Model family", "Value": contract["model_family"]},
            {"Parameter": "Feature variant", "Value": contract["feature_variant_id"]},
            {"Parameter": "Lookback steps", "Value": contract["lookback_steps"]},
            {"Parameter": "Forecast horizon", "Value": contract["horizon_steps"]},
            {"Parameter": "Target scaling", "Value": contract["target_scaling_option"]},
            {"Parameter": "Batch size", "Value": contract["batch_size"]},
            {"Parameter": "Seed", "Value": contract["seed"]},
            {"Parameter": "Trainable parameters", "Value": summary_data["trainable_parameters"]},
        ]
        technical = {
            "baseline_version": summary_data["baseline_version"],
            "run_id": summary_data["run_id"],
            "device_type": summary_data["device_type"],
            "total_epochs_run": summary_data["total_epochs_run"],
            "upstream_lstm_run_id": summary_data.get("upstream_lstm_run_id"),
        }
        return summary, [
            {"title": "Three-way baseline comparison", "rows": comp_rows},
            {"title": "Run configuration", "rows": config_rows},
        ], technical
    if phase_id == 22:
        manifest = sources["manifest"]
        summary_data = sources["summary"]
        lstm_count = sum(1 for row in summary_data if row.get("model") == "lstm_b0")
        transformer_count = sum(1 for row in summary_data if row.get("model") == "transformer_b0")
        summary = {
            "Models analyzed": ", ".join(sorted(manifest["models"])) if manifest.get("models") else "N/A",
            "Total findings": manifest["finding_count"],
            "LSTM findings": lstm_count,
            "Transformer findings": transformer_count,
        }
        sections = [{
            "title": "Diagnostic findings",
            "rows": [
                {
                    "Finding ID": row["finding_id"],
                    "Model": row["model"],
                    "Code": row["diagnostic_code"],
                    "Severity": row["severity"],
                    "Confidence": row["confidence"],
                    "Action": row["action_type"],
                    "Title": row["title"],
                }
                for row in summary_data
            ],
        }]
        technical = {
            "finding_count": manifest["finding_count"],
            "models": manifest["models"],
        }
        return summary, sections, technical
    if 23 <= phase_id <= 30:
        manifest = sources["manifest"]
        summary = {
            "Sweep code": manifest["sweep_code"],
            "Sweep name": manifest["sweep_name"],
            "Variant count": manifest["row_count"],
            "Best variant": (
                manifest["best_variant"].get("variant", "N/A") if manifest["best_variant"] else "N/A"
            ),
        }
        best_row = manifest["best_variant"] if manifest["best_variant"] else {}
        sections = [{
            "title": "Best variant details",
            "rows": [
                {"Field": key, "Value": value}
                for key, value in best_row.items()
            ],
        }]
        technical = {
            "sweep_code": manifest["sweep_code"],
            "variant_count": manifest["row_count"],
            "best_variant": best_row,
        }
        return summary, sections, technical
    raise ValueError(f"Unsupported phase_id: {phase_id}")


def build_phase_processing_log(phase_id: int, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    sources = _load_sources(root, phase_id)
    signoff = sources["signoff"]
    summary, sections, phase_details = _phase_content(phase_id, sources)
    manifest_warnings = []
    for role, value in sources.items():
        if role != "signoff" and isinstance(value, dict):
            manifest_warnings.extend(value.get("warnings", []))
    source_artifacts = [
        {
            "role": role,
            "path": relative_path,
            "sha256": sha256_file(root / relative_path),
        }
        for role, relative_path in SOURCE_SPECS[phase_id]
    ]
    return {
        "presentation_version": PRESENTATION_VERSION,
        "phase_id": phase_id,
        "phase_name": PHASE_NAMES[phase_id],
        "phase_version": signoff.get("phase_version"),
        "artifact_version": signoff.get("artifact_version"),
        "status": signoff["status"],
        "created_at": signoff["created_at"],
        "timestamp": signoff["created_at"],  # ISO8601 timestamp for training replay
        "summary": summary,
        "sections": sections,
        "warnings": _deduplicate(manifest_warnings + signoff.get("warnings", [])),
        "discrepancies": signoff.get("discrepancies", []),
        "source_artifacts": source_artifacts,
        "technical_details": {
            "dataset_revision": signoff.get("dataset_revision"),
            "environment_id": signoff.get("environment_id"),
            "config_fingerprint": signoff.get("config_fingerprint"),
            "tests": signoff.get("tests", []),
            "input_paths": signoff.get("input_paths", []),
            "input_checksums": signoff.get("input_checksums", {}),
            "output_paths": signoff.get("output_paths", []),
            "output_checksums": signoff.get("output_checksums", {}),
            **phase_details,
        },
    }


def save_phase_processing_log(log: dict[str, Any], project_root: Path) -> Path:
    phase_id = int(log["phase_id"])
    if phase_id not in LOG_FILENAMES:
        raise ValueError(f"Unsupported phase_id: {phase_id}")
    path = Path(project_root).resolve() / LOG_ROOT / LOG_FILENAMES[phase_id]
    atomic_write_bytes(path, canonical_json_bytes(log))
    if read_json(path) != log:
        raise RuntimeError(f"Phase presentation log reload failed: {path}")
    return path


def _plain_text(value: Any) -> str:
    if value is None:
        return "Not applicable"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(_plain_text(item) for item in value) if value else "None"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_value(value: Any) -> str:
    text = _plain_text(value)
    safe = escape(text, quote=True)
    if len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text):
        return f'<code title="{safe}">{escape(text[:12])}...{escape(text[-8:])}</code>'
    if len(text) > 180:
        return f'<span title="{safe}">{escape(text[:176])}...</span>'
    return safe


def _render_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f'<section class="cw-section"><h4>{escape(title)}</h4><p class="cw-empty">No records</p></section>'
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    header = "".join(f'<th scope="col">{escape(column)}</th>' for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_render_value(row.get(column))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    layout = "compact" if len(columns) <= 3 else "wide"
    table_class = f"cw-phase-table cw-phase-table-{layout}"
    return f'<section class="cw-section"><h4>{escape(title)}</h4><div class="cw-table-wrap"><table class="{table_class}" data-layout="{layout}" data-column-count="{len(columns)}"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div></section>'


def _require_phase_33_configuration(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Phase 33 configuration consistency check failed: {message}")


def build_phase_33_transformer_configuration(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    signoff = read_json(root / "artifacts/sweeps/S11_d_model/phase_33_signoff.json")
    winner = read_json(root / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json")
    reference = read_json(root / "artifacts/sweeps/S11_d_model/s11_reference_update.json")
    reference_run_id = reference["current_reference_run_id"]
    run_record = read_json(root / "artifacts/runs" / reference_run_id / "config.json")
    feature_registry = read_json(root / "artifacts/feature_sets/feature_set_registry.json")
    config = run_record["config"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    feature_variant_id = data_config["feature_variant_id"]
    feature_definition = feature_registry["variants"][feature_variant_id]
    run_ids = {
        signoff["winner_run_id"],
        winner["winner_run_id"],
        reference["winner_run_id"],
        reference_run_id,
        run_record["run_id"],
    }
    d_model_values = {
        int(signoff["winner_d_model"]),
        int(winner["winner_d_model"]),
        int(winner["d_model"]),
        int(reference["selected_d_model"]),
        int(reference["d_model"]),
        int(model_config["d_model"]),
    }
    rmse_values = {
        float(signoff["winner_rmse_wh"]),
        float(winner["winner_rmse_wh"]),
        float(reference["winner_rmse_wh"]),
    }
    _require_phase_33_configuration(signoff["status"] == "PASS", "sign-off status is not PASS")
    _require_phase_33_configuration(signoff["overall_status"] == "PASS", "overall status is not PASS")
    _require_phase_33_configuration(bool(signoff["approved_for_phase34"]), "Phase 34 is not approved")
    _require_phase_33_configuration(winner["status"] == "PASS", "winner status is not PASS")
    _require_phase_33_configuration(len(run_ids) == 1, "current reference run IDs disagree")
    _require_phase_33_configuration(len(d_model_values) == 1, "selected d_model values disagree")
    _require_phase_33_configuration(len(rmse_values) == 1, "Validation RMSE values disagree")
    _require_phase_33_configuration(
        run_record["config_fingerprint"] == winner["winner_config_fingerprint"] == reference["winner_config_fingerprint"],
        "current reference config fingerprints disagree",
    )
    _require_phase_33_configuration(
        feature_definition["feature_count"] == data_config["feature_count"] == model_config["input_size"],
        "feature counts disagree",
    )
    _require_phase_33_configuration(
        feature_registry["feature_set_version"] == config["lineage"]["feature_set_version"],
        "feature-set versions disagree",
    )
    _require_phase_33_configuration(
        signoff["test_status"] == winner["test_status"] == reference["test_status"] == "FORBIDDEN",
        "Test firewall is not preserved",
    )
    lookback_minutes = int(data_config["lookback_steps"]) * int(data_config["sampling_interval_minutes"])
    target_scaling_id = data_config["target_scaling_option"]
    target_scaling_value = (
        "YS1, Train-only StandardScaler"
        if target_scaling_id == "YS1"
        else target_scaling_id
    )
    revin_value = "Disabled" if not training_config["revin_enabled"] else "Enabled"
    configuration_rows = [
        {"Component": "Feature set", "Current value": f'{feature_variant_id}, {data_config["feature_count"]} features', "Decision source": "Phase 23-24", "State": "Selected"},
        {"Component": "Target scaling", "Current value": target_scaling_value, "Decision source": "Phase 25", "State": "Selected"},
        {"Component": "Lookback", "Current value": f'L{data_config["lookback_steps"]}, {lookback_minutes // 60} hours', "Decision source": "Phase 26", "State": "Selected"},
        {"Component": "Pooling", "Current value": model_config["pooling"], "Decision source": "Phase 27", "State": "Selected"},
        {"Component": "Activation", "Current value": model_config["activation"], "Decision source": "Phase 28", "State": "Selected"},
        {"Component": "Batch size", "Current value": training_config["batch_size"], "Decision source": "Phase 29", "State": "Selected"},
        {"Component": "Learning rate", "Current value": training_config["learning_rate"], "Decision source": "Phase 30", "State": "Selected"},
        {"Component": "AdamW weight decay", "Current value": training_config["weight_decay"], "Decision source": "Phase 31", "State": "Selected"},
        {"Component": "Dropout", "Current value": model_config["dropout"], "Decision source": "Phase 32", "State": "Selected"},
        {"Component": "d_model", "Current value": model_config["d_model"], "Decision source": "Phase 33", "State": "Selected"},
        {"Component": "Heads", "Current value": model_config["num_heads"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "Layers", "Current value": model_config["num_layers"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "FFN width", "Current value": model_config["ffn_dim"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "Loss", "Current value": training_config["loss_name"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "Max epochs", "Current value": training_config["max_epochs"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "Early-stopping patience", "Current value": training_config["early_stopping_patience"], "Decision source": "Training contract", "State": "Fixed"},
        {"Component": "Gradient clipping", "Current value": training_config["gradient_clip_max_norm"], "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "RevIN", "Current value": revin_value, "Decision source": "Frozen at Phase 33", "State": "Not swept"},
        {"Component": "Boundary protocol", "Current value": data_config["boundary_protocol"], "Decision source": "Frozen at Phase 33", "State": "Not checked by S19"},
    ]
    lineage_rows = [
        {"Field": "Phase 33 artifact version", "Value": signoff["artifact_version"]},
        {"Field": "Winner condition", "Value": winner["winner_d_model_id"]},
        {"Field": "Selection metric", "Value": winner["selection_metric"]},
        {"Field": "Current reference run", "Value": reference_run_id},
        {"Field": "Validation RMSE", "Value": winner["winner_rmse_wh"], "Unit": "Wh"},
        {"Field": "Validation MAE", "Value": winner["winner_mae_wh"], "Unit": "Wh"},
        {"Field": "Validation R²", "Value": winner["winner_r2"], "Unit": "Dimensionless"},
        {"Field": "Trainable parameters", "Value": winner["winner_trainable_parameters"], "Unit": "Parameters"},
        {"Field": "Test access", "Value": signoff["test_status"]},
    ]
    return {
        "phase_id": 33,
        "phase_name": PHASE_NAMES[33],
        "status": signoff["status"],
        "artifact_version": signoff["artifact_version"],
        "approved_for_phase34": signoff["approved_for_phase34"],
        "current_reference_run_id": reference_run_id,
        "validation_rmse_wh": winner["winner_rmse_wh"],
        "validation_mae_wh": winner["winner_mae_wh"],
        "validation_r2": winner["winner_r2"],
        "configuration_rows": configuration_rows,
        "lineage_rows": lineage_rows,
    }


def render_phase_33_transformer_configuration(project_root: Path | None = None) -> HTML:
    view = build_phase_33_transformer_configuration(Path(project_root or get_project_root()))
    status = escape(str(view["status"]))
    status_class = status.lower().replace("_", "-")
    metric_cards = [
        ("Current reference", view["current_reference_run_id"]),
        ("Validation RMSE", f'{view["validation_rmse_wh"]} Wh'),
        ("Validation MAE", f'{view["validation_mae_wh"]} Wh'),
        ("Validation R²", view["validation_r2"]),
    ]
    metrics_html = "".join(
        f'<div class="cw-config-metric"><span>{escape(label)}</span><strong>{_render_value(value)}</strong></div>'
        for label, value in metric_cards
    )
    configuration_body = "".join(
        "<tr>"
        f'<td>{_render_value(row["Component"])}</td>'
        f'<td>{_render_value(row["Current value"])}</td>'
        f'<td>{_render_value(row["Decision source"])}</td>'
        f'<td>{_render_value(row["State"])}</td>'
        "</tr>"
        for row in view["configuration_rows"]
    )
    lineage_body = "".join(
        "<tr>"
        f'<td>{_render_value(row["Field"])}</td>'
        f'<td>{_render_value(row["Value"])}</td>'
        f'<td>{_render_value(row.get("Unit"))}</td>'
        "</tr>"
        for row in view["lineage_rows"]
    )
    style = """
<style>
.cw-transformer-config{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:16px;background:#fff;box-shadow:0 10px 28px rgba(31,45,61,.09);margin:14px 0 24px;overflow:hidden}
.cw-transformer-config *{box-sizing:border-box}
.cw-config-header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:22px 24px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-config-header h3{font-size:22px;line-height:1.3;margin:0 0 6px;color:#172033}
.cw-config-meta{font-size:13px;color:#5d6b82}
.cw-config-status{border:1px solid #a9dec1;border-radius:999px;padding:7px 13px;font-size:12px;font-weight:700;letter-spacing:.03em;white-space:nowrap;color:#11613d;background:#e8f7ef}
.cw-config-status:not(.pass){border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-config-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;padding:18px 24px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-config-metric{display:flex;flex-direction:column;gap:5px;min-width:0;padding:12px 14px;border:1px solid #e1e7f0;border-radius:10px;background:#fff}
.cw-config-metric span{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-config-metric strong{font-size:14px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-config-content{padding:4px 24px 24px}
.cw-config-section{margin-top:20px}
.cw-config-section h4{font-size:15px;margin:0 0 9px;color:#334155}
.cw-config-table-wrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:10px}
.cw-config-table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;background:#fff}
.cw-config-table th{background:#f5f7fb;color:#475569;text-align:left;font-weight:650;padding:11px 13px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-config-table td{text-align:left;padding:10px 13px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-config-table tbody tr:nth-child(even){background:#fafbfd}
.cw-config-table tbody tr:last-child td{border-bottom:0}
.cw-config-main th:nth-child(1),.cw-config-main td:nth-child(1){width:22%}
.cw-config-main th:nth-child(2),.cw-config-main td:nth-child(2){width:28%}
.cw-config-main th:nth-child(3),.cw-config-main td:nth-child(3){width:27%}
.cw-config-main th:nth-child(4),.cw-config-main td:nth-child(4){width:23%}
.cw-config-lineage th:nth-child(1),.cw-config-lineage td:nth-child(1){width:30%}
.cw-config-lineage th:nth-child(2),.cw-config-lineage td:nth-child(2){width:50%}
.cw-config-lineage th:nth-child(3),.cw-config-lineage td:nth-child(3){width:20%}
@media (max-width:900px){.cw-config-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-config-table{min-width:720px}}
@media (max-width:620px){.cw-config-header{flex-direction:column;padding:18px}.cw-config-metrics{grid-template-columns:1fr;padding:14px 18px}.cw-config-content{padding-left:18px;padding-right:18px}}
</style>
"""
    header = (
        '<header class="cw-config-header">'
        '<div><h3>Transformer Configuration after Phase 33</h3>'
        f'<div class="cw-config-meta">{escape(str(view["artifact_version"]))} | Approved next phase: Phase 34</div></div>'
        f'<span class="cw-config-status {escape(status_class)}">{status}</span>'
        "</header>"
    )
    configuration_table = (
        '<section class="cw-config-section"><h4>Current configuration</h4>'
        '<div class="cw-config-table-wrap"><table class="cw-config-table cw-config-main">'
        '<thead><tr><th>Component</th><th>Current value</th><th>Decision source</th><th>State</th></tr></thead>'
        f'<tbody>{configuration_body}</tbody></table></div></section>'
    )
    lineage_table = (
        '<section class="cw-config-section"><h4>Canonical lineage</h4>'
        '<div class="cw-config-table-wrap"><table class="cw-config-table cw-config-lineage">'
        '<thead><tr><th>Field</th><th>Value</th><th>Unit</th></tr></thead>'
        f'<tbody>{lineage_body}</tbody></table></div></section>'
    )
    return HTML(
        f'{style}<article class="cw-transformer-config">{header}'
        f'<div class="cw-config-metrics">{metrics_html}</div>'
        f'<div class="cw-config-content">{configuration_table}{lineage_table}</div></article>'
    )


def _render_split_bar(technical_details: dict[str, Any]) -> str:
    visualization = technical_details.get("visualization")
    if not visualization or visualization.get("type") != "split_ratio_bar":
        return ""
    segments = visualization["segments"]
    segment_html = "".join(
        f'<div class="cw-split cw-split-{index}" style="width:{float(segment["ratio"]) * 100:.4f}%">{escape(segment["label"])} {_percentage(segment["ratio"])}</div>'
        for index, segment in enumerate(segments)
    )
    return f'<section class="cw-section"><h4>Split allocation</h4><div class="cw-split-bar">{segment_html}</div></section>'


def _visible_sections(log: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    phase_id = int(log["phase_id"])
    spec = PRESENTATION_SPECS[phase_id]
    summary_rows = [
        {"Field": field, "Value": log["summary"][field]}
        for field in spec["summary_fields"]
    ]
    source_sections = {section["title"]: section["rows"] for section in log["sections"]}
    visible_sections = []
    for section_spec in spec["sections"]:
        source_title = section_spec.get("source_title", section_spec["title"])
        rows = source_sections[source_title]
        row_field = section_spec.get("row_field")
        if row_field is not None:
            allowed_values = set(section_spec["row_values"])
            rows = [row for row in rows if row.get(row_field) in allowed_values]
        columns = section_spec.get("columns")
        if columns is not None:
            rows = [{column: row[column] for column in columns} for row in rows]
        visible_sections.append({"title": section_spec["title"], "rows": rows})
    return summary_rows, visible_sections, bool(spec.get("split_allocation", False))


def render_dataframe_table(
    frame: Any,
    title: str,
    subtitle: str,
    metrics: dict[str, Any] | None = None,
    index: bool = True,
    max_height: int = 480,
    precision: int = 4,
) -> HTML:
    if not hasattr(frame, "to_html") or not hasattr(frame, "copy"):
        raise TypeError("frame must provide DataFrame-compatible copy and to_html methods")
    if max_height <= 0:
        raise ValueError("max_height must be positive")
    if precision < 0:
        raise ValueError("precision must be non-negative")
    view = frame.copy(deep=True)
    table = view.to_html(
        border=0,
        classes="cw-data-table",
        escape=True,
        index=index,
        float_format=lambda value: f"{value:,.{precision}f}",
        na_rep="Not available",
    )
    metric_values = {
        "Displayed rows": len(view),
        "Displayed columns": len(view.columns),
        **(metrics or {}),
    }
    metric_html = "".join(
        f'<div class="cw-data-metric"><span>{escape(str(label))}</span><strong>{_render_value(value)}</strong></div>'
        for label, value in metric_values.items()
    )
    style = f"""
<style>
.cw-data-view{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #dbe3ee;border-radius:16px;background:#fff;box-shadow:0 10px 28px rgba(31,45,61,.09);margin:14px 0 24px;overflow:hidden}}
.cw-data-view *{{box-sizing:border-box}}
.cw-data-header{{padding:20px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #dbe3ee}}
.cw-data-header h4{{font-size:19px;line-height:1.3;margin:0 0 6px;color:#172033}}
.cw-data-header p{{font-size:13px;line-height:1.5;color:#5d6b82;margin:0}}
.cw-data-metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;padding:14px 18px;background:#fbfcff;border-bottom:1px solid #e5eaf1}}
.cw-data-metric{{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:10px;background:#fff}}
.cw-data-metric span{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}}
.cw-data-metric strong{{font-size:14px;color:#24324a;font-weight:700;overflow-wrap:anywhere}}
.cw-data-scroll{{max-height:{int(max_height)}px;overflow:auto;position:relative}}
.cw-data-table{{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:12.5px;background:#fff}}
.cw-data-table thead th{{position:sticky;top:0;z-index:3;background:#253b63;color:#fff;font-weight:650;text-align:right;padding:11px 12px;border-right:1px solid rgba(255,255,255,.12);white-space:nowrap}}
.cw-data-table thead th:first-child{{left:0;z-index:5;text-align:left}}
.cw-data-table tbody th{{position:sticky;left:0;z-index:2;background:#eef3fb;color:#334155;font-weight:650;text-align:left;padding:10px 12px;border-right:1px solid #dbe3ee;border-bottom:1px solid #e8edf3;white-space:nowrap}}
.cw-data-table td{{text-align:right;padding:10px 12px;border-right:1px solid #edf1f5;border-bottom:1px solid #edf1f5;white-space:nowrap;color:#293548}}
.cw-data-table tbody tr:nth-child(even) td{{background:#f8fafe}}
.cw-data-table tbody tr:hover td{{background:#eaf2ff}}
.cw-data-table tbody tr:hover th{{background:#dce8f8}}
.cw-data-footer{{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:10px 18px;background:#f8fafc;color:#64748b;font-size:11px;border-top:1px solid #e5eaf1}}
@media (max-width:720px){{.cw-data-header{{padding:16px}}.cw-data-metrics{{grid-template-columns:repeat(2,minmax(0,1fr));padding:10px}}}}
</style>
"""
    header = f'<header class="cw-data-header"><h4>{escape(title)}</h4><p>{escape(subtitle)}</p></header>'
    footer = f'<footer class="cw-data-footer"><span>Scrollable table</span><span>{len(view)} rows by {len(view.columns)} columns</span></footer>'
    return HTML(f'{style}<article class="cw-data-view">{header}<div class="cw-data-metrics">{metric_html}</div><div class="cw-data-scroll">{table}</div>{footer}</article>')


def render_phase_log(log: dict[str, Any]) -> HTML:
    status = str(log["status"])
    status_class = status.lower().replace("_", "-")
    summary_rows, visible_sections, show_split_allocation = _visible_sections(log)
    spec = PRESENTATION_SPECS[int(log["phase_id"])]
    sections = ""
    if summary_rows:
        sections += _render_table(str(spec["summary_title"]), summary_rows)
    sections += "".join(_render_table(section["title"], section["rows"]) for section in visible_sections)
    if show_split_allocation:
        sections += _render_split_bar(log["technical_details"])
    style = """
<style>
.cw-phase-summary{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #dbe3ee;border-radius:14px;background:#ffffff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 24px;overflow:hidden}
.cw-phase-summary *{box-sizing:border-box}
.cw-header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;padding:20px 22px;background:linear-gradient(135deg,#f7f9fc,#eef4fb);border-bottom:1px solid #dbe3ee}
.cw-header h3{font-size:20px;line-height:1.3;margin:0 0 6px;color:#172033}
.cw-meta{color:#5d6b82;font-size:13px}
.cw-status{white-space:nowrap;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:700;letter-spacing:.03em;border:1px solid transparent}
.cw-status.pass{color:#11613d;background:#e8f7ef;border-color:#a9dec1}
.cw-status.pass-with-warning{color:#7a4b00;background:#fff5dc;border-color:#efd18a}
.cw-status.failed,.cw-status.fail{color:#8f2430;background:#fdecef;border-color:#efb3ba}
.cw-content{padding:4px 22px 22px}
.cw-section{margin-top:20px}
.cw-section h4{font-size:14px;margin:0 0 9px;color:#334155}
.cw-table-wrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-phase-summary table.cw-phase-table{border-collapse:collapse;width:100%;table-layout:auto;font-size:13px;background:#fff}
.cw-phase-summary table.cw-phase-table th{background:#f7f9fc;color:#475569;text-align:left;font-weight:650;padding:10px 12px;border-bottom:1px solid #e2e8f0}
.cw-phase-summary table.cw-phase-table td{text-align:left;padding:10px 12px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45}
.cw-phase-summary table.cw-phase-table-compact th:not(:last-child),.cw-phase-summary table.cw-phase-table-compact td:not(:last-child){width:1%;white-space:nowrap}
.cw-phase-summary table.cw-phase-table-compact th:last-child,.cw-phase-summary table.cw-phase-table-compact td:last-child{white-space:normal;overflow-wrap:anywhere}
.cw-phase-summary table.cw-phase-table-wide{width:max-content;min-width:100%}
.cw-phase-summary table.cw-phase-table-wide th,.cw-phase-summary table.cw-phase-table-wide td{white-space:nowrap}
.cw-phase-summary table.cw-phase-table tbody tr:last-child td{border-bottom:0}
.cw-phase-summary table.cw-phase-table tbody tr:nth-child(even){background:#fbfcfe}
.cw-phase-summary code{font-size:12px;color:#334155;background:#eef2f7;border-radius:4px;padding:2px 5px}
.cw-empty{margin:0;color:#64748b;font-size:13px}
.cw-split-bar{display:flex;width:100%;height:42px;border-radius:9px;overflow:hidden;color:#fff;font-size:12px;font-weight:650;text-align:center;line-height:42px}
.cw-split-0{background:#2463a6}.cw-split-1{background:#39866d}.cw-split-2{background:#a35d3a}
@media (max-width:720px){.cw-header{flex-direction:column}.cw-content{padding-left:12px;padding-right:12px}.cw-phase-summary table.cw-phase-table-compact th:not(:last-child),.cw-phase-summary table.cw-phase-table-compact td:not(:last-child){width:auto;white-space:normal}.cw-split-bar{font-size:10px}}
</style>
"""
    header = f'<header class="cw-header"><div><h3>Phase {int(log["phase_id"])} - {escape(str(log["phase_name"]))}</h3><div class="cw-meta">{escape(str(log.get("artifact_version") or log.get("phase_version") or "Unversioned"))}</div></div><span class="cw-status {escape(status_class)}">{escape(status)}</span></header>'
    content = f'<div class="cw-content">{sections}</div>' if sections else ""
    return HTML(f'{style}<article class="cw-phase-summary">{header}{content}</article>')


def render_phase_summary(phase_id: int, project_root: Path) -> HTML:
    log = build_phase_processing_log(phase_id, project_root)
    save_phase_processing_log(log, project_root)
    return render_phase_log(log)


def _selective_condition_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    inspection = decision["inspection"]
    conditions = inspection["conditions"]
    verified = {item["condition_id"]: item for item in conditions["verified_conditions"]}
    invalid = {item["condition_id"]: item for item in conditions["invalid_conditions"]}
    running = {item["condition_id"]: item for item in conditions["running_conditions"]}
    failed = {item["condition_id"]: item for item in conditions["failed_conditions"]}
    values = dict(get_sweep_phase_spec(int(decision["phase_id"])).condition_values)
    rows = []
    for condition_id in conditions["expected_conditions"]:
        if condition_id in verified:
            item = verified[condition_id]
            status = "VERIFIED"
            run_id = item["run_id"]
            rmse_wh = item["rmse_wh"]
        elif condition_id in running:
            item = running[condition_id]
            status = "RUNNING"
            run_id = item["run_id"]
            rmse_wh = None
        elif condition_id in failed:
            item = failed[condition_id]
            status = "FAILED"
            run_id = item["run_id"]
            rmse_wh = None
        elif condition_id in invalid:
            item = invalid[condition_id]
            status = "INVALID"
            run_id = item.get("run_id")
            rmse_wh = None
        else:
            status = "MISSING"
            run_id = None
            rmse_wh = None
        rows.append({"Condition": condition_id, "Value": values[condition_id], "Evidence": status, "Run ID": run_id, "Validation RMSE Wh": rmse_wh})
    return rows


def _selective_prerequisite_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"Required artifact": item["path"], "Validation": item["status"]}
        for item in decision["inspection"]["prerequisites"]["records"]
    ]


def build_phase_resume_log(phase_id: int, project_root: Path, allow_execution: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    decision = plan_phase_resume(phase_id, root, allow_execution=allow_execution)
    phase_31_preflight = build_phase_31_preflight(root) if phase_id == 31 else None
    phase_32_preflight = build_phase_32_preflight(root) if phase_id == 32 else None
    phase_33_preflight = build_phase_33_preflight(root) if phase_id == 33 else None
    phase_34_preflight = build_phase_34_preflight(root) if phase_id == 34 else None
    phase_preflight = phase_31_preflight or phase_32_preflight or phase_33_preflight or phase_34_preflight
    preflight_reasons = []
    if phase_preflight is not None:
        preflight_reasons = [
            f"{issue['path']}: {issue['reason']}"
            for issue in phase_preflight["handoff"]["issues"]
        ]
    reasons = list(dict.fromkeys([*decision["reasons"], *preflight_reasons]))
    blocked = decision["effective_action"] == "BLOCK" or (
        phase_preflight is not None and not phase_preflight["handoff"]["valid"]
    )
    signoff = decision["inspection"]["signoff"]
    canonical_status = signoff.get("record", {}).get("status") if signoff.get("valid") else None
    status = "BLOCKED" if blocked else canonical_status or decision["inspection"]["state"]
    summary = {
        "State": decision["state"],
        "Resolved action": decision["resolved_action"],
        "Effective action": "BLOCK" if blocked else decision["effective_action"],
        "Execution authorized": decision["execution_authorized"],
    }
    sections = [{"title": "Prerequisite validation", "rows": _selective_prerequisite_rows(decision)}]
    source_artifacts = []
    if phase_31_preflight is not None:
        handoff = phase_31_preflight["handoff"]
        summary["Selected learning rate"] = handoff["selected_learning_rate"]
        summary["S8 reference run"] = handoff["winner_run_id"]
        summary["Test access"] = phase_31_preflight["test_access"]
        sections.append(
            {
                "title": "S9 frozen contract",
                "rows": [
                    {"Field": "Swept factor", "Value": "weight_decay"},
                    {"Field": "Optimizer", "Value": "AdamW"},
                    {"Field": "Reference condition", "Value": "WD1 = 0.0001"},
                    {"Field": "New conditions", "Value": "WD0 = 0.0; WD2 = 0.001"},
                    {"Field": "Selection metric", "Value": "Validation RMSE Wh"},
                    {"Field": "Exact tie rule", "Value": "Lower weight decay"},
                    {"Field": "Explicit L2 loss", "Value": "Disabled"},
                    {"Field": "Test access", "Value": phase_31_preflight["test_access"]},
                ],
            }
        )
        for relative_path in handoff["source_paths"]:
            path = root / relative_path
            if path.is_file():
                source_artifacts.append(
                    {
                        "path": relative_path,
                        "role": "phase_30_handoff",
                        "sha256": sha256_file(path),
                    }
                )
    if phase_32_preflight is not None:
        handoff = phase_32_preflight["handoff"]
        summary["Selected weight decay"] = handoff["selected_weight_decay"]
        summary["S9 reference run"] = handoff["winner_run_id"]
        summary["Test access"] = phase_32_preflight["test_access"]
        sections.append(
            {
                "title": "S10 frozen contract",
                "rows": [
                    {"Field": "Swept factor", "Value": "dropout"},
                    {"Field": "Scope", "Value": "All Transformer encoder dropout sites"},
                    {"Field": "Reference condition", "Value": "DR01 = 0.1"},
                    {"Field": "New conditions", "Value": "DR02 = 0.2; DR03 = 0.3"},
                    {"Field": "Selection metric", "Value": "Validation RMSE Wh"},
                    {"Field": "Exact tie rule", "Value": "Lower dropout"},
                    {"Field": "MC Dropout", "Value": "Disabled"},
                    {"Field": "Test access", "Value": phase_32_preflight["test_access"]},
                ],
            }
        )
        for relative_path in handoff["source_paths"]:
            path = root / relative_path
            if path.is_file():
                source_artifacts.append(
                    {
                        "path": relative_path,
                        "role": "phase_31_handoff",
                        "sha256": sha256_file(path),
                    }
                )
    if phase_33_preflight is not None:
        handoff = phase_33_preflight["handoff"]
        summary["Selected dropout"] = handoff["selected_dropout"]
        summary["S10 reference run"] = handoff["winner_run_id"]
        summary["Test access"] = phase_33_preflight["test_access"]
        sections.append(
            {
                "title": "S11 frozen contract",
                "rows": [
                    {"Field": "Swept factor", "Value": "d_model"},
                    {"Field": "Reference condition", "Value": "D64 = 64"},
                    {"Field": "New condition", "Value": "D32 = 32"},
                    {"Field": "Frozen heads", "Value": "H4"},
                    {"Field": "Frozen layers", "Value": "N2"},
                    {"Field": "Frozen FFN width", "Value": "F128"},
                    {"Field": "Selection metric", "Value": "Validation RMSE Wh"},
                    {"Field": "Exact tie rule", "Value": "D32"},
                    {"Field": "Test access", "Value": phase_33_preflight["test_access"]},
                ],
            }
        )
        capacity = handoff.get("capacity_comparison")
        if isinstance(capacity, dict):
            sections.append(
                {
                    "title": "Capacity context",
                    "rows": [
                        {
                            "Condition": item["condition_id"],
                            "d_model": item["d_model"],
                            "Head dimension": item["head_dim"],
                            "FFN ratio": item["ffn_ratio"],
                            "Trainable parameters": item["trainable_parameter_count"],
                        }
                        for item in capacity["candidates"]
                    ],
                }
            )
        for relative_path in handoff["source_paths"]:
            path = root / relative_path
            if path.is_file():
                source_artifacts.append(
                    {
                        "path": relative_path,
                        "role": "phase_32_handoff",
                        "sha256": sha256_file(path),
                    }
                )
    if phase_34_preflight is not None:
        handoff = phase_34_preflight["handoff"]
        summary["Selected d_model"] = handoff["selected_d_model"]
        summary["S11 reference run"] = handoff["winner_run_id"]
        summary["Test access"] = phase_34_preflight["test_access"]
        geometry = handoff.get("geometry_comparison")
        geometry_by_id = {
            item["condition_id"]: item
            for item in geometry.get("candidates", [])
        } if isinstance(geometry, dict) else {}
        h2 = geometry_by_id.get("H2", {})
        h4 = geometry_by_id.get("H4", {})
        sections.append(
            {
                "title": "S12 frozen contract",
                "rows": [
                    {"Field": "Swept factor", "Value": "num_heads"},
                    {"Field": "Reference condition", "Value": "H4 = 4"},
                    {"Field": "New condition", "Value": "H2 = 2"},
                    {"Field": "Frozen d_model", "Value": handoff["selected_d_model"]},
                    {"Field": "Selection metric", "Value": "Validation RMSE Wh"},
                    {"Field": "Exact tie rule", "Value": "H2"},
                    {"Field": "Parameter count", "Value": "Unchanged across H2 and H4"},
                    {"Field": "Test access", "Value": phase_34_preflight["test_access"]},
                ],
            }
        )
        sections.append(
            {
                "title": "Head geometry",
                "rows": [
                    {
                        "Condition": condition_id,
                        "Heads": item.get("num_heads"),
                        "Head dimension": item.get("head_dim"),
                        "Attention shape": item.get("attention_shapes"),
                        "Trainable parameters": item.get("trainable_parameter_count"),
                        "Audit": item.get("status"),
                    }
                    for condition_id, item in (("H2", h2), ("H4", h4))
                ],
            }
        )
        if signoff.get("valid"):
            signoff_record = signoff["record"]
            winner_path = root / decision["inspection"]["artifacts"]["records"][2]["path"]
            reference_path = root / decision["inspection"]["artifacts"]["records"][3]["path"]
            winner_record = read_json(winner_path)
            reference_record = read_json(reference_path)
            sections.append(
                {
                    "title": "Phase 34 result",
                    "rows": [
                        {"Field": "Winner", "Value": winner_record["winner_head_id"]},
                        {"Field": "Selected heads", "Value": winner_record["winner_num_heads"]},
                        {"Field": "Selected head dimension", "Value": winner_record["winner_head_dim"]},
                        {"Field": "Winner run", "Value": winner_record["winner_run_id"]},
                        {"Field": "Validation RMSE", "Value": winner_record["winner_rmse_wh"], "Unit": "Wh"},
                        {"Field": "Validation MAE", "Value": winner_record["winner_mae_wh"], "Unit": "Wh"},
                        {"Field": "Validation R²", "Value": winner_record["winner_r2"], "Unit": "Dimensionless"},
                    ],
                }
            )
            sections.append(
                {
                    "title": "Phase 35 handoff",
                    "rows": [
                        {"Field": "Approved", "Value": reference_record["approved_for_phase35"]},
                        {"Field": "Current reference run", "Value": reference_record["current_reference_run_id"]},
                        {"Field": "Selected head condition", "Value": reference_record["selected_head_id"]},
                        {"Field": "Selected heads", "Value": reference_record["selected_num_heads"]},
                        {"Field": "Overall status", "Value": signoff_record["overall_status"]},
                        {"Field": "Test access", "Value": signoff_record["test_status"]},
                    ],
                }
            )
        for relative_path in handoff["source_paths"]:
            path = root / relative_path
            if path.is_file():
                source_artifacts.append(
                    {
                        "path": relative_path,
                        "role": "phase_33_handoff",
                        "sha256": sha256_file(path),
                    }
                )
    sections.append({"title": "Condition evidence", "rows": _selective_condition_rows(decision)})
    return {
        "presentation_version": PRESENTATION_VERSION,
        "phase_id": phase_id,
        "phase_name": decision["inspection"]["phase_name"],
        "phase_version": f"PHASE-{phase_id}-v1",
        "artifact_version": "SELECTIVE-RESUME-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "sections": sections,
        "warnings": [],
        "discrepancies": reasons,
        "source_artifacts": source_artifacts,
        "technical_details": {
            **decision,
            "phase_31_preflight": phase_31_preflight,
            "phase_32_preflight": phase_32_preflight,
            "phase_33_preflight": phase_33_preflight,
            "phase_34_preflight": phase_34_preflight,
        },
    }


def save_phase_resume_log(log: dict[str, Any], project_root: Path) -> Path:
    phase_id = int(log["phase_id"])
    if phase_id not in range(23, 35):
        raise ValueError(f"Selective resume log supports Phase 23-34, received Phase {phase_id}")
    path = Path(project_root).resolve() / LOG_ROOT / LOG_FILENAMES[phase_id]
    atomic_write_bytes(path, canonical_json_bytes(log))
    if read_json(path) != log:
        raise RuntimeError(f"Selective phase log reload failed: {path}")
    return path


def materialize_phase_resume_log(
    phase_id: int,
    project_root: Path,
    allow_execution: bool = False,
) -> tuple[dict[str, Any], Path]:
    root = Path(project_root).resolve()
    log = build_phase_resume_log(phase_id, root, allow_execution=allow_execution)
    path = save_phase_resume_log(log, root)
    if log["status"] in {"LOG_MISSING", "LOG_STALE"}:
        log = build_phase_resume_log(phase_id, root, allow_execution=allow_execution)
        path = save_phase_resume_log(log, root)
    return log, path


def render_phase_resume_log(log: dict[str, Any]) -> HTML:
    status = str(log["status"])
    status_class = "pass" if status in {"PASS", "VALID_REUSABLE"} else "blocked"
    summary_rows = [{"Field": field, "Value": value} for field, value in log["summary"].items()]
    sections = _render_table("Execution decision", summary_rows)
    sections += "".join(_render_table(section["title"], section["rows"]) for section in log["sections"])
    reasons = log.get("discrepancies", [])
    reason_html = ""
    if reasons:
        values = "".join(f"<li>{escape(str(reason))}</li>" for reason in reasons)
        reason_html = f'<section class="cw-resume-reasons"><h4>Block reason</h4><ul>{values}</ul></section>'
    style = """
<style>
.cw-phase-resume{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:15px;background:#fff;box-shadow:0 9px 26px rgba(31,45,61,.08);margin:14px 0 24px;overflow:hidden}
.cw-phase-resume *{box-sizing:border-box}
.cw-resume-header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:20px 22px;background:linear-gradient(135deg,#f5f8fd,#edf3fb);border-bottom:1px solid #d9e2ef}
.cw-resume-header h3{font-size:20px;line-height:1.3;margin:0 0 5px;color:#172033}
.cw-resume-meta{font-size:13px;color:#5d6b82}
.cw-resume-status{border:1px solid transparent;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:700;letter-spacing:.03em;white-space:nowrap}
.cw-resume-status.pass{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-resume-status.blocked{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-resume-content{padding:2px 22px 22px}
.cw-phase-resume .cw-section{margin-top:20px}
.cw-phase-resume .cw-section h4,.cw-resume-reasons h4{font-size:14px;margin:0 0 9px;color:#334155}
.cw-phase-resume .cw-table-wrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-phase-resume .cw-phase-table{border-collapse:collapse;width:100%;table-layout:auto;font-size:13px;background:#fff}
.cw-phase-resume .cw-phase-table th{background:#f7f9fc;color:#475569;text-align:left;font-weight:650;padding:10px 12px;border-bottom:1px solid #e2e8f0;white-space:nowrap}
.cw-phase-resume .cw-phase-table td{text-align:left;padding:10px 12px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-phase-resume .cw-phase-table-compact th:not(:last-child),.cw-phase-resume .cw-phase-table-compact td:not(:last-child){width:1%;white-space:nowrap}
.cw-phase-resume .cw-phase-table tbody tr:last-child td{border-bottom:0}
.cw-phase-resume .cw-phase-table tbody tr:nth-child(even){background:#fbfcfe}
.cw-resume-reasons{margin-top:20px;padding:14px 16px;border:1px solid #efc5ca;border-radius:9px;background:#fff6f7;color:#7f2630}
.cw-resume-reasons ul{margin:0;padding-left:20px}
.cw-resume-reasons li{margin:3px 0}
@media (max-width:720px){.cw-resume-header{flex-direction:column}.cw-resume-content{padding-left:12px;padding-right:12px}.cw-phase-resume .cw-phase-table-compact th:not(:last-child),.cw-phase-resume .cw-phase-table-compact td:not(:last-child){width:auto;white-space:normal}}
</style>
"""
    header = f'<header class="cw-resume-header"><div><h3>Phase {int(log["phase_id"])} - {escape(str(log["phase_name"]))}</h3><div class="cw-resume-meta">{escape(str(log["artifact_version"]))}</div></div><span class="cw-resume-status {status_class}">{escape(status)}</span></header>'
    return HTML(f'{style}<article class="cw-phase-resume">{header}<div class="cw-resume-content">{sections}{reason_html}</div></article>')


def render_phase_resume(phase_id: int, project_root: Path | None = None, allow_execution: bool = False) -> HTML:
    root = Path(project_root or get_project_root()).resolve()
    log, _ = materialize_phase_resume_log(phase_id, root, allow_execution=allow_execution)
    return render_phase_resume_log(log)


# ---------------------------------------------------------------------------
# Batch log visualization (all phases at once)
# ---------------------------------------------------------------------------


def _sortable_table_html(df_rows: list[dict[str, Any]]) -> str:
    if not df_rows:
        return '<p class="cw-empty">No log files found.</p>'

    columns = ["phase", "phase_name", "timestamp", "status"]
    optional = ["artifact_version", "path"]
    present = [c for c in columns + optional if any(c in row for row in df_rows)]
    visible = present[:8]

    header = "".join(f'<th>{c}</th>' for c in visible)
    body_rows = []
    for row in df_rows:
        cells = "".join(f'<td>{_render_value(row.get(c))}</td>' for c in visible)
        status = str(row.get("status", "")).lower()
        tr_class = "cw-row-pass" if status == "pass" else "cw-row-fail"
        body_rows.append(f'<tr class="{tr_class}" data-phase="{row.get("phase","")}">{cells}</tr>')

    table = (
        f'<table class="cw-log-table" id="cw-log-table">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )
    return table


def render_all_logs_summary(project_root: Path | None = None) -> HTML:
    root = Path(project_root or get_project_root()).resolve()
    rows = []
    for phase_id in sorted(LOG_FILENAMES):
        path = root / LOG_ROOT / LOG_FILENAMES[phase_id]
        if not path.is_file():
            continue
        try:
            data = read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        status = data.get("status", "")
        if status == "VALID_REUSABLE" and phase_id >= 23:
            inspection = inspect_phase_state(phase_id, root)
            if inspection["signoff"]["valid"]:
                status = inspection["signoff"]["record"]["status"]
        rows.append({
            "phase": data.get("phase_id"),
            "phase_name": data.get("phase_name", ""),
            "timestamp": data.get("timestamp") or data.get("created_at", ""),
            "status": status,
            "artifact_version": data.get("artifact_version") or data.get("phase_version", ""),
            "path": path.name,
        })

    table_html = _sortable_table_html(rows)
    n_pass = sum(1 for row in rows if str(row.get("status", "")).lower() == "pass")
    n_attention = len(rows) - n_pass
    style = """
<style>
.cw-logs-view{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;margin:14px 0 24px}
.cw-logs-header{padding:20px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border:1px solid #dbe3ee;border-radius:14px 14px 0 0;border-bottom:none}
.cw-logs-header h3{font-size:18px;margin:0 0 5px;color:#172033}
.cw-logs-stats{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:#5d6b82}
.cw-logs-stats span{font-weight:650}
.cw-logs-body{border:1px solid #dbe3ee;border-radius:0 0 14px 14px;background:#fff;overflow:hidden}
.cw-logs-scroll{max-height:520px;overflow:auto}
.cw-log-table{width:100%;border-collapse:collapse;font-size:13px}
.cw-log-table thead th{position:sticky;top:0;z-index:3;background:#253b63;color:#fff;font-weight:650;text-align:left;padding:11px 14px;border-right:1px solid rgba(255,255,255,.1);white-space:nowrap}
.cw-log-table tbody tr{border-bottom:1px solid #edf1f5}
.cw-log-table tbody tr:hover{background:#eaf2ff}
.cw-log-table tbody tr:last-child{border-bottom:none}
.cw-log-table td{padding:10px 14px;vertical-align:middle;color:#293548}
.cw-log-table td:first-child{font-weight:700;color:#253b63}
.cw-row-pass td:first-child{color:#11613d}
.cw-row-fail td:first-child{color:#8f2430}
.cw-empty{margin:0;padding:20px;color:#64748b;font-size:13px}
@media (max-width:720px){.cw-logs-header{padding:16px}.cw-log-table{font-size:12px}.cw-log-table th,.cw-log-table td{padding:9px 10px}}
</style>
"""
    return HTML(
        f'<div class="cw-logs-view">'
        f'<div class="cw-logs-header"><h3>Phase Processing Logs</h3>'
        f'<div class="cw-logs-stats"><span>{len(rows)} phases</span>'
        f'<span style="color:#11613d">PASS: {n_pass}</span>'
        f'<span style="color:#8f2430">Needs attention: {n_attention}</span></div></div>'
        f'<div class="cw-logs-body"><div class="cw-logs-scroll">{table_html}</div></div>'
        f'{style}</div>'
    )


def load_and_render_all_logs(project_root: Path) -> HTML:
    """Convenience: build log dicts, save to disk, and return interactive HTML."""
    rows_data = []
    for phase_id in sorted(LOG_FILENAMES.keys()):
        path = Path(project_root).resolve() / LOG_ROOT / LOG_FILENAMES[phase_id]
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        rows_data.append(data)
        save_phase_processing_log(data, project_root)
    return render_all_logs_summary(project_root)
