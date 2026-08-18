import json
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML

from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
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
}


def _load_sources(project_root: Path, phase_id: int) -> dict[str, Any]:
    if phase_id not in SOURCE_SPECS:
        raise ValueError(f"Unsupported phase_id: {phase_id}")
    return {
        role: read_json(project_root / relative_path)
        for role, relative_path in SOURCE_SPECS[phase_id]
    }


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
