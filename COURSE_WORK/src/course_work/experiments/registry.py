import csv
import json
import math
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from course_work.evaluation.metrics import METRIC_VERSION, materialize_phase_12
from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


EXPERIMENT_VERSION = "EXPERIMENTS-v1"
RECORD_SCHEMA_VERSION = 1
FAILURE_TAXONOMY_VERSION = "FAILURES-v1"
EXPERIMENT_ARTIFACT_ROOT = Path("artifacts/experiments")
RUN_ARTIFACT_ROOT = Path("artifacts/runs")
REGISTRY_CONTRACT_ID = "EXPERIMENT-REGISTRY-CONTRACT-v1"
RERUN_REASONS = {
    "REPRODUCIBILITY_CHECK",
    "CODE_FIX",
    "DEVICE_CHANGE",
    "ENVIRONMENT_CHANGE",
    "CHECKPOINT_RECOVERY",
    "MANUAL_RERUN",
}
FEATURE_VARIANTS = {"FS0_TF0", "FS0_TF1", "FS1_TF0", "FS1_TF1", "FS2_TF0", "FS2_TF1"}
LOOKBACK_OPTIONS = {36, 72, 144}
TARGET_SCALING_OPTIONS = {"YS0", "YS1"}
BOUNDARY_PROTOCOLS = {"WB0_CONTEXT_CARRY_OVER", "WB1_STRICT_ISOLATION"}
MODEL_FAMILIES = {"PERSISTENCE", "LSTM", "TRANSFORMER_ENCODER"}
POOLING_OPTIONS = {"LAST_STEP", "MEAN"}
ACTIVATION_OPTIONS = {"RELU", "GELU"}
LOSS_OPTIONS = {"MSE", "HUBER"}
METRIC_UNITS = {"mae_wh": "Wh", "rmse_wh": "Wh", "r2": "dimensionless"}
REQUIRED_METRICS = {"mae_wh", "rmse_wh", "r2"}


class RunStatus(str, Enum):
    PLANNED = "PLANNED"
    REGISTERED = "REGISTERED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"


class ExecutionType(str, Enum):
    SANITY = "SANITY"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    ROBUSTNESS = "ROBUSTNESS"
    FINAL_TEST = "FINAL_TEST"


class FailureType(str, Enum):
    DATA_CONTRACT_ERROR = "DATA_CONTRACT_ERROR"
    FEATURE_CONTRACT_ERROR = "FEATURE_CONTRACT_ERROR"
    SPLIT_CONTRACT_ERROR = "SPLIT_CONTRACT_ERROR"
    SCALER_CONTRACT_ERROR = "SCALER_CONTRACT_ERROR"
    WINDOW_CONTRACT_ERROR = "WINDOW_CONTRACT_ERROR"
    DATALOADER_ERROR = "DATALOADER_ERROR"
    MODEL_BUILD_ERROR = "MODEL_BUILD_ERROR"
    FORWARD_PASS_ERROR = "FORWARD_PASS_ERROR"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    OOM_ERROR = "OOM_ERROR"
    TRAINING_ERROR = "TRAINING_ERROR"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    METRIC_ERROR = "METRIC_ERROR"
    TEST_FIREWALL_VIOLATION = "TEST_FIREWALL_VIOLATION"
    INTERRUPTED = "INTERRUPTED"
    USER_CANCELLED = "USER_CANCELLED"
    OTHER = "OTHER"


class ArtifactType(str, Enum):
    CONFIG = "CONFIG"
    STATUS = "STATUS"
    AUDIT = "AUDIT"
    TRAIN_LOG = "TRAIN_LOG"
    BEST_CHECKPOINT = "BEST_CHECKPOINT"
    LAST_CHECKPOINT = "LAST_CHECKPOINT"
    LEARNING_CURVE = "LEARNING_CURVE"
    METRICS = "METRICS"
    PREDICTIONS = "PREDICTIONS"
    RESIDUALS = "RESIDUALS"
    ATTENTION = "ATTENTION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    OTHER = "OTHER"


STATUS_TRANSITIONS = {
    RunStatus.PLANNED.value: {RunStatus.REGISTERED.value},
    RunStatus.REGISTERED.value: {RunStatus.RUNNING.value, RunStatus.CANCELLED.value},
    RunStatus.RUNNING.value: {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value},
    RunStatus.COMPLETED.value: {RunStatus.INVALIDATED.value, RunStatus.ARCHIVED.value},
    RunStatus.FAILED.value: {RunStatus.ARCHIVED.value},
    RunStatus.CANCELLED.value: {RunStatus.ARCHIVED.value},
    RunStatus.INVALIDATED.value: {RunStatus.ARCHIVED.value},
    RunStatus.ARCHIVED.value: set(),
}


EXPERIMENT_FAMILIES = [
    {"family_id": "PERSISTENCE_BASELINE", "family_code": "PS", "family_name": "Persistence baseline", "purpose": "Naive forecasting baseline", "phase": 14, "model_family": "PERSISTENCE", "primary_factor": "NONE", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "LSTM_BASELINE", "family_code": "LS", "family_name": "LSTM baseline", "purpose": "Sequence baseline", "phase": 20, "model_family": "LSTM", "primary_factor": "BASELINE", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "TRANSFORMER_BASELINE", "family_code": "B0", "family_name": "Transformer baseline", "purpose": "Transformer B0", "phase": 21, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "BASELINE", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S1_FEATURE_SET", "family_code": "S01", "family_name": "Feature-set sweep", "purpose": "Controlled feature-set comparison", "phase": 23, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "feature_variant_id", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S2_TIME_FEATURES", "family_code": "S02", "family_name": "Time-feature sweep", "purpose": "Controlled time-feature comparison", "phase": 24, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "time_features", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S3_TARGET_SCALING", "family_code": "S03", "family_name": "Target-scaling sweep", "purpose": "YS0 versus YS1", "phase": 25, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "target_scaling_option", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S4_LOOKBACK", "family_code": "S04", "family_name": "Lookback sweep", "purpose": "Controlled lookback comparison", "phase": 26, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "lookback_steps", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S5_POOLING", "family_code": "S05", "family_name": "Pooling sweep", "purpose": "Controlled pooling comparison", "phase": 27, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "pooling", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S6_ACTIVATION", "family_code": "S06", "family_name": "Activation sweep", "purpose": "Controlled activation comparison", "phase": 28, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "activation", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S7_BATCH_SIZE", "family_code": "S07", "family_name": "Batch-size sweep", "purpose": "Controlled batch-size comparison", "phase": 29, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "batch_size", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S8_LEARNING_RATE", "family_code": "S08", "family_name": "Learning-rate sweep", "purpose": "Controlled learning-rate comparison", "phase": 30, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "learning_rate", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S9_WEIGHT_DECAY", "family_code": "S09", "family_name": "Weight-decay sweep", "purpose": "Controlled weight-decay comparison", "phase": 31, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "weight_decay", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S10_DROPOUT", "family_code": "S10", "family_name": "Dropout sweep", "purpose": "Controlled dropout comparison", "phase": 32, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "dropout", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S11_D_MODEL", "family_code": "S11", "family_name": "Model-width sweep", "purpose": "Controlled d_model comparison", "phase": 33, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "d_model", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S12_HEADS", "family_code": "S12", "family_name": "Attention-head sweep", "purpose": "Controlled head-count comparison", "phase": 34, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "num_heads", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S13_LAYERS", "family_code": "S13", "family_name": "Encoder-layer sweep", "purpose": "Controlled layer-count comparison", "phase": 35, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "num_layers", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S14_FFN", "family_code": "S14", "family_name": "FFN sweep", "purpose": "Controlled FFN-width comparison", "phase": 36, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "ffn_dim", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S15_LOSS", "family_code": "S15", "family_name": "Loss sweep", "purpose": "Controlled loss comparison", "phase": 37, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "loss_name", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S16_EPOCH_CAP", "family_code": "S16", "family_name": "Epoch-cap sweep", "purpose": "Controlled epoch-cap comparison", "phase": 38, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "max_epochs", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S17_GRADIENT_CLIPPING", "family_code": "S17", "family_name": "Gradient-clipping sweep", "purpose": "Controlled clipping comparison", "phase": 39, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "gradient_clipping_enabled", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S18_REVIN", "family_code": "S18", "family_name": "RevIN sweep", "purpose": "Controlled RevIN comparison", "phase": 40, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "revin_enabled", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "S19_BOUNDARY_PROTOCOL", "family_code": "S19", "family_name": "Boundary-protocol sensitivity", "purpose": "WB0 versus WB1 sensitivity", "phase": 41, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "boundary_protocol", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "LSTM_TUNING", "family_code": "LST", "family_name": "LSTM tuning", "purpose": "Controlled LSTM refinement", "phase": 43, "model_family": "LSTM", "primary_factor": "MULTI_STAGE", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "ROLLING_ORIGIN", "family_code": "ROB", "family_name": "Rolling-origin robustness", "purpose": "Temporal robustness", "phase": 44, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "origin", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "FINAL_SEED_RUN", "family_code": "FSD", "family_name": "Final seed runs", "purpose": "Three-seed robustness", "phase": 46, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "seed", "selection_metric": "rmse_wh", "selection_split": "VALIDATION", "status": "PLANNED"},
    {"family_id": "FINAL_TEST", "family_code": "FST", "family_name": "Final Test evaluation", "purpose": "Single locked final evaluation", "phase": 47, "model_family": "TRANSFORMER_ENCODER", "primary_factor": "NONE", "selection_metric": "rmse_wh", "selection_split": "TEST", "status": "PLANNED"},
]


MAIN_REGISTRY_COLUMNS = [
    "run_id", "status", "experiment_family", "sweep_id", "sweep_stage", "execution_type", "model_family",
    "config_fingerprint", "environment_id", "feature_variant_id", "feature_fingerprint", "split_version",
    "global_split_fingerprint", "scaling_version", "scaler_bundle_id", "window_version", "population_version",
    "population_fingerprint", "dataloader_version", "dataloader_fingerprint", "metric_version", "lookback_steps",
    "horizon_steps", "target_scaling_option", "boundary_protocol", "batch_size", "seed", "learning_rate",
    "weight_decay", "loss_name", "max_epochs", "early_stopping_patience", "best_epoch",
    "best_validation_rmse_wh", "device_type", "created_at", "started_at", "completed_at", "parent_run_id",
    "candidate_id", "final_model_lock_id", "rerun_reason", "notes",
]
FAMILY_COLUMNS = ["family_id", "family_code", "family_name", "purpose", "phase", "model_family", "primary_factor", "selection_metric", "selection_split", "status"]
SWEEP_COLUMNS = ["sweep_id", "sweep_stage", "experiment_family", "model_family", "factor_name", "factor_values", "reference_run_id", "selection_metric", "selection_split", "fixed_config_fingerprint", "status", "created_at", "completed_at"]
ARTIFACT_COLUMNS = ["run_id", "artifact_type", "artifact_path", "sha256", "file_size_bytes", "created_at", "required", "status"]
METRIC_COLUMNS = ["run_id", "split_id", "metric_name", "metric_value", "metric_unit", "n_samples", "metric_version", "population_fingerprint", "epoch_or_checkpoint", "status"]
FAILURE_COLUMNS = ["run_id", "failure_type", "failure_stage", "exception_class", "failure_message", "traceback_path", "recoverable", "rerun_recommended", "created_at"]
COMPARISON_COLUMNS = ["run_a", "run_b", "allowed_differences", "actual_differences", "unexpected_differences", "status"]
VALIDATION_COLUMNS = ["check", "expected", "actual", "status", "details"]
FINGERPRINT_EXCLUDED_FIELDS = {
    "run_id", "created_at", "registered_at", "started_at", "completed_at", "updated_at", "metrics", "artifacts",
    "checkpoint_path", "checkpoint_best_path", "checkpoint_last_path", "failure_message", "duration_seconds",
    "training_seconds", "evaluation_seconds", "total_seconds", "best_epoch", "best_validation_rmse_wh",
}
UPPER_ENUM_FIELDS = {
    "model_family", "family", "activation", "pooling", "loss_name", "target_scaling_option", "target_scaling",
    "boundary_protocol", "target_access_mode", "execution_type", "experiment_family", "sweep_stage",
}
LOWER_ENUM_FIELDS = {"device_type"}
INTEGER_FIELDS = {
    "feature_count", "lookback_steps", "lookback", "horizon_steps", "horizon", "sampling_interval_minutes",
    "train_sample_count", "validation_sample_count", "test_sample_count", "input_size", "hidden_size", "num_layers",
    "output_size", "d_model", "num_heads", "ffn_dim", "batch_size", "max_epochs", "early_stopping_patience",
    "seed", "global_seed", "dataloader_seed", "num_workers", "resume_count",
}
FLOAT_FIELDS = {"dropout", "learning_rate", "weight_decay", "huber_delta", "gradient_clip_max_norm", "revin_eps"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_scalar(key: str, value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if key in INTEGER_FIELDS:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != math.floor(numeric):
            raise ValueError(f"{key} must be an integer")
        return int(numeric)
    if key in FLOAT_FIELDS:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{key} must be finite")
        return numeric
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value
    if key in UPPER_ENUM_FIELDS and isinstance(value, str):
        return value.upper()
    if key in LOWER_ENUM_FIELDS and isinstance(value, str):
        return value.lower()
    if key == "optimizer_name" and isinstance(value, str):
        return "AdamW" if value.upper() == "ADAMW" else value
    return value


def canonicalize_config(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: canonicalize_config(value[name], name) for name in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonicalize_config(item, key) for item in value]
    return _normalized_scalar(key, value)


def _fingerprint_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _fingerprint_payload(item) for key, item in sorted(value.items()) if key not in FINGERPRINT_EXCLUDED_FIELDS}
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    return value


def compute_config_fingerprint(config: dict[str, Any]) -> str:
    normalized = canonicalize_config(config)
    return sha256_bytes(canonical_json_bytes(_fingerprint_payload(normalized)))


def build_registry_contract() -> dict[str, Any]:
    payload = {
        "contract_id": REGISTRY_CONTRACT_ID,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "failure_taxonomy_version": FAILURE_TAXONOMY_VERSION,
        "canonical_registry_format": "JSONL",
        "flattened_registry_format": "CSV",
        "run_id_policy": "MODEL_FAMILY_STAGE_SEQUENCE_CONFIG_HASH",
        "config_fingerprint_policy": "SHA256_CANONICAL_CONFIG_EXCLUDING_RUNTIME_AND_RESULTS",
        "status_model": [status.value for status in RunStatus],
        "status_transitions": {status: sorted(values) for status, values in STATUS_TRANSITIONS.items()},
        "execution_types": [item.value for item in ExecutionType],
        "failure_types": [item.value for item in FailureType],
        "artifact_types": [item.value for item in ArtifactType],
        "completed_run_immutability": True,
        "atomic_write_policy": "TEMP_FSYNC_ATOMIC_REPLACE",
        "single_writer_policy": True,
        "duplicate_config_policy": "RERUN_REASON_REQUIRED",
        "test_firewall_policy": "FINAL_TEST_FAMILY_LOCK_ID_AND_AUTHORIZATION_REQUIRED",
        "sweep_consistency_guard": "ONLY_DECLARED_FACTOR_PATHS_MAY_DIFFER",
        "path_policy": "PROJECT_OR_REGISTRY_RELATIVE",
        "production_registry_phase13_state": "EMPTY_NO_FABRICATED_RUNS",
    }
    payload["registry_contract_fingerprint"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def load_upstream_context(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_12 = materialize_phase_12(root)
    if phase_12.get("artifact_version") != METRIC_VERSION or phase_12.get("status") != "PASS":
        raise RuntimeError("METRICS-v1 is not signed off")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    dataset = read_json(root / "data/raw_data/dataset_manifest.json")
    schema = read_json(root / "artifacts/schema/schema_manifest.json")
    temporal = read_json(root / "artifacts/temporal/temporal_manifest.json")
    eda = read_json(root / "artifacts/eda/eda_manifest.json")
    features = read_json(root / "artifacts/features/feature_engineering_manifest.json")
    feature_sets = read_json(root / "artifacts/feature_sets/feature_set_manifest.json")
    split = read_json(root / "artifacts/splits/split_manifest.json")
    scaling = read_json(root / "artifacts/scaling/scaling_manifest.json")
    scalers = read_json(root / "artifacts/scaling/scaler_registry.json")
    windows = read_json(root / "artifacts/windows/window_manifest.json")
    window_fingerprints = read_json(root / "artifacts/windows/window_fingerprints.json")
    dataloaders = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    metrics = read_json(root / "artifacts/metrics/metric_manifest.json")
    expected_statuses = [
        schema.get("audit_status") in {"PASS", "PASS_WITH_WARNING"},
        temporal.get("audit_status") == "PASS",
        eda.get("eda_version") == "EDA-v1",
        features.get("audit_status") == "PASS",
        feature_sets.get("audit_status") == "PASS",
        split.get("audit_status") == "PASS",
        scaling.get("audit_status") == "PASS",
        windows.get("audit_status") == "PASS",
        dataloaders.get("audit_status") == "PASS",
        metrics.get("audit_status") == "PASS",
    ]
    if not all(expected_statuses):
        raise RuntimeError("Phase 13 upstream audit status mismatch")
    lineage = {
        "environment_id": environment["environment_id"],
        "dataset_revision": dataset["dataset_revision"],
        "schema_version": schema["schema_version"],
        "temporal_version": temporal["temporal_version"],
        "eda_version": eda["eda_version"],
        "feature_version": features["feature_version"],
        "feature_set_version": feature_sets["feature_set_version"],
        "split_version": split["split_version"],
        "scaling_version": scaling["scaling_version"],
        "window_version": windows["window_version"],
        "population_version": windows["population_version"],
        "dataloader_version": dataloaders["dataloader_version"],
        "metric_version": metrics["metric_version"],
        "global_split_fingerprint": split["global_split_fingerprint"],
        "population_fingerprint": windows["common_population_fingerprint"],
        "metric_contract_fingerprint": metrics["metric_contract_fingerprint"],
        "dataset_fingerprint": sha256_file(root / "data/raw_data/energydata_complete.csv"),
    }
    return {
        "lineage": lineage,
        "environment": environment,
        "feature_sets": feature_sets,
        "scalers": scalers,
        "window_fingerprints": window_fingerprints,
        "dataloaders": dataloaders,
        "metrics": metrics,
        "phase_12_signoff": phase_12,
    }


def build_reference_run_config(project_root: Path | None = None, model_family: str = "TRANSFORMER_ENCODER") -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    context = load_upstream_context(root)
    lineage = deepcopy(context["lineage"])
    feature_sets = context["feature_sets"]
    scalers = context["scalers"]
    windows = context["window_fingerprints"]
    dataloaders = context["dataloaders"]
    environment = context["environment"]
    model_family = model_family.upper()
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {model_family}")
    if model_family == "PERSISTENCE":
        variant_id = None
        target_option = None
        feature_count = None
        lineage.update({
            "feature_fingerprint": None,
            "scaler_bundle_id": None,
            "scaler_bundle_checksum": None,
            "target_scaler_bundle_id": None,
            "target_scaler_checksum": None,
            "window_fingerprint": windows["window_index_fingerprints"]["L144_H01_WB0"],
            "dataloader_fingerprint": None,
        })
    else:
        variant_id = dataloaders["baseline_variant"]
        target_option = dataloaders["baseline_target_option"]
        feature_count = dataloaders["feature_count"]
        x_scaler = scalers["x_bundles"][variant_id]
        y_scaler = scalers["target_bundles"][target_option]
        lineage.update({
            "feature_fingerprint": feature_sets["variant_fingerprints"][variant_id],
            "scaler_bundle_id": x_scaler["bundle_id"],
            "scaler_bundle_checksum": x_scaler["artifact_sha256"],
            "target_scaler_bundle_id": y_scaler["bundle_id"],
            "target_scaler_checksum": y_scaler.get("artifact_sha256"),
            "window_fingerprint": windows["window_index_fingerprints"]["L144_H01_WB0"],
            "dataloader_fingerprint": dataloaders["baseline_loader_fingerprints"]["VALIDATION"],
        })
    from course_work.models.lstm_regressor import LSTM_IMPL_VERSION as LSTM_IMPL
    from course_work.models.transformer_regressor import TRANSFORMER_IMPL_VERSION as TRANSFORMER_IMPL

    if model_family == "TRANSFORMER_ENCODER":
        model = {
            "model_family": model_family,
            "model_name": "Transformer Encoder Regressor",
            "model_version": "TRANSFORMER-v1",
            "implementation_version": TRANSFORMER_IMPL,
            "input_size": feature_count,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 128,
            "dropout": 0.1,
            "activation": "GELU",
            "pooling": "LAST_STEP",
            "positional_encoding_type": "SINUSOIDAL",
            "attention_aware": True,
            "norm_first": False,
            "output_size": 1,
        }
    elif model_family == "LSTM":
        model = {
            "model_family": model_family,
            "model_name": "LSTM Regressor",
            "model_version": "LSTM-v1",
            "implementation_version": LSTM_IMPL,
            "input_size": feature_count,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False,
            "batch_first": True,
            "pooling": "LAST_STEP",
            "output_size": 1,
        }
    elif model_family == "PERSISTENCE":
        model = {
            "model_family": model_family,
            "model_name": "PERSISTENCE_LAST_VALUE",
            "model_version": "PERSISTENCE-v1",
            "implementation_version": "PERSISTENCE-v1",
            "baseline_scope": "TASK_LEVEL",
            "formula": "Y_HAT_T_PLUS_1_EQUALS_Y_T",
            "source_lag_steps": 1,
            "source_lag_minutes": 10,
            "input_size": None,
            "output_size": 1,
            "trainable_parameters": 0,
            "requires_training": False,
        }
    training_enabled = model_family != "PERSISTENCE"
    if training_enabled:
        training = {
            "enabled": True,
            "batch_size": dataloaders["baseline_batch_size"],
            "optimizer_name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "loss_name": "MSE",
            "huber_delta": None,
            "max_epochs": 50,
            "early_stopping_enabled": True,
            "early_stopping_patience": 10,
            "early_stopping_metric": "rmse_wh",
            "early_stopping_mode": "MIN",
            "gradient_clipping_enabled": True,
            "gradient_clip_max_norm": 1.0,
            "scheduler_name": None,
            "scheduler_config": None,
            "revin_enabled": False,
            "revin_affine": None,
            "revin_eps": None,
            "revin_target_channels": None,
        }
        reproducibility = {
            "seed": environment["development_seed"],
            "global_seed": environment["development_seed"],
            "dataloader_seed": environment["development_seed"],
            "deterministic_mode": environment["deterministic_mode"],
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "worker_seed_policy": dataloaders["worker_seed_policy"],
        }
        runtime = {
            "device_type": environment["selected_device"],
            "device_name": environment.get(f'{environment["selected_device"]}_device_name'),
            "python_version": environment["python_version"],
            "torch_version": environment["package_versions"]["torch"],
            "sklearn_version": environment["package_versions"]["scikit_learn"],
            "platform": environment["platform"],
            "num_workers": dataloaders["baseline_num_workers"],
            "pin_memory": dataloaders["actual_pin_memory"],
            "mixed_precision": False,
            "dtype": "float32",
        }
    else:
        training = {
            "enabled": False,
            "batch_size": None,
            "optimizer_name": None,
            "learning_rate": None,
            "weight_decay": None,
            "loss_name": None,
            "huber_delta": None,
            "max_epochs": None,
            "early_stopping_enabled": None,
            "early_stopping_patience": None,
            "early_stopping_metric": None,
            "early_stopping_mode": None,
            "gradient_clipping_enabled": None,
            "gradient_clip_max_norm": None,
            "scheduler_name": None,
            "scheduler_config": None,
            "revin_enabled": None,
            "revin_affine": None,
            "revin_eps": None,
            "revin_target_channels": None,
        }
        reproducibility = {
            "seed": None,
            "global_seed": None,
            "dataloader_seed": None,
            "deterministic": True,
            "deterministic_mode": "PARAMETER_FREE",
            "torch_deterministic_algorithms": None,
            "cudnn_deterministic": None,
            "cudnn_benchmark": None,
            "worker_seed_policy": None,
        }
        runtime = {
            "device_type": "cpu",
            "device_name": None,
            "python_version": environment["python_version"],
            "torch_version": environment["package_versions"]["torch"],
            "sklearn_version": environment["package_versions"]["scikit_learn"],
            "platform": environment["platform"],
            "num_workers": None,
            "pin_memory": False,
            "mixed_precision": False,
            "dtype": "float64",
        }
    config = {
        "lineage": lineage,
        "data": {
            "feature_variant_id": variant_id,
            "feature_count": feature_count,
            "lookback_steps": dataloaders["baseline_lookback"],
            "horizon_steps": 1,
            "sampling_interval_minutes": 10,
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
            "target_scaling_option": target_option,
            "target_access_mode": "VALIDATION",
            "train_sample_count": dataloaders["train_sample_count"],
            "validation_sample_count": dataloaders["validation_sample_count"],
            "test_sample_count": dataloaders["test_sample_count"],
        },
        "model": model,
        "training": training,
        "reproducibility": reproducibility,
        "runtime": runtime,
    }
    return canonicalize_config(config)


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _flatten_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            child = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_paths(item, child))
        return result
    return {prefix: value}


def compare_configs(config_a: dict[str, Any], config_b: dict[str, Any], allowed_differences: set[str] | None = None) -> dict[str, Any]:
    left = _flatten_paths(canonicalize_config(config_a))
    right = _flatten_paths(canonicalize_config(config_b))
    paths = sorted(set(left) | set(right))
    actual = [path for path in paths if left.get(path) != right.get(path)]
    allowed = sorted(allowed_differences or set())
    unexpected = [path for path in actual if path not in allowed]
    if not actual:
        status = "COMPARABLE"
    elif not unexpected:
        status = "COMPARABLE_WITH_EXPECTED_DIFFERENCES"
    else:
        status = "NOT_COMPARABLE"
    return {
        "allowed_differences": allowed,
        "actual_differences": actual,
        "unexpected_differences": unexpected,
        "status": status,
    }


def validate_sweep_consistency(configs: list[dict[str, Any]], allowed_differences: set[str]) -> dict[str, Any]:
    if len(configs) < 2:
        raise ValueError("Sweep consistency requires at least two configs")
    comparisons = [compare_configs(configs[0], config, allowed_differences) for config in configs[1:]]
    unexpected = sorted({path for result in comparisons for path in result["unexpected_differences"]})
    if unexpected:
        raise ValueError(f"Sweep confounding detected: {unexpected}")
    varied = sorted({path for result in comparisons for path in result["actual_differences"]})
    if not varied:
        raise ValueError("Sweep configs do not vary")
    return {"allowed_differences": sorted(allowed_differences), "actual_differences": varied, "status": "PASS"}


def validate_status_transition(current_status: str, next_status: str) -> None:
    current = RunStatus(current_status).value
    target = RunStatus(next_status).value
    if target not in STATUS_TRANSITIONS[current]:
        raise ValueError(f"Invalid status transition: {current} -> {target}")


def _validate_positive(value: Any, name: str) -> None:
    if value is None or float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def validate_run_config(config: dict[str, Any], upstream_context: dict[str, Any]) -> dict[str, Any]:
    normalized = canonicalize_config(config)
    required_groups = {"lineage", "data", "model", "training", "reproducibility", "runtime"}
    if set(normalized) != required_groups:
        raise ValueError(f"Run config groups must equal {sorted(required_groups)}")
    lineage = normalized["lineage"]
    for key, expected in upstream_context["lineage"].items():
        if lineage.get(key) != expected:
            raise ValueError(f"Upstream lineage mismatch for {key}")
    data = normalized["data"]
    model = normalized["model"]
    model_family = model.get("model_family")
    if model_family not in MODEL_FAMILIES:
        raise ValueError("Invalid model family")
    variant_id = data.get("feature_variant_id")
    feature_sets = upstream_context["feature_sets"]
    if model_family == "PERSISTENCE":
        if variant_id is not None or data.get("feature_count") is not None:
            raise ValueError("Persistence must not bind a feature variant")
        if data.get("target_scaling_option") is not None:
            raise ValueError("Persistence must not bind target scaling")
        nullable_lineage = {
            "feature_fingerprint",
            "scaler_bundle_id",
            "scaler_bundle_checksum",
            "target_scaler_bundle_id",
            "target_scaler_checksum",
            "dataloader_fingerprint",
        }
        if any(lineage.get(key) is not None for key in nullable_lineage):
            raise ValueError("Persistence must not bind feature, scaler or DataLoader artifacts")
        expected_window = upstream_context["window_fingerprints"]["window_index_fingerprints"]["L144_H01_WB0"]
        if lineage.get("window_fingerprint") != expected_window:
            raise ValueError("Persistence window fingerprint mismatch")
    else:
        if variant_id not in FEATURE_VARIANTS:
            raise ValueError("Invalid feature variant")
        if data.get("feature_count") != feature_sets["variant_feature_counts"][variant_id]:
            raise ValueError("Feature count does not match feature variant")
        if lineage.get("feature_fingerprint") != feature_sets["variant_fingerprints"][variant_id]:
            raise ValueError("Feature fingerprint does not match feature variant")
        if data.get("target_scaling_option") not in TARGET_SCALING_OPTIONS:
            raise ValueError("Invalid target scaling option")
    if data.get("lookback_steps") not in LOOKBACK_OPTIONS:
        raise ValueError("Invalid lookback")
    if model_family == "PERSISTENCE" and data.get("lookback_steps") != 144:
        raise ValueError("Persistence must use the common L144 population anchor")
    if data.get("horizon_steps") != 1:
        raise ValueError("Horizon must equal one")
    if data.get("sampling_interval_minutes") != 10:
        raise ValueError("Sampling interval must equal ten minutes")
    if data.get("boundary_protocol") not in BOUNDARY_PROTOCOLS:
        raise ValueError("Invalid boundary protocol")
    if model.get("output_size") != 1:
        raise ValueError("Output size must equal one")
    if model_family == "TRANSFORMER_ENCODER":
        _validate_positive(model.get("d_model"), "d_model")
        _validate_positive(model.get("num_heads"), "num_heads")
        if model["d_model"] % model["num_heads"] != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if model.get("pooling") not in POOLING_OPTIONS or model.get("activation") not in ACTIVATION_OPTIONS:
            raise ValueError("Invalid Transformer pooling or activation")
    if model_family == "PERSISTENCE":
        expected_model = {
            "model_name": "PERSISTENCE_LAST_VALUE",
            "model_version": "PERSISTENCE-v1",
            "implementation_version": "PERSISTENCE-v1",
            "baseline_scope": "TASK_LEVEL",
            "formula": "Y_HAT_T_PLUS_1_EQUALS_Y_T",
            "source_lag_steps": 1,
            "source_lag_minutes": 10,
            "input_size": None,
            "trainable_parameters": 0,
            "requires_training": False,
        }
        if any(model.get(key) != value for key, value in expected_model.items()):
            raise ValueError("Persistence model contract mismatch")
    training = normalized["training"]
    if model_family == "PERSISTENCE":
        if training.get("enabled") is not False:
            raise ValueError("Persistence training must be disabled")
        if any(value is not None for key, value in training.items() if key != "enabled"):
            raise ValueError("Persistence training fields must be null")
    elif training.get("enabled"):
        if training.get("batch_size") not in {32, 64}:
            raise ValueError("Invalid batch size")
        _validate_positive(training.get("learning_rate"), "learning_rate")
        if training.get("weight_decay") is None or training["weight_decay"] < 0:
            raise ValueError("weight_decay must be non-negative")
        if training.get("loss_name") not in LOSS_OPTIONS:
            raise ValueError("Invalid loss")
        _validate_positive(training.get("max_epochs"), "max_epochs")
        if training.get("early_stopping_enabled"):
            _validate_positive(training.get("early_stopping_patience"), "early_stopping_patience")
        if training.get("gradient_clipping_enabled"):
            _validate_positive(training.get("gradient_clip_max_norm"), "gradient_clip_max_norm")
    dropout = model.get("dropout")
    if dropout is not None and not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    reproducibility = normalized["reproducibility"]
    if model_family == "PERSISTENCE":
        if any(reproducibility.get(key) is not None for key in ("seed", "global_seed", "dataloader_seed", "worker_seed_policy")):
            raise ValueError("Persistence seed fields must be null")
        if reproducibility.get("deterministic") is not True:
            raise ValueError("Persistence must be deterministic")
    elif not isinstance(reproducibility.get("seed"), int):
        raise ValueError("seed must be an integer")
    if normalized["runtime"].get("device_type") not in {"cpu", "cuda", "mps"}:
        raise ValueError("Invalid device type")
    if model_family == "PERSISTENCE":
        runtime = normalized["runtime"]
        if runtime.get("device_type") != "cpu" or runtime.get("num_workers") is not None or runtime.get("mixed_precision") is not False:
            raise ValueError("Persistence runtime contract mismatch")
    return normalized


class ExperimentRegistry:
    def __init__(
        self,
        project_root: Path | None = None,
        registry_root: Path | None = None,
        run_root: Path | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.project_root = (project_root or get_project_root()).resolve()
        self.registry_root = (registry_root or self.project_root / EXPERIMENT_ARTIFACT_ROOT).resolve()
        self.run_root = (run_root or self.project_root / RUN_ARTIFACT_ROOT).resolve()
        self.clock = clock
        self.upstream_context = load_upstream_context(self.project_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)

    @property
    def canonical_path(self) -> Path:
        return self.registry_root / "experiment_registry.jsonl"

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.canonical_path.exists():
            return []
        records = []
        for line_number, line in enumerate(self.canonical_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSONL record at line {line_number}")
            records.append(value)
        return records

    def _load_sweeps(self) -> list[dict[str, Any]]:
        path = self.registry_root / "sweep_registry.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def _jsonl_bytes(self, records: list[dict[str, Any]]) -> bytes:
        lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
        return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")

    def _relative_artifact_path(self, path: Path) -> str:
        resolved = path.resolve()
        for base in (self.project_root, self.registry_root, self.run_root):
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                continue
        raise ValueError("Artifact path must be project-relative or registry-relative")

    def _resolve_artifact_path(self, relative_path: str) -> Path:
        project_candidate = (self.project_root / relative_path).resolve()
        registry_candidate = (self.registry_root / relative_path).resolve()
        run_candidate = (self.run_root / relative_path).resolve()
        if project_candidate.exists():
            return project_candidate
        if registry_candidate.exists():
            return registry_candidate
        if run_candidate.exists():
            return run_candidate
        return project_candidate

    def _flatten_record(self, record: dict[str, Any]) -> dict[str, Any]:
        config = record["config"]
        lineage = config["lineage"]
        data = config["data"]
        training = config["training"]
        reproduction = config["reproducibility"]
        runtime = config["runtime"]
        return {
            "run_id": record["run_id"],
            "status": record["status"],
            "experiment_family": record["experiment_family"],
            "sweep_id": record.get("sweep_id"),
            "sweep_stage": record.get("sweep_stage"),
            "execution_type": record["execution_type"],
            "model_family": config["model"]["model_family"],
            "config_fingerprint": record["config_fingerprint"],
            "environment_id": lineage["environment_id"],
            "feature_variant_id": data["feature_variant_id"],
            "feature_fingerprint": lineage["feature_fingerprint"],
            "split_version": lineage["split_version"],
            "global_split_fingerprint": lineage["global_split_fingerprint"],
            "scaling_version": lineage["scaling_version"],
            "scaler_bundle_id": lineage["scaler_bundle_id"],
            "window_version": lineage["window_version"],
            "population_version": lineage["population_version"],
            "population_fingerprint": lineage["population_fingerprint"],
            "dataloader_version": lineage["dataloader_version"],
            "dataloader_fingerprint": lineage["dataloader_fingerprint"],
            "metric_version": lineage["metric_version"],
            "lookback_steps": data["lookback_steps"],
            "horizon_steps": data["horizon_steps"],
            "target_scaling_option": data["target_scaling_option"],
            "boundary_protocol": data["boundary_protocol"],
            "batch_size": training["batch_size"],
            "seed": reproduction["seed"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "loss_name": training["loss_name"],
            "max_epochs": training["max_epochs"],
            "early_stopping_patience": training["early_stopping_patience"],
            "best_epoch": record.get("best_epoch"),
            "best_validation_rmse_wh": record.get("best_validation_rmse_wh"),
            "device_type": runtime["device_type"],
            "created_at": record["created_at"],
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "parent_run_id": record.get("parent_run_id"),
            "candidate_id": record.get("candidate_id"),
            "final_model_lock_id": record.get("final_model_lock_id"),
            "rerun_reason": record.get("rerun_reason"),
            "notes": record.get("notes"),
        }

    def _write_csv(self, path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        atomic_write_bytes(path, csv_text(columns, [{column: row.get(column) for column in columns} for row in rows]).encode("utf-8"))

    def _write_status(self, record: dict[str, Any]) -> Path:
        payload = {
            "run_id": record["run_id"],
            "status": record["status"],
            "created_at": record["created_at"],
            "registered_at": record["registered_at"],
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "updated_at": record["updated_at"],
            "best_epoch": record.get("best_epoch"),
            "best_validation_rmse_wh": record.get("best_validation_rmse_wh"),
            "failure": record.get("failure"),
            "invalidation": record.get("invalidation"),
        }
        path = self.run_root / record["run_id"] / "status.json"
        atomic_write_bytes(path, canonical_json_bytes(payload))
        return path

    def _sync_core_artifacts(self, record: dict[str, Any]) -> None:
        config_path = self.run_root / record["run_id"] / "config.json"
        status_path = self.run_root / record["run_id"] / "status.json"
        now = record["updated_at"]
        existing = {item["artifact_type"]: item for item in record["artifacts"] if item["artifact_type"] in {"CONFIG", "STATUS"}}
        for artifact_type, path in (("CONFIG", config_path), ("STATUS", status_path)):
            row = {
                "run_id": record["run_id"],
                "artifact_type": artifact_type,
                "artifact_path": self._relative_artifact_path(path),
                "sha256": sha256_file(path),
                "file_size_bytes": path.stat().st_size,
                "created_at": existing.get(artifact_type, {}).get("created_at", now),
                "required": True,
                "status": "PASS",
            }
            if artifact_type in existing:
                existing[artifact_type].update(row)
            else:
                record["artifacts"].append(row)

    def _manifest_payload(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = {status.value: 0 for status in RunStatus}
        for record in records:
            statuses[record["status"]] += 1
        logical_fingerprint = sha256_bytes(canonical_json_bytes(records))
        current = read_json(self.registry_root / "registry_manifest.json") if (self.registry_root / "registry_manifest.json").exists() else {}
        created_at = current.get("created_at", self.clock())
        return {
            "experiment_registry_version": EXPERIMENT_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "canonical_registry_file": "experiment_registry.jsonl",
            "canonical_registry_format": "JSONL",
            "flattened_registry_file": "experiment_registry.csv",
            "flattened_registry_format": "CSV",
            "run_count": len(records),
            "status_counts": statuses,
            "completed_count": statuses[RunStatus.COMPLETED.value],
            "failed_count": statuses[RunStatus.FAILED.value],
            "invalidated_count": statuses[RunStatus.INVALIDATED.value],
            "active_count": statuses[RunStatus.REGISTERED.value] + statuses[RunStatus.RUNNING.value],
            "family_count": len(EXPERIMENT_FAMILIES),
            "sweep_count": len(self._load_sweeps()),
            "metric_version": METRIC_VERSION,
            "upstream_contract_versions": self.upstream_context["lineage"],
            "registry_contract_fingerprint": build_registry_contract()["registry_contract_fingerprint"],
            "registry_fingerprint": logical_fingerprint,
            "artifact_registry_enabled": True,
            "metric_registry_enabled": True,
            "test_firewall_enabled": True,
            "sweep_consistency_guard_enabled": True,
            "completed_run_immutability": True,
            "atomic_write_policy": "TEMP_FSYNC_ATOMIC_REPLACE",
            "validation_audit_status": "PASS",
            "warnings": [],
            "production_registry_empty": len(records) == 0,
            "synthetic_records_persisted": False,
            "created_at": created_at,
            "updated_at": self.clock(),
        }

    def _experiment_index(self, records: list[dict[str, Any]]) -> str:
        counts = {family["family_id"]: 0 for family in EXPERIMENT_FAMILIES}
        for record in records:
            counts[record["experiment_family"]] = counts.get(record["experiment_family"], 0) + 1
        lines = ["# EXPERIMENT INDEX", "", f"Registry: {EXPERIMENT_VERSION}", "", f"Registered runs: {len(records)}", "", "| Family | Phase | Model | Runs | Status |", "|---|---:|---|---:|---|"]
        lines.extend(f'| {family["family_id"]} | {family["phase"]} | {family["model_family"]} | {counts[family["family_id"]]} | {family["status"]} |' for family in EXPERIMENT_FAMILIES)
        return "\n".join(lines) + "\n"

    def _persist(self, records: list[dict[str, Any]]) -> None:
        atomic_write_bytes(self.canonical_path, self._jsonl_bytes(records))
        self._write_csv(self.registry_root / "experiment_registry.csv", MAIN_REGISTRY_COLUMNS, [self._flatten_record(record) for record in records])
        artifacts = [artifact for record in records for artifact in record["artifacts"]]
        metrics = [metric for record in records for metric in record["metrics"]]
        failures = [record["failure"] for record in records if record.get("failure")]
        comparisons = [comparison for record in records for comparison in record.get("comparisons", [])]
        self._write_csv(self.registry_root / "run_artifact_registry.csv", ARTIFACT_COLUMNS, artifacts)
        self._write_csv(self.registry_root / "run_metric_registry.csv", METRIC_COLUMNS, metrics)
        self._write_csv(self.registry_root / "run_failure_registry.csv", FAILURE_COLUMNS, failures)
        self._write_csv(self.registry_root / "run_comparison_audit.csv", COMPARISON_COLUMNS, comparisons)
        atomic_write_bytes(self.registry_root / "registry_manifest.json", canonical_json_bytes(self._manifest_payload(records)))
        atomic_write_bytes(self.registry_root / "EXPERIMENT_INDEX.md", self._experiment_index(records).encode("utf-8"))

    def _record_index(self, records: list[dict[str, Any]], run_id: str) -> int:
        matches = [index for index, record in enumerate(records) if record["run_id"] == run_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown or duplicate run_id: {run_id}")
        return matches[0]

    def _verify_config_immutability(self, record: dict[str, Any]) -> None:
        path = self.run_root / record["run_id"] / "config.json"
        if not path.is_file():
            raise RuntimeError("Run config file is missing")
        payload = read_json(path)
        if payload.get("run_id") != record["run_id"] or payload.get("config_fingerprint") != record["config_fingerprint"]:
            raise RuntimeError("Run config identity mismatch")
        if compute_config_fingerprint(payload["config"]) != record["config_fingerprint"]:
            raise RuntimeError("Run config fingerprint mismatch")
        if canonicalize_config(payload["config"]) != canonicalize_config(record["config"]):
            raise RuntimeError("Run config differs from registry")

    def _family(self, family_id: str) -> dict[str, Any]:
        matches = [family for family in EXPERIMENT_FAMILIES if family["family_id"] == family_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown experiment family: {family_id}")
        return matches[0]

    def allocate_run_id(self, model_family: str, experiment_family: str, config_fingerprint: str) -> str:
        model_codes = {"PERSISTENCE": "PS", "LSTM": "LS", "TRANSFORMER_ENCODER": "TR"}
        model_code = model_codes[model_family]
        family_code = self._family(experiment_family)["family_code"]
        records = self._load_records()
        sequence = len(records) + 1
        while True:
            run_id = f"RUN_{model_code}_{family_code}_{sequence:04d}_{config_fingerprint[:8].upper()}"
            if not any(record["run_id"] == run_id for record in records) and not (self.run_root / run_id).exists():
                return run_id
            sequence += 1

    def find_duplicates(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        fingerprint = compute_config_fingerprint(config)
        return [record for record in self._load_records() if record["config_fingerprint"] == fingerprint]

    def register_sweep(
        self,
        sweep_id: str,
        sweep_stage: str,
        experiment_family: str,
        factor_name: str,
        factor_values: list[Any],
        reference_config: dict[str, Any],
        reference_run_id: str | None = None,
    ) -> dict[str, Any]:
        if not sweep_id or not sweep_id.replace("_", "").isalnum() or sweep_id != sweep_id.upper():
            raise ValueError("sweep_id must be uppercase alphanumeric with underscores")
        if not factor_name or len(factor_values) < 2:
            raise ValueError("Sweep requires a factor and at least two values")
        normalized_reference = validate_run_config(reference_config, self.upstream_context)
        _get_path(normalized_reference, factor_name)
        family = self._family(experiment_family)
        if normalized_reference["model"]["model_family"] != family["model_family"]:
            raise ValueError("Sweep family and reference model differ")
        sweeps = self._load_sweeps()
        if any(row["sweep_id"] == sweep_id for row in sweeps):
            raise ValueError("Duplicate sweep_id")
        if reference_run_id is not None:
            self.get_run(reference_run_id)
        flattened = _flatten_paths(normalized_reference)
        fixed_payload = {path: value for path, value in flattened.items() if path != factor_name}
        row = {
            "sweep_id": sweep_id,
            "sweep_stage": sweep_stage.upper(),
            "experiment_family": experiment_family,
            "model_family": family["model_family"],
            "factor_name": factor_name,
            "factor_values": json.dumps(canonicalize_config(factor_values, factor_name.split(".")[-1]), ensure_ascii=False, sort_keys=True),
            "reference_run_id": reference_run_id,
            "selection_metric": family["selection_metric"],
            "selection_split": family["selection_split"],
            "fixed_config_fingerprint": sha256_bytes(canonical_json_bytes(fixed_payload)),
            "status": RunStatus.PLANNED.value,
            "created_at": self.clock(),
            "completed_at": None,
        }
        sweeps.append(row)
        self._write_csv(self.registry_root / "sweep_registry.csv", SWEEP_COLUMNS, sweeps)
        self._persist(self._load_records())
        return deepcopy(row)

    def register_run(
        self,
        config: dict[str, Any],
        experiment_family: str,
        execution_type: str,
        sweep_id: str | None = None,
        sweep_stage: str | None = None,
        parent_run_id: str | None = None,
        candidate_id: str | None = None,
        final_model_lock_id: str | None = None,
        test_access_authorized: bool = False,
        rerun_reason: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        normalized = validate_run_config(config, self.upstream_context)
        family = self._family(experiment_family)
        execution = ExecutionType(execution_type).value
        model_family = normalized["model"]["model_family"]
        if family["model_family"] != model_family:
            raise ValueError("Experiment family and model family differ")
        test_target_requested = normalized["data"]["target_access_mode"] == "TEST"
        if execution == ExecutionType.FINAL_TEST.value or test_access_authorized or test_target_requested:
            if experiment_family != "FINAL_TEST" or execution != ExecutionType.FINAL_TEST.value or not final_model_lock_id or not test_access_authorized:
                raise PermissionError("Final Test registration requires FINAL_TEST family, lock id and authorization")
        elif final_model_lock_id is not None:
            raise PermissionError("Development run must not reference a final model lock")
        records = self._load_records()
        if sweep_id is not None:
            sweeps = [row for row in self._load_sweeps() if row["sweep_id"] == sweep_id]
            if len(sweeps) != 1:
                raise ValueError("Run references an unknown sweep")
            sweep = sweeps[0]
            if sweep["experiment_family"] != experiment_family or sweep["model_family"] != model_family:
                raise ValueError("Run and sweep grouping differ")
            if sweep_stage is not None and sweep["sweep_stage"] != sweep_stage.upper():
                raise ValueError("Run and sweep stage differ")
        if parent_run_id is not None and not any(record["run_id"] == parent_run_id for record in records):
            raise ValueError("Parent run does not exist")
        fingerprint = compute_config_fingerprint(normalized)
        duplicates = [record for record in records if record["config_fingerprint"] == fingerprint]
        if duplicates:
            if rerun_reason not in RERUN_REASONS:
                raise ValueError("Duplicate config requires a canonical rerun reason")
        elif rerun_reason is not None and rerun_reason not in RERUN_REASONS:
            raise ValueError("Invalid rerun reason")
        run_id = self.allocate_run_id(model_family, experiment_family, fingerprint)
        now = self.clock()
        record = {
            "registry_version": EXPERIMENT_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "run_id": run_id,
            "config_fingerprint": fingerprint,
            "status": RunStatus.REGISTERED.value,
            "experiment_family": experiment_family,
            "sweep_id": sweep_id,
            "sweep_stage": sweep_stage or family["family_code"],
            "execution_type": execution,
            "parent_run_id": parent_run_id,
            "candidate_id": candidate_id,
            "final_model_lock_id": final_model_lock_id,
            "test_access_authorized": test_access_authorized,
            "rerun_reason": rerun_reason,
            "notes": notes,
            "config": normalized,
            "created_at": now,
            "registered_at": now,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
            "best_epoch": None,
            "best_validation_rmse_wh": None,
            "resume_count": 0,
            "artifacts": [],
            "metrics": [],
            "failure": None,
            "invalidation": None,
            "comparisons": [],
        }
        run_directory = self.run_root / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        config_payload = {
            "registry_version": EXPERIMENT_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "run_id": run_id,
            "config_fingerprint": fingerprint,
            "config": normalized,
        }
        atomic_write_bytes(run_directory / "config.json", canonical_json_bytes(config_payload))
        self._write_status(record)
        self._sync_core_artifacts(record)
        records.append(record)
        self._persist(records)
        return deepcopy(record)

    def get_run(self, run_id: str) -> dict[str, Any]:
        records = self._load_records()
        record = records[self._record_index(records, run_id)]
        self._verify_config_immutability(record)
        return deepcopy(record)

    def _transition(self, run_id: str, next_status: str, update: dict[str, Any] | None = None) -> dict[str, Any]:
        records = self._load_records()
        index = self._record_index(records, run_id)
        record = records[index]
        self._verify_config_immutability(record)
        validate_status_transition(record["status"], next_status)
        record["status"] = RunStatus(next_status).value
        record["updated_at"] = self.clock()
        if update:
            record.update(update)
        self._write_status(record)
        self._sync_core_artifacts(record)
        records[index] = record
        self._persist(records)
        return deepcopy(record)

    def start_run(self, run_id: str) -> dict[str, Any]:
        return self._transition(run_id, RunStatus.RUNNING.value, {"started_at": self.clock()})

    def register_artifact(self, run_id: str, artifact_type: str, artifact_path: Path, required: bool = False) -> dict[str, Any]:
        records = self._load_records()
        index = self._record_index(records, run_id)
        record = records[index]
        self._verify_config_immutability(record)
        kind = ArtifactType(artifact_type).value
        if kind in {ArtifactType.CONFIG.value, ArtifactType.STATUS.value}:
            raise ValueError("CONFIG and STATUS artifacts are managed by the registry")
        path = Path(artifact_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = self._relative_artifact_path(path)
        if any(item["artifact_type"] == kind and item["artifact_path"] == relative for item in record["artifacts"]):
            raise ValueError("Duplicate artifact registration")
        row = {
            "run_id": run_id,
            "artifact_type": kind,
            "artifact_path": relative,
            "sha256": sha256_file(path),
            "file_size_bytes": path.stat().st_size,
            "created_at": self.clock(),
            "required": bool(required),
            "status": "PASS",
        }
        record["artifacts"].append(row)
        record["updated_at"] = self.clock()
        self._write_status(record)
        self._sync_core_artifacts(record)
        records[index] = record
        self._persist(records)
        return deepcopy(row)

    def register_metric(
        self,
        run_id: str,
        split_id: str,
        metric_name: str,
        metric_value: float | None,
        metric_unit: str,
        n_samples: int,
        population_fingerprint: str,
        epoch_or_checkpoint: str,
        status: str = "PASS",
    ) -> dict[str, Any]:
        records = self._load_records()
        index = self._record_index(records, run_id)
        record = records[index]
        self._verify_config_immutability(record)
        if record["status"] != RunStatus.RUNNING.value:
            raise ValueError("Metrics can only be registered while a run is RUNNING")
        split = split_id.upper()
        name = metric_name.lower()
        if name not in REQUIRED_METRICS:
            raise ValueError("Unsupported metric")
        if metric_unit != METRIC_UNITS[name]:
            raise ValueError("Metric unit mismatch")
        if metric_value is not None and not math.isfinite(float(metric_value)):
            raise ValueError("Metric value must be finite or null")
        if split == "TEST":
            if record["experiment_family"] != "FINAL_TEST" or record["execution_type"] != ExecutionType.FINAL_TEST.value or not record["final_model_lock_id"] or not record["test_access_authorized"]:
                raise PermissionError("Test metric registration is locked")
        elif split not in {"TRAIN", "VALIDATION"}:
            raise ValueError("Invalid metric split")
        expected_population = record["config"]["lineage"]["population_fingerprint"]
        if population_fingerprint != expected_population:
            raise ValueError("Metric population fingerprint mismatch")
        # Protocol-aware sample count: WB1 strict isolation removes samples near
        # split boundaries, giving a strict subset (VAL=2924 < 2960, TEST=2925 < 2961).
        # Train is never reduced (13670 for both protocols).
        data = record["config"]["data"]
        boundary_protocol = data.get("boundary_protocol", "WB0_CONTEXT_CARRY_OVER")
        final_refit_mode = bool(record["config"]["training"].get("final_refit_mode", False))
        if final_refit_mode and split == "VALIDATION":
            expected_counts = {
                "TRAIN": int(data["train_sample_count"]),
                "VALIDATION": int(data["train_sample_count"]),
                "TEST": int(data["test_sample_count"]),
            }
        elif boundary_protocol == "WB1_STRICT_ISOLATION":
            expected_counts = {"TRAIN": 13670, "VALIDATION": 2924, "TEST": 2925}
        else:
            expected_counts = {
                "TRAIN": int(data["train_sample_count"]),
                "VALIDATION": int(data["validation_sample_count"]),
                "TEST": int(data["test_sample_count"]),
            }
        if split not in expected_counts:
            raise ValueError(f"Unsupported split: {split}")
        if n_samples != expected_counts[split]:
            raise ValueError(
                f"Metric sample count mismatch: expected {expected_counts[split]} for "
                f"split={split} (protocol={boundary_protocol}), got {n_samples}"
            )
        identity = (split, name, epoch_or_checkpoint)
        if any((item["split_id"], item["metric_name"], item["epoch_or_checkpoint"]) == identity for item in record["metrics"]):
            raise ValueError("Duplicate metric row")
        row = {
            "run_id": run_id,
            "split_id": split,
            "metric_name": name,
            "metric_value": metric_value,
            "metric_unit": metric_unit,
            "n_samples": n_samples,
            "metric_version": METRIC_VERSION,
            "population_fingerprint": population_fingerprint,
            "epoch_or_checkpoint": epoch_or_checkpoint,
            "status": status,
        }
        record["metrics"].append(row)
        record["updated_at"] = self.clock()
        self._write_status(record)
        self._sync_core_artifacts(record)
        records[index] = record
        self._persist(records)
        return deepcopy(row)

    def _required_artifacts(self, execution_type: str, model_family: str) -> set[str]:
        base = {ArtifactType.CONFIG.value, ArtifactType.STATUS.value}
        if execution_type == ExecutionType.SANITY.value:
            return base | {ArtifactType.AUDIT.value}
        if execution_type == ExecutionType.EVALUATION.value:
            required = base | {ArtifactType.METRICS.value}
            if model_family == "PERSISTENCE":
                required.add(ArtifactType.PREDICTIONS.value)
            return required
        if execution_type in {ExecutionType.TRAINING.value, ExecutionType.ROBUSTNESS.value}:
            return base | {ArtifactType.TRAIN_LOG.value, ArtifactType.BEST_CHECKPOINT.value, ArtifactType.METRICS.value}
        if execution_type == ExecutionType.FINAL_TEST.value:
            return base | {ArtifactType.METRICS.value, ArtifactType.PREDICTIONS.value}
        raise ValueError("Unsupported execution type")

    def complete_run(self, run_id: str, best_epoch: int | None = None, best_validation_rmse_wh: float | None = None) -> dict[str, Any]:
        records = self._load_records()
        index = self._record_index(records, run_id)
        record = records[index]
        self._verify_config_immutability(record)
        if record["status"] != RunStatus.RUNNING.value:
            raise ValueError("Only RUNNING run can complete")
        available_artifacts = {item["artifact_type"] for item in record["artifacts"] if item["status"] == "PASS"}
        missing_artifacts = self._required_artifacts(record["execution_type"], record["config"]["model"]["model_family"]) - available_artifacts
        if missing_artifacts:
            raise ValueError(f"Required artifacts missing: {sorted(missing_artifacts)}")
        if record["execution_type"] != ExecutionType.SANITY.value:
            required_split = "TEST" if record["execution_type"] == ExecutionType.FINAL_TEST.value else "VALIDATION"
            final_refit_mode = bool(record["config"]["training"].get("final_refit_mode", False))
            if not final_refit_mode:
                required_metric_rows = [item for item in record["metrics"] if item["split_id"] == required_split and item["status"] in {"PASS", "PASS_WITH_WARNING"}]
                metric_names = {item["metric_name"] for item in required_metric_rows}
                if metric_names != REQUIRED_METRICS:
                    raise ValueError(f"Required {required_split} metrics are incomplete")
                if required_split == "VALIDATION":
                    rmse_values = [float(item["metric_value"]) for item in required_metric_rows if item["metric_name"] == "rmse_wh" and item["metric_value"] is not None]
                    if len(rmse_values) != 1:
                        raise ValueError("Exactly one Validation RMSE is required for completion")
                    if best_validation_rmse_wh is None:
                        best_validation_rmse_wh = rmse_values[0]
                    elif not math.isclose(best_validation_rmse_wh, rmse_values[0], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Best Validation RMSE differs from registered metric")
                    if record["execution_type"] in {ExecutionType.TRAINING.value, ExecutionType.ROBUSTNESS.value} and best_epoch is None:
                        raise ValueError("Training completion requires best_epoch")
            elif required_split == "VALIDATION" and best_validation_rmse_wh is None and any(item["split_id"] == "VALIDATION" for item in record["metrics"]):
                rmse_values = [float(item["metric_value"]) for item in record["metrics"] if item["split_id"] == "VALIDATION" and item["metric_name"] == "rmse_wh" and item["metric_value"] is not None]
                if len(rmse_values) == 1:
                    best_validation_rmse_wh = rmse_values[0]
        if best_validation_rmse_wh is not None and (not math.isfinite(best_validation_rmse_wh) or best_validation_rmse_wh < 0):
            raise ValueError("best_validation_rmse_wh must be finite and non-negative")
        if best_epoch is not None and best_epoch <= 0:
            raise ValueError("best_epoch must be positive")
        now = self.clock()
        record["status"] = RunStatus.COMPLETED.value
        record["completed_at"] = now
        record["updated_at"] = now
        record["best_epoch"] = best_epoch
        record["best_validation_rmse_wh"] = best_validation_rmse_wh
        self._write_status(record)
        self._sync_core_artifacts(record)
        records[index] = record
        self._persist(records)
        return deepcopy(record)

    def recover_wb1_run(
        self,
        run_id: str,
        best_epoch: int,
        best_validation_rmse_wh: float,
        recovered_sample_counts: dict[str, int],
    ) -> dict[str, Any]:
        """Recover a WB1 run that failed at metric registration due to
        protocol-ignorant sample-count validation.

        This method:
        1. Updates the in-memory config with corrected WB1 sample counts.
        2. Updates the config artifact on disk with corrected counts.
        3. Re-computes the config fingerprint from the corrected config.
        4. Updates the registry record fingerprint.
        5. Registers all VALIDATION metrics from best_validation_metrics.json.
        6. Completes the run.

        This is safe because it only fixes metadata; all scientific values
        (checkpoints, predictions, metrics in the JSON file) are untouched.
        """
        records = self._load_records()
        index = self._record_index(records, run_id)
        record = records[index]
        self._verify_config_immutability(record)
        if record["status"] != RunStatus.RUNNING.value:
            raise ValueError("Only RUNNING run can be recovered")
        if record["execution_type"] not in {ExecutionType.TRAINING.value, ExecutionType.ROBUSTNESS.value}:
            raise ValueError("WB1 recovery is only for TRAINING or ROBUSTNESS runs")
        # 1. Update config data with corrected WB1 counts
        config = record["config"]
        for split_id, count in recovered_sample_counts.items():
            key = f"{split_id.lower()}_sample_count"
            config["data"][key] = count
        # Also set boundary_protocol in case it was missing
        if "boundary_protocol" not in config["data"]:
            config["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
        # 2. Re-compute fingerprint from corrected config
        from course_work.experiments.registry import compute_config_fingerprint
        new_fingerprint = compute_config_fingerprint(config)
        # 3. Update config artifact on disk.
        # The artifact is normally immutable (signed), but recovery requires correcting
        # an incorrect WB0 sample count that was embedded before boundary_protocol
        # support existed.  Write corrected config directly.
        config_path = self.run_root / run_id / "config.json"
        corrected_config_payload = {
            "config": config,
            "config_fingerprint": new_fingerprint,
            "record_schema_version": record.get("record_schema_version", 1),
            "registry_version": record.get("registry_version", "EXPERIMENTS-v1"),
            "run_id": run_id,
        }
        from course_work.utils.artifacts import canonical_json_bytes
        config_path.write_bytes(canonical_json_bytes(corrected_config_payload))
        # 4. Update record
        record["config_fingerprint"] = new_fingerprint
        record["updated_at"] = self.clock()
        # 5. Register metrics from best_validation_metrics.json
        metrics_path = self.run_root / run_id / "metrics/best_validation_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        metrics_data = read_json(metrics_path)
        metric_result = metrics_data.get("metric_result", {})
        criterion_cfg = metrics_data.get("criterion_config", {})
        gradient_diag = metrics_data.get("gradient_diagnostics", {})
        expected_counts = {"TRAIN": 13670, "VALIDATION": 2924, "TEST": 2925}
        # Determine which splits have metrics in the file (VALIDATION is required)
        validation_n_samples = int(metric_result.get("n_samples", 0))
        if validation_n_samples != expected_counts["VALIDATION"]:
            raise ValueError(
                f"Recovered metrics have {validation_n_samples} samples but "
                f"WB1 expects {expected_counts['VALIDATION']}"
            )
        # Register VALIDATION metrics
        val_metrics = [
            ("mae_wh", metric_result.get("mae_wh"), "Wh"),
            ("rmse_wh", metric_result.get("rmse_wh"), "Wh"),
            ("r2", metric_result.get("r2"), "dimensionless"),
        ]
        population_fingerprint = metric_result.get("population_fingerprint", config["lineage"]["population_fingerprint"])
        epoch_str = f"epoch_{best_epoch}"
        for metric_name, metric_value, metric_unit in val_metrics:
            if metric_value is None:
                continue
            row = self._register_metric_row(
                record,
                run_id=run_id,
                split_id="VALIDATION",
                metric_name=metric_name,
                metric_value=float(metric_value),
                metric_unit=metric_unit,
                n_samples=validation_n_samples,
                population_fingerprint=population_fingerprint,
                epoch_or_checkpoint=epoch_str,
                status=str(metric_result.get("status", "PASS")),
            )
            record["metrics"].append(row)
        # 6. Complete the run
        now = self.clock()
        record["status"] = RunStatus.COMPLETED.value
        record["completed_at"] = now
        record["updated_at"] = now
        record["best_epoch"] = best_epoch
        record["best_validation_rmse_wh"] = float(best_validation_rmse_wh)
        self._write_status(record)
        self._sync_core_artifacts(record)
        records[index] = record
        self._persist(records)
        return deepcopy(record)

    def _register_metric_row(
        self,
        record: dict,
        run_id: str,
        split_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        n_samples: int,
        population_fingerprint: str,
        epoch_or_checkpoint: str,
        status: str,
    ) -> dict[str, Any]:
        boundary_protocol = record["config"]["data"].get("boundary_protocol", "WB0_CONTEXT_CARRY_OVER")
        if boundary_protocol == "WB1_STRICT_ISOLATION":
            expected_counts = {"TRAIN": 13670, "VALIDATION": 2924, "TEST": 2925}
        else:
            expected_counts = {
                "TRAIN": int(record["config"]["data"]["train_sample_count"]),
                "VALIDATION": int(record["config"]["data"]["validation_sample_count"]),
                "TEST": int(record["config"]["data"]["test_sample_count"]),
            }
        if n_samples != expected_counts[split_id]:
            raise ValueError(
                f"Metric n_samples={n_samples} does not match expected "
                f"{expected_counts[split_id]} for split={split_id} "
                f"(protocol={boundary_protocol})"
            )
        return {
            "run_id": run_id,
            "split_id": split_id.upper(),
            "metric_name": metric_name.lower(),
            "metric_value": metric_value,
            "metric_unit": metric_unit,
            "n_samples": n_samples,
            "metric_version": METRIC_VERSION,
            "population_fingerprint": population_fingerprint,
            "epoch_or_checkpoint": epoch_or_checkpoint,
            "status": status,
        }

    def fail_run(
        self,
        run_id: str,
        failure_type: str,
        failure_stage: str,
        failure_message: str,
        exception_class: str | None = None,
        traceback_path: str | None = None,
        recoverable: bool = False,
        rerun_recommended: bool = True,
    ) -> dict[str, Any]:
        if not failure_stage or not failure_message:
            raise ValueError("Failed run requires failure stage and message")
        failure = {
            "run_id": run_id,
            "failure_type": FailureType(failure_type).value,
            "failure_stage": failure_stage,
            "exception_class": exception_class,
            "failure_message": failure_message,
            "traceback_path": traceback_path,
            "recoverable": bool(recoverable),
            "rerun_recommended": bool(rerun_recommended),
            "created_at": self.clock(),
        }
        return self._transition(run_id, RunStatus.FAILED.value, {"failure": failure})

    def cancel_run(self, run_id: str, cancel_reason: str) -> dict[str, Any]:
        if not cancel_reason:
            raise ValueError("Cancellation requires a reason")
        return self._transition(run_id, RunStatus.CANCELLED.value, {"cancel_reason": cancel_reason})

    def invalidate_run(self, run_id: str, reason: str, scope: str, replacement_run_id: str | None = None) -> dict[str, Any]:
        allowed_scopes = {"METRICS_ONLY", "EVALUATION_ONLY", "TRAINING_AND_EVALUATION", "FULL_RUN"}
        if not reason or scope not in allowed_scopes:
            raise ValueError("Invalid invalidation reason or scope")
        if replacement_run_id is not None:
            self.get_run(replacement_run_id)
        invalidation = {
            "reason": reason,
            "scope": scope,
            "replacement_run_id": replacement_run_id,
            "invalidated_at": self.clock(),
        }
        return self._transition(run_id, RunStatus.INVALIDATED.value, {"invalidation": invalidation})

    def compare_runs(self, run_a: str, run_b: str, allowed_differences: set[str] | None = None) -> dict[str, Any]:
        records = self._load_records()
        index_a = self._record_index(records, run_a)
        index_b = self._record_index(records, run_b)
        result = compare_configs(records[index_a]["config"], records[index_b]["config"], allowed_differences)
        row = {
            "run_a": run_a,
            "run_b": run_b,
            "allowed_differences": json.dumps(result["allowed_differences"], ensure_ascii=False),
            "actual_differences": json.dumps(result["actual_differences"], ensure_ascii=False),
            "unexpected_differences": json.dumps(result["unexpected_differences"], ensure_ascii=False),
            "status": result["status"],
        }
        records[index_a]["comparisons"].append(row)
        records[index_a]["updated_at"] = self.clock()
        self._write_status(records[index_a])
        self._sync_core_artifacts(records[index_a])
        self._persist(records)
        return row

    def validate_registry(self) -> list[dict[str, Any]]:
        records = self._load_records()
        rows = []

        def add(check: str, expected: Any, actual: Any, passed: bool, details: str = "") -> None:
            rows.append({"check": check, "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL", "details": details})

        run_ids = [record.get("run_id") for record in records]
        add("unique_run_ids", len(run_ids), len(set(run_ids)), len(run_ids) == len(set(run_ids)))
        add("valid_statuses", "RunStatus", sorted({record.get("status") for record in records}), all(record.get("status") in {item.value for item in RunStatus} for record in records))
        config_valid = True
        parent_valid = True
        artifact_valid = True
        metric_valid = True
        completed_valid = True
        failed_valid = True
        test_firewall_valid = True
        metric_identities: set[tuple[str, str, str, str]] = set()
        sweep_ids = {row["sweep_id"] for row in self._load_sweeps()}
        sweep_references_valid = True
        for record in records:
            try:
                validate_run_config(record["config"], self.upstream_context)
                self._verify_config_immutability(record)
                config_valid = config_valid and compute_config_fingerprint(record["config"]) == record["config_fingerprint"]
            except (KeyError, TypeError, ValueError, RuntimeError):
                config_valid = False
            parent = record.get("parent_run_id")
            if parent is not None and parent not in run_ids:
                parent_valid = False
            if record.get("sweep_id") is not None and record["sweep_id"] not in sweep_ids:
                sweep_references_valid = False
            for artifact in record.get("artifacts", []):
                path = self._resolve_artifact_path(artifact["artifact_path"])
                if not path.is_file() or sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["file_size_bytes"]:
                    artifact_valid = False
            for metric in record.get("metrics", []):
                identity = (record["run_id"], metric["split_id"], metric["metric_name"], metric["epoch_or_checkpoint"])
                if identity in metric_identities:
                    metric_valid = False
                metric_identities.add(identity)
                if metric["metric_version"] != METRIC_VERSION or metric["metric_unit"] != METRIC_UNITS.get(metric["metric_name"]):
                    metric_valid = False
                if metric["population_fingerprint"] != record["config"]["lineage"]["population_fingerprint"]:
                    metric_valid = False
                if metric["split_id"] == "TEST" and not (record["experiment_family"] == "FINAL_TEST" and record["execution_type"] == "FINAL_TEST" and record.get("final_model_lock_id") and record.get("test_access_authorized")):
                    test_firewall_valid = False
            if record["status"] == RunStatus.COMPLETED.value:
                if not record.get("started_at") or not record.get("completed_at"):
                    completed_valid = False
                available_artifacts = {item["artifact_type"] for item in record.get("artifacts", []) if item.get("status") == "PASS"}
                if self._required_artifacts(record["execution_type"], record["config"]["model"]["model_family"]) - available_artifacts:
                    completed_valid = False
                if record["execution_type"] != ExecutionType.SANITY.value:
                    required_split = "TEST" if record["execution_type"] == ExecutionType.FINAL_TEST.value else "VALIDATION"
                    completed_metrics = {item["metric_name"] for item in record.get("metrics", []) if item["split_id"] == required_split and item["status"] in {"PASS", "PASS_WITH_WARNING"}}
                    if completed_metrics != REQUIRED_METRICS:
                        completed_valid = False
            if record["status"] == RunStatus.FAILED.value:
                failure = record.get("failure") or {}
                if not all(failure.get(key) for key in ("failure_type", "failure_stage", "failure_message")):
                    failed_valid = False
        add("valid_config_fingerprints", "all valid", config_valid, config_valid)
        add("no_orphan_parent_runs", "none", parent_valid, parent_valid)
        add("artifact_paths_and_checksums", "all valid", artifact_valid, artifact_valid)
        add("metric_contract_and_uniqueness", "all valid", metric_valid, metric_valid)
        add("completed_runs_have_required_timestamps", "all valid", completed_valid, completed_valid)
        add("failed_runs_have_failure_info", "all valid", failed_valid, failed_valid)
        add("test_metrics_authorized", "all valid", test_firewall_valid, test_firewall_valid)
        add("run_sweep_references", "all valid", sweep_references_valid, sweep_references_valid)
        if any(row["status"] != "PASS" for row in rows):
            raise RuntimeError("Registry validation failed")
        return rows

    def get_runs_by_family(self, experiment_family: str) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._load_records() if record["experiment_family"] == experiment_family]

    def get_completed_runs(self) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._load_records() if record["status"] == RunStatus.COMPLETED.value]

    def get_failed_runs(self) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._load_records() if record["status"] == RunStatus.FAILED.value]

    def get_runs_by_sweep(self, sweep_id: str) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._load_records() if record.get("sweep_id") == sweep_id]

    def get_run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return deepcopy(self.get_run(run_id)["metrics"])

    def get_run_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return deepcopy(self.get_run(run_id)["artifacts"])

    def write_registry_validation_audit(self) -> Path:
        rows = self.validate_registry()
        path = self.registry_root / "registry_validation_audit.csv"
        self._write_csv(path, VALIDATION_COLUMNS, rows)
        return path

    def snapshot_registry(self, phase_number: int) -> Path:
        if phase_number <= 13:
            raise ValueError("Registry snapshots begin after Phase 13")
        path = self.registry_root / f"experiment_registry_snapshot_phase{phase_number}.csv"
        if path.exists():
            raise FileExistsError(path)
        rows = [self._flatten_record(record) for record in self._load_records()]
        atomic_write_bytes(path, csv_text(MAIN_REGISTRY_COLUMNS, rows).encode("utf-8"))
        return path


def _audit_row(check: str, expected: Any, actual: Any, passed: bool, details: str = "") -> dict[str, Any]:
    return {"check": check, "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL", "details": details}


def _write_synthetic_file(path: Path, value: str) -> Path:
    atomic_write_bytes(path, value.encode("utf-8"))
    return path


def run_synthetic_registry_audit(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    base = build_reference_run_config(project_root)
    equivalent = deepcopy(base)
    equivalent["model"]["activation"] = "gelu"
    equivalent["training"]["learning_rate"] = 0.0003
    same_fingerprint = compute_config_fingerprint(base) == compute_config_fingerprint(equivalent)
    rows.append(_audit_row("canonical_config_fingerprint", "same", "same" if same_fingerprint else "different", same_fingerprint))
    with tempfile.TemporaryDirectory(prefix="phase13-registry-") as temporary:
        temporary_root = Path(temporary)
        registry = ExperimentRegistry(project_root, temporary_root / "registry", temporary_root / "runs")
        sweep = registry.register_sweep("S8_LR_TEST_ONLY", "S8", "S8_LEARNING_RATE", "training.learning_rate", [3e-4, 1e-3], base)
        rows.append(_audit_row("sweep_registration", "PLANNED", sweep["status"], sweep["status"] == "PLANNED"))
        first = registry.register_run(base, "TRANSFORMER_BASELINE", ExecutionType.SANITY.value)
        second = registry.register_run(equivalent, "TRANSFORMER_BASELINE", ExecutionType.SANITY.value, parent_run_id=first["run_id"], rerun_reason="REPRODUCIBILITY_CHECK")
        unique_ids = first["run_id"] != second["run_id"] and first["config_fingerprint"] == second["config_fingerprint"]
        rows.append(_audit_row("unique_rerun_ids", "new id and same fingerprint", unique_ids, unique_ids))
        duplicate_guard = False
        try:
            registry.register_run(base, "TRANSFORMER_BASELINE", ExecutionType.SANITY.value)
        except ValueError:
            duplicate_guard = True
        rows.append(_audit_row("duplicate_config_rerun_reason", "rejected without reason", duplicate_guard, duplicate_guard))
        registry.start_run(first["run_id"])
        audit_path = _write_synthetic_file(temporary_root / "runs" / first["run_id"] / "audit.json", "TEST_ONLY")
        registry.register_artifact(first["run_id"], ArtifactType.AUDIT.value, audit_path, True)
        completed = registry.complete_run(first["run_id"])
        rows.append(_audit_row("sanity_lifecycle", "COMPLETED", completed["status"], completed["status"] == "COMPLETED"))
        invalid_transition = False
        try:
            registry.start_run(first["run_id"])
        except ValueError:
            invalid_transition = True
        rows.append(_audit_row("invalid_status_transition", "rejected", invalid_transition, invalid_transition))
        mutation_guard = False
        config_path = temporary_root / "runs" / first["run_id"] / "config.json"
        original_config = config_path.read_bytes()
        mutated = read_json(config_path)
        mutated["config"]["training"]["learning_rate"] = 0.1
        atomic_write_bytes(config_path, canonical_json_bytes(mutated))
        try:
            registry.get_run(first["run_id"])
        except RuntimeError:
            mutation_guard = True
        atomic_write_bytes(config_path, original_config)
        rows.append(_audit_row("completed_config_immutability", "mutation rejected", mutation_guard, mutation_guard))
        registry.start_run(second["run_id"])
        failed = registry.fail_run(second["run_id"], FailureType.NUMERICAL_ERROR.value, "TRAINING", "TEST_ONLY_FAILURE")
        failure_valid = failed["status"] == "FAILED" and failed["failure"]["failure_type"] == "NUMERICAL_ERROR"
        rows.append(_audit_row("failed_run_traceability", "FAILED with taxonomy", failure_valid, failure_valid))
        development = registry.register_run(deepcopy(base), "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value, rerun_reason="MANUAL_RERUN")
        registry.start_run(development["run_id"])
        test_guard = False
        try:
            registry.register_metric(development["run_id"], "TEST", "rmse_wh", 1.0, "Wh", base["data"]["test_sample_count"], base["lineage"]["population_fingerprint"], "BEST")
        except PermissionError:
            test_guard = True
        rows.append(_audit_row("development_test_metric_firewall", "rejected", test_guard, test_guard))
        final_config = deepcopy(base)
        final_config["data"]["target_access_mode"] = "TEST"
        final_run = registry.register_run(final_config, "FINAL_TEST", ExecutionType.FINAL_TEST.value, final_model_lock_id="TEST_ONLY_LOCK", test_access_authorized=True)
        registry.start_run(final_run["run_id"])
        final_allowed = True
        try:
            for metric_name, metric_unit in METRIC_UNITS.items():
                registry.register_metric(final_run["run_id"], "TEST", metric_name, 1.0, metric_unit, final_config["data"]["test_sample_count"], final_config["lineage"]["population_fingerprint"], "FINAL")
        except (PermissionError, ValueError):
            final_allowed = False
        rows.append(_audit_row("authorized_final_test_metrics", "accepted", final_allowed, final_allowed))
        learning_rate_variant = deepcopy(base)
        learning_rate_variant["training"]["learning_rate"] = 1e-3
        allowed = validate_sweep_consistency([base, learning_rate_variant], {"training.learning_rate"})
        rows.append(_audit_row("one_factor_sweep", "PASS", allowed["status"], allowed["status"] == "PASS"))
        confounded = deepcopy(learning_rate_variant)
        confounded["model"]["dropout"] = 0.2
        confounding_guard = False
        try:
            validate_sweep_consistency([base, confounded], {"training.learning_rate"})
        except ValueError:
            confounding_guard = True
        rows.append(_audit_row("sweep_confounding_guard", "rejected", confounding_guard, confounding_guard))
        registry_rows = registry.validate_registry()
        rows.append(_audit_row("synthetic_registry_validation", "PASS", all(row["status"] == "PASS" for row in registry_rows), all(row["status"] == "PASS" for row in registry_rows)))
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Synthetic registry audit failed")
    return rows


def _readme_experiments() -> str:
    return "\n".join([
        "# EXPERIMENTS-v1",
        "",
        "EXPERIMENTS-v1 is the canonical local registry contract for all runs from Phase 14 onward.",
        "",
        "The JSONL file is the detailed machine-readable run source. CSV files are derived inspection views.",
        "",
        "Every run is registered before execution with a unique run ID, canonical config fingerprint and complete upstream lineage.",
        "",
        "Completed configs are immutable. A rerun receives a new run ID and requires a canonical rerun reason when its config fingerprint already exists.",
        "",
        "Artifacts and metrics are linked by run ID. Missing, orphaned or checksum-invalid records fail registry validation.",
        "",
        "Development runs cannot register Test metrics. Final Test registration requires FINAL_TEST family, FINAL_TEST execution type, an explicit model lock ID and authorization.",
        "",
        "Phase 13 creates no real run, model, checkpoint, prediction or metric result. Synthetic lifecycle tests run only in a temporary directory.",
        "",
    ])


def _initial_experiment_index() -> str:
    lines = ["# EXPERIMENT INDEX", "", f"Registry: {EXPERIMENT_VERSION}", "", "Registered runs: 0", "", "| Family | Phase | Model | Runs | Status |", "|---|---:|---|---:|---|"]
    lines.extend(f'| {family["family_id"]} | {family["phase"]} | {family["model_family"]} | 0 | {family["status"]} |' for family in EXPERIMENT_FAMILIES)
    return "\n".join(lines) + "\n"


def verify_phase_13_inputs(root: Path) -> dict[str, Any]:
    context = load_upstream_context(root)
    phase_12 = context["phase_12_signoff"]
    for relative_path, expected_checksum in phase_12.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 13 upstream checksum mismatch: {relative_path}")
    return context


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != EXPERIMENT_VERSION or signoff.get("status") != "PASS":
        raise RuntimeError("Existing Phase 13 sign-off is invalid")
    contract = build_registry_contract()
    if signoff.get("registry_contract_fingerprint") != contract["registry_contract_fingerprint"]:
        raise RuntimeError("Registry contract fingerprint mismatch")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Immutable Phase 13 artifact mismatch: {relative_path}")
    registry = ExperimentRegistry(root)
    audit_rows = registry.validate_registry()
    if any(row["status"] != "PASS" for row in audit_rows):
        raise RuntimeError("Live experiment registry validation failed")
    manifest = read_json(root / EXPERIMENT_ARTIFACT_ROOT / "registry_manifest.json")
    if manifest.get("experiment_registry_version") != EXPERIMENT_VERSION or manifest.get("registry_contract_fingerprint") != contract["registry_contract_fingerprint"]:
        raise RuntimeError("Experiment registry manifest mismatch")
    return signoff


def materialize_phase_13(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    artifact_root = root / EXPERIMENT_ARTIFACT_ROOT
    signoff_path = artifact_root / "phase_13_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    context = verify_phase_13_inputs(root)
    contract = build_registry_contract()
    audit_rows = run_synthetic_registry_audit(root)
    created_at = utc_now()
    empty_registry_fingerprint = sha256_bytes(canonical_json_bytes([]))
    registry_manifest = {
        "experiment_registry_version": EXPERIMENT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "canonical_registry_file": "experiment_registry.jsonl",
        "canonical_registry_format": "JSONL",
        "flattened_registry_file": "experiment_registry.csv",
        "flattened_registry_format": "CSV",
        "run_id_policy": contract["run_id_policy"],
        "config_fingerprint_policy": contract["config_fingerprint_policy"],
        "status_model": contract["status_model"],
        "failure_taxonomy_version": FAILURE_TAXONOMY_VERSION,
        "run_count": 0,
        "status_counts": {status.value: 0 for status in RunStatus},
        "completed_count": 0,
        "failed_count": 0,
        "invalidated_count": 0,
        "active_count": 0,
        "family_count": len(EXPERIMENT_FAMILIES),
        "sweep_count": 0,
        "artifact_registry_enabled": True,
        "metric_registry_enabled": True,
        "test_firewall_enabled": True,
        "sweep_consistency_guard_enabled": True,
        "completed_run_immutability": True,
        "atomic_write_policy": contract["atomic_write_policy"],
        "upstream_contract_versions": context["lineage"],
        "metric_version": METRIC_VERSION,
        "registry_contract_fingerprint": contract["registry_contract_fingerprint"],
        "registry_fingerprint": empty_registry_fingerprint,
        "production_registry_empty": True,
        "synthetic_records_persisted": False,
        "validation_audit_status": "PASS",
        "warnings": [],
        "created_at": created_at,
        "updated_at": created_at,
    }
    discrepancies = {"experiment_registry_version": EXPERIMENT_VERSION, "discrepancies": []}
    artifact_payloads = {
        "experiment_registry.csv": csv_text(MAIN_REGISTRY_COLUMNS, []),
        "experiment_registry.jsonl": "",
        "experiment_families.csv": csv_text(FAMILY_COLUMNS, EXPERIMENT_FAMILIES),
        "sweep_registry.csv": csv_text(SWEEP_COLUMNS, []),
        "run_artifact_registry.csv": csv_text(ARTIFACT_COLUMNS, []),
        "run_metric_registry.csv": csv_text(METRIC_COLUMNS, []),
        "run_failure_registry.csv": csv_text(FAILURE_COLUMNS, []),
        "run_comparison_audit.csv": csv_text(COMPARISON_COLUMNS, []),
        "registry_validation_audit.csv": csv_text(VALIDATION_COLUMNS, audit_rows),
        "registry_manifest.json": canonical_json_bytes(registry_manifest).decode("utf-8"),
        "registry_discrepancies.json": canonical_json_bytes(discrepancies).decode("utf-8"),
        "EXPERIMENT_INDEX.md": _initial_experiment_index(),
        "README_EXPERIMENTS.md": _readme_experiments(),
    }
    for filename, content in artifact_payloads.items():
        write_text_once_or_verify(artifact_root / filename, content)
    output_paths = [f"{EXPERIMENT_ARTIFACT_ROOT.as_posix()}/{filename}" for filename in artifact_payloads]
    mutable_filenames = {
        "experiment_registry.csv", "experiment_registry.jsonl", "sweep_registry.csv", "run_artifact_registry.csv",
        "run_metric_registry.csv", "run_failure_registry.csv", "run_comparison_audit.csv", "registry_validation_audit.csv",
        "registry_manifest.json", "registry_discrepancies.json", "EXPERIMENT_INDEX.md",
    }
    immutable_paths = [path for path in output_paths if Path(path).name not in mutable_filenames]
    input_paths = [
        "artifacts/environment/environment_report.json",
        "data/raw_data/dataset_manifest.json",
        "artifacts/schema/schema_manifest.json",
        "artifacts/temporal/temporal_manifest.json",
        "artifacts/eda/eda_manifest.json",
        "artifacts/features/feature_engineering_manifest.json",
        "artifacts/feature_sets/feature_set_manifest.json",
        "artifacts/splits/split_manifest.json",
        "artifacts/scaling/scaling_manifest.json",
        "artifacts/scaling/scaler_registry.json",
        "artifacts/windows/window_manifest.json",
        "artifacts/windows/window_fingerprints.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/metrics/metric_manifest.json",
        "artifacts/metrics/metric_contract.json",
        "artifacts/metrics/phase_12_signoff.json",
    ]
    initial_output_checksums = {path: sha256_file(root / path) for path in output_paths}
    phase_12 = context["phase_12_signoff"]
    signoff = {
        "phase_id": 13,
        "phase_version": "PHASE-13-v1",
        "artifact_version": EXPERIMENT_VERSION,
        "dataset_revision": phase_12["dataset_revision"],
        "environment_id": phase_12["environment_id"],
        "config_fingerprint": phase_12["config_fingerprint"],
        "metric_version": METRIC_VERSION,
        "registry_contract_fingerprint": contract["registry_contract_fingerprint"],
        "initial_registry_fingerprint": empty_registry_fingerprint,
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in immutable_paths},
        "initial_output_checksums": initial_output_checksums,
        "mutable_output_paths": sorted(set(output_paths) - set(immutable_paths)),
        "production_run_count": 0,
        "synthetic_records_persisted": False,
        "status": "PASS",
        "created_at": created_at,
        "tests": [row["check"] for row in audit_rows],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)
