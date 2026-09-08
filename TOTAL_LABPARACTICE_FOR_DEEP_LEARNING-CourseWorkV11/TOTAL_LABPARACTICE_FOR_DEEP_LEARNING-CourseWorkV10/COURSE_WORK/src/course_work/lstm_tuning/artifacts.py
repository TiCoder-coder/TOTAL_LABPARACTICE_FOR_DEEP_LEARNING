"""Producers for O43.1-O43.37 outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from course_work.lstm_tuning.shared_data_contract import SharedDataContract
from course_work.lstm_tuning.winners import StageWinner
from course_work.utils.artifacts import canonical_json_bytes, sha256_bytes, write_text_once_or_verify


@dataclass(frozen=True)
class StageResult:
    stage: str
    candidates: list[dict[str, Any]]
    winner: StageWinner
    reference_used_run_id: str | None
    reference_used_option: str
    applicability: str = "APPLICABLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "candidates": self.candidates,
            "winner": self.winner.to_dict(),
            "reference_used_run_id": self.reference_used_run_id,
            "reference_used_option": self.reference_used_option,
            "applicability": self.applicability,
        }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_once_or_verify(path, canonical_json_bytes(payload).decode("utf-8"))
    return path


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return path


def write_lstm_tuning_manifest(
    project_root: Path,
    contract: SharedDataContract,
    reference_resolution: dict[str, Any],
    max_fresh_runs: int,
    status: str = "PREPARED",
) -> Path:
    payload = {
        "version": "LSTM_TUNING-v1",
        "phase": 43,
        "model_family": "STANDARD_UNIDIRECTIONAL_LSTM",
        "reference_id": "LSTM_T0_REF",
        "source_phase42_signoff": "COURSE_WORK/artifacts/candidate_synthesis/phase_42_signoff.json",
        "source_phase43_handoff": contract.handoff_source,
        "stages": ["LT1", "LT2", "LT3", "LT4", "LT5"],
        "max_fresh_runs": max_fresh_runs,
        "selection_metric": "validation_rmse_wh",
        "selection_direction": "MIN",
        "seed": 42,
        "test_access": "forbidden",
        "sequential_tuning": True,
        "cartesian_search": False,
        "shared_data_contract_fingerprint": contract.feature_fingerprint,
        "reference_resolution": reference_resolution,
        "status": status,
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuning_manifest.json", payload)


def write_lstm_tuning_contract(project_root: Path) -> Path:
    payload = {
        "max_epochs": 50,
        "patience": 10,
        "min_delta": 0,
        "loss": "MSE",
        "optimizer": "AdamW",
        "gradient_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup": None,
        "accumulation": 1,
        "amp": False,
        "seed": 42,
        "stages": ["LT1", "LT2", "LT3", "LT4", "LT5"],
        "unidirectional_lstm_only": True,
        "bidirectional": False,
        "proj_size": 0,
        "revin_enabled": False,
        "status": "FROZEN",
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuning_contract.json", payload)


def write_lstm_shared_data_contract(project_root: Path, contract: SharedDataContract) -> Path:
    payload = {
        "forecast_task": contract.forecast_task,
        "target": contract.target,
        "horizon": contract.forecast_horizon,
        "feature_variant_id": contract.feature_variant_id,
        "feature_count": contract.feature_count,
        "feature_names": contract.feature_names,
        "feature_fingerprint": contract.feature_fingerprint,
        "target_scaling_id": contract.target_scaling_id,
        "target_scaler_bundle_id": contract.target_scaler_bundle_id,
        "target_scaler_checksum": contract.target_scaler_checksum,
        "lookback_id": contract.lookback_id,
        "lookback_steps": contract.lookback_steps,
        "boundary_protocol": contract.boundary_protocol,
        "window_population_version": contract.window_population_version,
        "population_fingerprint": contract.population_fingerprint,
        "train_target_ids_fingerprint": contract.train_target_ids_fingerprint,
        "validation_target_ids_fingerprint": contract.validation_target_ids_fingerprint,
        "x_scaler_bundle_id": contract.x_scaler_bundle_id,
        "x_scaler_checksum": contract.x_scaler_checksum,
        "split_version": contract.split_version,
        "metric_version": contract.metric_version,
        "batch_size": contract.batch_size,
        "train_sample_count": contract.train_sample_count,
        "validation_sample_count": contract.validation_sample_count,
        "test_sample_count": contract.test_sample_count,
        "test_locked": contract.test_locked,
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_shared_data_contract.json", payload)


def write_reference_resolution(project_root: Path, resolution: dict[str, Any]) -> Path:
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_reference_resolution.json", resolution)


def write_lstm_tuning_space(project_root: Path) -> Path:
    payload = {
        "LT1": {"hidden_size": [32, 64, 128]},
        "LT2": {"num_layers": [1, 2]},
        "LT3": {"dropout": [0.0, 0.1, 0.2], "applicable_if_num_layers_gte_2": True},
        "LT4": {"learning_rate": [1e-4, 3e-4, 1e-3]},
        "LT5": {"weight_decay": [0.0, 1e-4, 1e-3]},
        "fixed": {
            "loss": "MSE",
            "max_epochs": 50,
            "patience": 10,
            "gradient_clip_max_norm": 1.0,
            "scheduler": None,
            "warmup": None,
            "seed": 42,
        },
        "max_fresh_runs": 10,
        "selection_metric": "validation_rmse_wh",
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuning_space.json", payload)


def _stage_metrics_rows(stage: str, stage_result: StageResult) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for cand in stage_result.candidates:
        rows.append([
            cand["option"],
            cand.get("run_id") or "",
            f"{cand.get('value', '')}",
            cand.get("source_type", "FRESH"),
            cand.get("epochs_completed") or "",
            cand.get("stop_reason") or "",
            cand.get("best_epoch") or "",
            cand.get("trainable_parameters") or "",
            f"{cand.get('validation_rmse_wh', ''):.6f}" if cand.get("validation_rmse_wh") is not None else "",
            f"{cand.get('validation_mae_wh', ''):.6f}" if cand.get("validation_mae_wh") is not None else "",
            f"{cand.get('validation_r2', ''):.4f}" if cand.get("validation_r2") is not None else "",
            cand.get("rmse_rank") or "",
            "WINNER" if cand["option"] == stage_result.winner.option else ("FAILED" if cand.get("status") == "FAILED" else "OK"),
        ])
    return rows


STAGE_METRICS_HEADER = [
    "candidate_id",
    "run_id",
    "candidate_value",
    "source_type",
    "epochs_completed",
    "stop_reason",
    "best_epoch",
    "trainable_parameters",
    "validation_rmse_wh",
    "validation_mae_wh",
    "validation_r2",
    "rmse_rank",
    "is_winner",
]


def write_stage_metrics(project_root: Path, stage_result: StageResult) -> Path:
    stage = stage_result.stage.lower()
    name_by_stage = {
        "lt1": "lt1_hidden_size_metrics.csv",
        "lt2": "lt2_layers_metrics.csv",
        "lt3": "lt3_dropout_metrics.csv",
        "lt4": "lt4_learning_rate_metrics.csv",
        "lt5": "lt5_weight_decay_metrics.csv",
    }
    path = project_root / "artifacts/lstm_tuning" / name_by_stage[stage]
    rows = _stage_metrics_rows(stage, stage_result)
    return _write_csv(path, STAGE_METRICS_HEADER, rows)


def write_stage_winner(project_root: Path, stage_result: StageResult) -> Path:
    stage = stage_result.stage.lower()
    name_by_stage = {
        "lt1": "lt1_hidden_size_winner.json",
        "lt2": "lt2_layers_winner.json",
        "lt3": "lt3_dropout_winner.json",
        "lt4": "lt4_learning_rate_winner.json",
        "lt5": "lt5_weight_decay_winner.json",
    }
    payload = {
        "stage": stage_result.stage,
        "winner_option": stage_result.winner.option,
        "winner_run_id": stage_result.winner.run_id,
        "winner_value": stage_result.winner.value,
        "winner_rmse_wh": stage_result.winner.validation_rmse_wh,
        "winner_mae_wh": stage_result.winner.validation_mae_wh,
        "winner_r2": stage_result.winner.validation_r2,
        "best_epoch": stage_result.winner.best_epoch,
        "epochs_completed": stage_result.winner.epochs_completed,
        "stop_reason": stage_result.winner.stop_reason,
        "trainable_parameters": stage_result.winner.trainable_parameters,
        "exact_tie": stage_result.winner.exact_tie,
        "tie_rule_applied": stage_result.winner.tie_rule_applied,
        "reference_used_run_id": stage_result.reference_used_run_id,
        "reference_used_option": stage_result.reference_used_option,
        "applicability": stage_result.applicability,
    }
    return _write_json(project_root / "artifacts/lstm_tuning" / name_by_stage[stage], payload)


def write_lt3_applicability(project_root: Path, num_layers: int, applicable: bool, status_label: str | None = None) -> Path:
    payload = {
        "applicable": bool(applicable),
        "num_layers": int(num_layers),
        "selected_effective_dropout": 0.0 if applicable else 0.0,
        "stage_status": status_label or ("APPLICABLE" if applicable else "SKIPPED_NOT_APPLICABLE"),
        "reason": None if applicable else "NO_INTER_LAYER_DROPOUT_SITE",
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lt3_dropout_applicability.json", payload)


def write_run_matrix(project_root: Path, rows: list[list[Any]]) -> Path:
    header = [
        "run_id",
        "stage_id",
        "candidate_id",
        "source_type",
        "feature_variant_id",
        "target_scaling_id",
        "lookback_id",
        "boundary_protocol",
        "batch_size",
        "hidden_size",
        "num_layers",
        "dropout_arg",
        "learning_rate",
        "weight_decay",
        "loss",
        "max_epochs",
        "patience",
        "gradient_clip",
        "seed",
        "validation_rmse_wh",
        "validation_mae_wh",
        "validation_r2",
        "trainable_parameters",
        "status",
    ]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_run_matrix.csv", header, rows)


def write_stage_lineage(project_root: Path, lineage_rows: list[list[Any]]) -> Path:
    header = [
        "stage",
        "parameter",
        "reference_run_id",
        "reference_option",
        "candidate_run_ids",
        "winner_option",
        "winner_run_id",
        "winner_value",
        "winner_rmse_wh",
        "next_stage_reference",
        "status",
    ]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_stage_lineage.csv", header, lineage_rows)


def write_architecture_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = [
        "run_id",
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout_arg",
        "effective_inter_layer_dropout",
        "bidirectional",
        "proj_size",
        "batch_first",
        "pooling_semantics",
        "head_in_features",
        "head_out_features",
        "output_shape_verified",
        "parameter_count",
        "status",
    ]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_architecture_audit.csv", header, rows)


def write_training_config_delta_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["stage", "candidate_id", "field", "reference_value", "candidate_value", "allowed_to_differ", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_training_config_delta_audit.csv", header, rows)


def write_initialization_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["stage", "candidate_id", "seed", "shape_compatible_with_reference", "initial_state_fingerprint", "reference_fingerprint_available", "match", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_initialization_audit.csv", header, rows)


def write_sample_order_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["stage", "epoch_or_probe", "candidate_id", "sample_order_fingerprint", "reference_order_fingerprint", "same_order", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_sample_order_audit.csv", header, rows)


def write_optimizer_group_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "group_id", "parameter_count", "parameter_names_fingerprint", "learning_rate", "weight_decay", "policy_id", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_optimizer_group_audit.csv", header, rows)


def write_optimizer_budget_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "train_samples_per_epoch", "batch_size", "steps_per_epoch", "epochs_completed", "total_optimizer_steps", "best_epoch", "steps_to_best", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_optimizer_budget_audit.csv", header, rows)


def write_gradient_diagnostics(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "epoch", "mean_preclip_grad_norm", "max_preclip_grad_norm", "clipped_batches", "total_batches", "clipping_fraction", "nonfinite_events", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_gradient_diagnostics.csv", header, rows)


def write_convergence_diagnostics(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "stage", "first_epoch_rmse", "best_epoch", "best_rmse", "last_epoch", "last_rmse", "early_stopped", "cap_reached", "epochs_after_best", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_convergence_diagnostics.csv", header, rows)


def write_runtime_diagnostics(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "device", "parameter_count", "epochs_completed", "total_runtime_seconds", "mean_epoch_seconds", "median_epoch_seconds", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_runtime_diagnostics.csv", header, rows)


def write_run_provenance(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["run_id", "stage", "candidate_id", "source_reference_run_id", "config_fingerprint", "data_fingerprint", "best_checksum", "metric_artifact", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_run_provenance.csv", header, rows)


def write_tuning_effect(project_root: Path, rows: list[list[Any]]) -> Path:
    header = [
        "reference_run_id",
        "tuned_run_id",
        "reference_rmse_wh",
        "tuned_rmse_wh",
        "rmse_improvement_wh",
        "rmse_improvement_pct",
        "reference_mae_wh",
        "tuned_mae_wh",
        "mae_improvement_wh",
        "reference_r2",
        "tuned_r2",
        "r2_delta",
        "reference_params",
        "tuned_params",
        "status",
    ]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_tuning_effect.csv", header, rows)


def write_contextual_baseline_comparison(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["model", "run_id", "role", "sample_count", "mae_wh", "rmse_wh", "r2", "population_fingerprint", "comparable", "notes"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_contextual_baseline_comparison.csv", header, rows)


def write_reference_audit(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["check", "expected", "actual", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_reference_audit.csv", header, rows)


def write_tuned_winner(
    project_root: Path,
    contract: SharedDataContract,
    tuned_config: dict[str, Any],
    run_id: str | None,
    config_fingerprint: str | None,
    validation_rmse_wh: float | None,
    validation_mae_wh: float | None,
    validation_r2: float | None,
    best_epoch: int | None,
    parameter_count: int | None,
    population_fingerprint: str | None,
) -> Path:
    payload = {
        "version": "LSTM_TUNING-v1",
        "model_family": "LSTM",
        "winner_id": "LSTM_TUNED",
        "winner_run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "feature_variant_id": contract.feature_variant_id,
        "target_scaling_id": contract.target_scaling_id,
        "lookback_id": contract.lookback_id,
        "lookback_steps": contract.lookback_steps,
        "boundary_protocol": contract.boundary_protocol,
        "batch_size": contract.batch_size,
        "hidden_size": tuned_config.get("model", {}).get("hidden_size"),
        "num_layers": tuned_config.get("model", {}).get("num_layers"),
        "dropout_arg": tuned_config.get("model", {}).get("dropout"),
        "learning_rate": tuned_config.get("training", {}).get("learning_rate"),
        "weight_decay": tuned_config.get("training", {}).get("weight_decay"),
        "loss": tuned_config.get("training", {}).get("loss_name", "MSE"),
        "max_epochs": tuned_config.get("training", {}).get("max_epochs"),
        "patience": tuned_config.get("training", {}).get("early_stopping_patience", tuned_config.get("training", {}).get("patience")),
        "gradient_clip": tuned_config.get("training", {}).get("gradient_clip_max_norm"),
        "bidirectional": tuned_config.get("model", {}).get("bidirectional", False),
        "proj_size": tuned_config.get("model", {}).get("proj_size", 0),
        "pooling": tuned_config.get("model", {}).get("pooling", "LAST_SEQUENCE_OUTPUT"),
        "seed": tuned_config.get("reproducibility", {}).get("seed", 42),
        "validation_mae_wh": validation_mae_wh,
        "validation_rmse_wh": validation_rmse_wh,
        "validation_r2": validation_r2,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "population_fingerprint": population_fingerprint or contract.population_fingerprint,
        "metric_version": contract.metric_version,
        "test_status": "NOT_ACCESSED",
        "status": "CANDIDATE_FOR_HUMAN_TRAINING",
    }
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuned_winner.json", payload)


def write_phase44_lstm_handoff(
    project_root: Path,
    contract: SharedDataContract,
    tuned_winner_path: Path,
) -> Path:
    payload = {
        "lstm_tuning_version": "LSTM_TUNING-v1",
        "winner_run_id": None,
        "winner_config": None,
        "winner_config_fingerprint": None,
        "shared_feature_variant": contract.feature_variant_id,
        "shared_target_scaling": contract.target_scaling_id,
        "shared_lookback": contract.lookback_steps,
        "boundary_protocol": contract.boundary_protocol,
        "window_population_policy": contract.window_population_version,
        "loss": "MSE",
        "max_epochs": 50,
        "patience": 10,
        "gradient_clip": 1.0,
        "metric_version": contract.metric_version,
        "seed_policy_for_fold_training": 42,
        "test_locked": True,
        "source_phase42_transformer_shortlist_fingerprint": contract.handoff_phase42_shortlist_fingerprint,
        "transformer_primary_candidate_id": contract.transformer_primary_candidate_id,
        "transformer_primary_run_id": contract.transformer_primary_run_id,
        "ready_for_phase44": False,
        "phase43_signoff_status": "PREPARED",
        "winner_artifact": str(tuned_winner_path.relative_to(project_root)),
    }
    return _write_json(project_root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json", payload)


def write_findings(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["finding_code", "description"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_tuning_findings.csv", header, rows)


def write_tests(project_root: Path, rows: list[list[Any]]) -> Path:
    header = ["test_case", "result", "status"]
    return _write_csv(project_root / "artifacts/lstm_tuning/lstm_tuning_tests.csv", header, rows)


def write_discrepancies(project_root: Path, discrepancies: list[dict[str, Any]]) -> Path:
    payload = {"discrepancies": discrepancies}
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuning_discrepancies.json", payload)


def write_summary(project_root: Path, payload: dict[str, Any]) -> Path:
    return _write_json(project_root / "artifacts/lstm_tuning/lstm_tuning_summary.json", payload)


def write_report(project_root: Path, content: str) -> Path:
    path = project_root / "artifacts/lstm_tuning/lstm_tuning_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_once_or_verify(path, content)
    return path


def write_readme(project_root: Path, content: str) -> Path:
    path = project_root / "artifacts/lstm_tuning/README_LSTM_TUNING.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_once_or_verify(path, content)
    return path


def write_phase43_signoff(
    project_root: Path,
    payload: dict[str, Any],
) -> Path:
    payload.setdefault("phase", 43)
    payload.setdefault("phase_id", 43)
    payload.setdefault("phase_name", "LSTM tuning")
    payload.setdefault("version", "LSTM_TUNING-v1")
    payload.setdefault("artifact_version", "LSTM_TUNING-v1")
    payload.setdefault("phase_version", "PHASE-43-v1")
    return _write_json(project_root / "artifacts/lstm_tuning/phase_43_signoff.json", payload)
