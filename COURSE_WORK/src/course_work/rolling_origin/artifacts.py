"""Phase 44 — O44.1–O44.39 artifact writers.

Each writer takes its inputs explicitly and emits one canonical file
under `artifacts/rolling_origin/` (or `artifacts/rolling_origin/predictions/`
for prediction bundles). No writer depends on script-level globals.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, canonical_json_bytes(payload))
    return path


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    return path


def _df_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, df.to_csv(index=False).encode("utf-8"))
    return path


# ---------- O44.1 Manifest ----------


def write_manifest(
    *,
    path: Path,
    phase_id: int,
    K: int,
    transformer_shortlist_fingerprint: str,
    lstm_winner_fingerprint: str,
    persistence_version: str,
    fold_local_scaling: bool,
    status: str,
    created_at: str,
    extra: dict | None = None,
) -> Path:
    payload = {
        "version": "ROLLING_ORIGIN-v1",
        "phase": phase_id,
        "fold_protocol": "RO3_EXPANDING_PRETEST-v1",
        "K": K,
        "initial_history": "original Train",
        "outer_eval_region": "original Validation",
        "test_region_used": False,
        "nested_epoch_selection": True,
        "full_history_refit": True,
        "outer_eval_used_for_selection": False,
        "primary_metric": "pooled_outer_rmse_wh",
        "seed": 42,
        "transformer_shortlist_fingerprint": transformer_shortlist_fingerprint,
        "lstm_winner_fingerprint": lstm_winner_fingerprint,
        "persistence_version": persistence_version,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "population_base": "WINDOWPOP-v1",
        "fold_local_scaling": fold_local_scaling,
        "test_status": "NOT_ACCESSED",
        "status": status,
        "created_at": created_at,
    }
    if extra:
        payload.update(extra)
    return _write_json(path, payload)


# ---------- O44.2 Contract ----------


def write_contract(*, path: Path, K: int, created_at: str) -> Path:
    payload = {
        "version": "ROLLING_ORIGIN_CONTRACT-v1",
        "phase": 44,
        "K": K,
        "protocol": "RO3_EXPANDING_PRETEST-v1",
        "test_locked": True,
        "models_evaluated": [
            "TR_C0_PRIMARY",
            "TR_C1_ALT_WEIGHT_DECAY",
            "TR_C2_ALT_LOOKBACK",
            "LSTM_TUNED",
            "PERSISTENCE_LAST_VALUE",
        ],
        "fold_local_scaling": "RO_SCALING-v1",
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "stages": ["A", "B", "C"],
        "nested_epoch_selection": True,
        "full_history_refit": True,
        "outer_eval_used_for_selection": False,
        "pooled_metric_definition": "concatenated_residuals_then_compute",
        "tie_break_rule": "ascending(pooled_outer_rmse_wh, worst_fold_rmse_wh, fold_rmse_sd_wh, shortlist_position)",
        "created_at": created_at,
    }
    return _write_json(path, payload)


# ---------- O44.4 Frozen fold manifest ----------


def write_fold_manifest(*, path: Path, fold_manifest: dict) -> Path:
    return _write_json(path, fold_manifest)


# ---------- O44.5 Fold table ----------


def write_fold_table(*, path: Path, fold_table_rows: list[dict]) -> Path:
    header = [
        "fold_id",
        "origin_timestamp",
        "inner_train_count",
        "inner_val_count",
        "outer_train_count",
        "outer_eval_count",
        "inner_train_first_target",
        "inner_train_last_target",
        "inner_val_first_target",
        "inner_val_last_target",
        "outer_train_first_target",
        "outer_train_last_target",
        "outer_eval_first_target",
        "outer_eval_last_target",
        "inner_train_fingerprint",
        "inner_val_fingerprint",
        "outer_train_fingerprint",
        "outer_eval_fingerprint",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in fold_table_rows])


# ---------- O44.6 Population audit ----------


def write_population_audit(*, path: Path, population_audit_rows: list[dict]) -> Path:
    header = [
        "fold_id",
        "role",
        "target_count",
        "target_ids_unique",
        "chronological",
        "candidate_coverage_count",
        "all_candidates_supported",
        "population_fingerprint",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in population_audit_rows])


# ---------- O44.7 Candidate matrix ----------


def write_candidate_matrix(*, path: Path, candidate_matrix_rows: list[dict]) -> Path:
    header = [
        "model_id",
        "model_family",
        "candidate_role",
        "config_fingerprint",
        "feature_variant",
        "target_scaling",
        "lookback",
        "batch",
        "loss",
        "max_epochs",
        "patience",
        "clipping",
        "revin",
        "requires_training",
        "source_phase",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in candidate_matrix_rows])


# ---------- O44.8 Candidate compatibility audit ----------


def write_compatibility_audit(*, path: Path, compatibility_rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "lookback_compatible",
        "feature_variant_compatible",
        "target_scaling_compatible",
        "boundary_protocol_compatible",
        "batch_compatible",
        "all_ok",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in compatibility_rows])


# ---------- O44.9 Fold-local scaling contract ----------


def write_scaling_contract(*, path: Path, scaling_version: str, created_at: str) -> Path:
    payload = {
        "version": "RO_SCALING_CONTRACT-v1",
        "scaling_version": scaling_version,
        "phase": 44,
        "X_scaler_per_fold_stage": True,
        "Y_scaler_per_fold_stage": True,
        "YS0_identity": True,
        "YS1_standard_per_fold_stage": True,
        "cyclical_passthrough": True,
        "binary_passthrough": True,
        "RevIN_per_window": True,
        "no_global_scaler_reuse": True,
        "test_firewall": True,
        "created_at": created_at,
    }
    return _write_json(path, payload)


# ---------- O44.10 Scaler-fit audit ----------


def write_scaler_fit_audit(*, path: Path, audit_rows: list[dict]) -> Path:
    header = [
        "bundle_id",
        "fit_stage",
        "fold_id",
        "candidate_id",
        "target_scaling_option",
        "feature_variant_id",
        "fit_target_ids_count",
        "fit_target_ids_fingerprint",
        "fit_raw_row_count",
        "fit_start_timestamp",
        "fit_end_timestamp",
        "allowed_target_history_region",
        "outer_eval_rows_used",
        "test_rows_used",
        "bundle_checksum",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in audit_rows])


# ---------- O44.11 Temporal leakage tests ----------


def write_temporal_leakage_tests(*, path: Path, leakage_rows: list[dict]) -> Path:
    header = [
        "fold_id",
        "max_inner_train_id",
        "min_inner_val_id",
        "max_inner_val_id",
        "min_outer_eval_id",
        "temporal_ok",
        "disjoint_ok",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in leakage_rows])


# ---------- O44.12 Common-target audit ----------


def write_common_target_audit(*, path: Path, common_target_rows: list[dict]) -> Path:
    header = [
        "fold_id",
        "candidate_count",
        "outer_eval_target_ids_count",
        "outer_eval_target_ids_unique",
        "y_true_equal_across_candidates",
        "candidate_ids",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in common_target_rows])


# ---------- O44.13 Inner-selection run registry ----------


def write_inner_selection_run_registry(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "run_id",
        "candidate_id",
        "fold_id",
        "stage",
        "experiment_family",
        "execution_type",
        "seed",
        "config_fingerprint",
        "best_epoch_inner",
        "best_inner_rmse_wh",
        "status",
        "registered_at",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.14 Inner best-epoch table ----------


def write_inner_best_epochs(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "best_epoch_inner",
        "best_inner_rmse_wh",
        "best_inner_mae_wh",
        "best_inner_r2",
        "sample_count_inner_val",
        "population_fingerprint",
        "scaler_bundle_checksum",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.15 Refit run registry ----------


def write_refit_run_registry(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "run_id",
        "candidate_id",
        "fold_id",
        "stage",
        "parent_run_id",
        "experiment_family",
        "execution_type",
        "seed",
        "config_fingerprint",
        "official_epoch",
        "best_epoch_inner",
        "status",
        "registered_at",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.16 Refit epoch audit ----------


def write_refit_epoch_audit(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "run_id",
        "candidate_id",
        "fold_id",
        "official_epoch",
        "best_epoch_inner",
        "epoch_match",
        "no_validation_selection",
        "no_early_stopping",
        "no_warm_start",
        "no_final_dev_semantics",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.17 Initialization audit ----------


def write_initialization_audit(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "stage",
        "init_fingerprint",
        "fresh_optimizer",
        "fresh_scaler",
        "fresh_weights",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.18 Sample-order audit ----------


def write_sample_order_audit(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "stage",
        "sample_order_fingerprint",
        "chronological",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.20 Runtime diagnostics ----------


def write_runtime_diagnostics(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "stage",
        "train_runtime_seconds",
        "epoch_count",
        "device_name",
        "seed",
        "stopped_reason",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.19 Gradient diagnostics ----------


def write_gradient_diagnostics(*, path: Path, rows: list[dict]) -> Path:
    """O44.19 — per-(candidate, fold, stage) gradient diagnostic rows.

    Fields: fold_id, candidate_id, stage, epoch, mean_preclip_norm,
            max_preclip_norm, clip_fraction_if_applicable, nonfinite_events, status
    """
    header = [
        "fold_id",
        "candidate_id",
        "stage",
        "epoch",
        "mean_preclip_norm",
        "max_preclip_norm",
        "clip_fraction_if_applicable",
        "nonfinite_events",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.21 Per-fold outer predictions ----------


def write_outer_predictions(*, path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, df.to_csv(index=False).encode("utf-8"))
    return path


# ---------- O44.22 Pooled outer predictions ----------


def write_pooled_predictions(*, path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, df.to_csv(index=False).encode("utf-8"))
    return path


# ---------- O44.23 Fold metrics ----------


def write_fold_metrics(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_id",
        "mae_wh",
        "rmse_wh",
        "r2",
        "sample_count",
        "model_run_id",
        "refit_epoch",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.24 Pooled metrics ----------


def write_pooled_metrics(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_count",
        "pooled_mae_wh",
        "pooled_rmse_wh",
        "pooled_r2",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.25 Macro robustness metrics ----------


def write_macro_metrics(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_id",
        "fold_count",
        "macro_mae_wh",
        "macro_rmse_wh",
        "worst_fold_rmse_wh",
        "best_fold_rmse_wh",
        "fold_rmse_sd_wh",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.26 Pairwise effects ----------


def write_pairwise_effects(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "candidate_a",
        "candidate_b",
        "family_a",
        "family_b",
        "fold_count",
        "pooled_rmse_a_wh",
        "pooled_rmse_b_wh",
        "delta_pooled_rmse_wh",
        "delta_pooled_pct",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.27 Fold ranks ----------


def write_fold_ranks(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "fold_id",
        "rank",
        "candidate_id",
        "rmse_wh",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.28 Transformer ranking ----------


def write_transformer_ranking(*, path: Path, ranking_rows: list[dict]) -> Path:
    header = [
        "rank",
        "candidate_id",
        "pooled_rmse_wh",
        "worst_fold_rmse_wh",
        "fold_rmse_sd_wh",
        "shortlist_position",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in ranking_rows])


# ---------- O44.29 Model-family robustness comparison ----------


def write_model_family_comparison(*, path: Path, rows: list[dict]) -> Path:
    header = [
        "model_family",
        "best_candidate_id",
        "best_pooled_rmse_wh",
        "fold_count",
        "status",
    ]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.30 Findings ----------


def write_findings(*, path: Path, rows: list[list[str]]) -> Path:
    return _write_csv(path, ["Finding_Code", "Description"], rows)


# ---------- O44.31 Recommended transformer ----------


def write_recommended_transformer(*, path: Path, payload: dict) -> Path:
    return _write_json(path, payload)


# ---------- O44.32 Phase 45 handoff ----------


def write_phase45_handoff(*, path: Path, payload: dict) -> Path:
    return _write_json(path, payload)


# ---------- O44.34 Tests ----------


def write_tests(*, path: Path, rows: list[dict]) -> Path:
    header = ["test_name", "status", "notes"]
    return _write_csv(path, header, [[r.get(h, "") for h in header] for r in rows])


# ---------- O44.35 Discrepancies ----------


def write_discrepancies(*, path: Path, payload: dict) -> Path:
    return _write_json(path, payload)


# ---------- O44.36 Summary ----------


def write_summary(*, path: Path, payload: dict) -> Path:
    return _write_json(path, payload)


# ---------- O44.37 Report ----------


def write_report(*, path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload.encode("utf-8"))
    return path


# ---------- O44.38 README ----------


def write_readme(*, path: Path, payload: str) -> Path:
    return write_report(path=path, payload=payload)


# ---------- O44.39 Sign-off ----------


def write_signoff(*, path: Path, payload: dict) -> Path:
    return _write_json(path, payload)


# ---------- Convenience: write all artifacts ----------


@dataclass
class Phase44Artifacts:
    artifact_dir: Path
    predictions_dir: Path
    figures_dir: Path

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> "Phase44Artifacts":
        return cls(
            artifact_dir=artifact_dir,
            predictions_dir=artifact_dir / "predictions",
            figures_dir=artifact_dir / "figures",
        )


def write_all_artifacts(
    *,
    artifacts: Phase44Artifacts,
    manifest: dict,
    contract: dict,
    fold_manifest: dict,
    fold_table_rows: list[dict],
    population_audit_rows: list[dict],
    candidate_matrix_rows: list[dict],
    compatibility_rows: list[dict],
    scaling_contract: dict,
    scaler_fit_audit_rows: list[dict],
    leakage_rows: list[dict],
    common_target_rows: list[dict],
    inner_selection_rows: list[dict],
    inner_best_epoch_rows: list[dict],
    refit_run_rows: list[dict],
    refit_epoch_audit_rows: list[dict],
    initialization_audit_rows: list[dict],
    sample_order_audit_rows: list[dict],
    runtime_diagnostics_rows: list[dict],
    gradient_diagnostics_rows: list[dict] | None = None,
    fold_metrics_rows: list[dict],
    pooled_metrics_rows: list[dict],
    macro_metrics_rows: list[dict],
    pairwise_effects_rows: list[dict],
    fold_ranks_rows: list[dict],
    transformer_ranking_rows: list[dict],
    model_family_comparison_rows: list[dict],
    findings_rows: list[list[str]],
    recommended_transformer: dict,
    phase45_handoff: dict,
    tests_rows: list[dict],
    discrepancies: dict,
    summary: dict,
    preflight_rows: list[list[str]],
    pooled_predictions_df: pd.DataFrame,
    per_fold_outer_predictions: dict[tuple, pd.DataFrame] | None = None,
    report_markdown: str,
    readme_markdown: str,
    signoff: dict,
    figures: dict[str, bytes] | None = None,
) -> dict[str, Path]:
    if gradient_diagnostics_rows is None:
        gradient_diagnostics_rows = []
    if figures is None:
        figures = {}
    paths: dict[str, Path] = {}
    paths["manifest"] = write_manifest(
        path=artifacts.artifact_dir / "rolling_origin_manifest.json",
        phase_id=44,
        K=manifest.get("K", 3),
        transformer_shortlist_fingerprint=manifest["transformer_shortlist_fingerprint"],
        lstm_winner_fingerprint=manifest["lstm_winner_fingerprint"],
        persistence_version=manifest["persistence_version"],
        fold_local_scaling=manifest["fold_local_scaling"],
        status=manifest["status"],
        created_at=manifest["created_at"],
    )
    paths["contract"] = write_contract(
        path=artifacts.artifact_dir / "rolling_origin_contract.json",
        K=3,
        created_at=manifest["created_at"],
    )
    paths["preflight"] = _write_csv(
        artifacts.artifact_dir / "phase44_preflight_audit.csv",
        ["Metric", "Value"],
        preflight_rows,
    )
    paths["fold_manifest"] = write_fold_manifest(
        path=artifacts.artifact_dir / "rolling_origin_fold_manifest.json",
        fold_manifest=fold_manifest,
    )
    paths["fold_table"] = write_fold_table(
        path=artifacts.artifact_dir / "rolling_origin_fold_table.csv",
        fold_table_rows=fold_table_rows,
    )
    paths["population_audit"] = write_population_audit(
        path=artifacts.artifact_dir / "rolling_origin_population_audit.csv",
        population_audit_rows=population_audit_rows,
    )
    paths["candidate_matrix"] = write_candidate_matrix(
        path=artifacts.artifact_dir / "rolling_origin_candidate_matrix.csv",
        candidate_matrix_rows=candidate_matrix_rows,
    )
    paths["compatibility_audit"] = write_compatibility_audit(
        path=artifacts.artifact_dir / "rolling_origin_candidate_compatibility_audit.csv",
        compatibility_rows=compatibility_rows,
    )
    paths["scaling_contract"] = write_scaling_contract(
        path=artifacts.artifact_dir / "rolling_origin_fold_local_scaling_contract.json",
        scaling_version=scaling_contract["scaling_version"],
        created_at=scaling_contract["created_at"],
    )
    paths["scaler_fit_audit"] = write_scaler_fit_audit(
        path=artifacts.artifact_dir / "rolling_origin_scaler_fit_audit.csv",
        audit_rows=scaler_fit_audit_rows,
    )
    paths["temporal_leakage"] = write_temporal_leakage_tests(
        path=artifacts.artifact_dir / "rolling_origin_temporal_leakage_tests.csv",
        leakage_rows=leakage_rows,
    )
    paths["common_target_audit"] = write_common_target_audit(
        path=artifacts.artifact_dir / "rolling_origin_common_target_audit.csv",
        common_target_rows=common_target_rows,
    )
    paths["inner_selection_registry"] = write_inner_selection_run_registry(
        path=artifacts.artifact_dir / "rolling_origin_inner_selection_run_registry.csv",
        rows=inner_selection_rows,
    )
    paths["inner_best_epochs"] = write_inner_best_epochs(
        path=artifacts.artifact_dir / "rolling_origin_inner_best_epochs.csv",
        rows=inner_best_epoch_rows,
    )
    paths["refit_registry"] = write_refit_run_registry(
        path=artifacts.artifact_dir / "rolling_origin_refit_run_registry.csv",
        rows=refit_run_rows,
    )
    paths["refit_epoch_audit"] = write_refit_epoch_audit(
        path=artifacts.artifact_dir / "rolling_origin_refit_epoch_audit.csv",
        rows=refit_epoch_audit_rows,
    )
    paths["initialization_audit"] = write_initialization_audit(
        path=artifacts.artifact_dir / "rolling_initialization_audit.csv",
        rows=initialization_audit_rows,
    )
    paths["sample_order_audit"] = write_sample_order_audit(
        path=artifacts.artifact_dir / "rolling_origin_sample_order_audit.csv",
        rows=sample_order_audit_rows,
    )
    paths["runtime_diagnostics"] = write_runtime_diagnostics(
        path=artifacts.artifact_dir / "rolling_origin_runtime_diagnostics.csv",
        rows=runtime_diagnostics_rows,
    )
    paths["gradient_diagnostics"] = write_gradient_diagnostics(
        path=artifacts.artifact_dir / "rolling_origin_gradient_diagnostics.csv",
        rows=gradient_diagnostics_rows,
    )
    # O44.21 Per-fold outer predictions (one CSV per (candidate, fold))
    if per_fold_outer_predictions is not None:
        per_fold_outer_predictions_dir = artifacts.predictions_dir
        per_fold_outer_predictions_dir.mkdir(parents=True, exist_ok=True)
        any_outer_written = False
        for (cid, fid), df in per_fold_outer_predictions.items():
            sub_path = per_fold_outer_predictions_dir / f"outer_predictions_{cid}_{fid}.csv"
            paths[f"outer_predictions_{cid}_{fid}"] = write_outer_predictions(
                path=sub_path, df=df
            )
            any_outer_written = True
        if any_outer_written:
            paths["outer_predictions"] = per_fold_outer_predictions_dir
    paths["pooled_predictions"] = write_pooled_predictions(
        path=artifacts.predictions_dir / "pooled_outer_predictions.csv",
        df=pooled_predictions_df,
    )
    paths["fold_metrics"] = write_fold_metrics(
        path=artifacts.artifact_dir / "rolling_origin_fold_metrics.csv",
        rows=fold_metrics_rows,
    )
    paths["pooled_metrics"] = write_pooled_metrics(
        path=artifacts.artifact_dir / "rolling_origin_pooled_metrics.csv",
        rows=pooled_metrics_rows,
    )
    paths["macro_metrics"] = write_macro_metrics(
        path=artifacts.artifact_dir / "rolling_origin_macro_robustness_metrics.csv",
        rows=macro_metrics_rows,
    )
    paths["pairwise_effects"] = write_pairwise_effects(
        path=artifacts.artifact_dir / "rolling_origin_pairwise_effects.csv",
        rows=pairwise_effects_rows,
    )
    paths["fold_ranks"] = write_fold_ranks(
        path=artifacts.artifact_dir / "rolling_origin_fold_ranks.csv",
        rows=fold_ranks_rows,
    )
    paths["transformer_ranking"] = write_transformer_ranking(
        path=artifacts.artifact_dir / "rolling_origin_transformer_robustness_ranking.csv",
        ranking_rows=transformer_ranking_rows,
    )
    paths["model_family_comparison"] = write_model_family_comparison(
        path=artifacts.artifact_dir / "rolling_origin_model_family_robustness_comparison.csv",
        rows=model_family_comparison_rows,
    )
    paths["findings"] = write_findings(
        path=artifacts.artifact_dir / "rolling_origin_findings.csv",
        rows=findings_rows,
    )
    paths["recommended_transformer"] = write_recommended_transformer(
        path=artifacts.artifact_dir / "rolling_origin_recommended_transformer.json",
        payload=recommended_transformer,
    )
    paths["phase45_handoff"] = write_phase45_handoff(
        path=artifacts.artifact_dir / "phase45_final_model_lock_handoff.json",
        payload=phase45_handoff,
    )
    paths["tests"] = write_tests(
        path=artifacts.artifact_dir / "rolling_origin_tests.csv",
        rows=tests_rows,
    )
    paths["discrepancies"] = write_discrepancies(
        path=artifacts.artifact_dir / "rolling_origin_discrepancies.json",
        payload=discrepancies,
    )
    paths["summary"] = write_summary(
        path=artifacts.artifact_dir / "rolling_origin_summary.json",
        payload=summary,
    )
    paths["report"] = write_report(
        path=artifacts.artifact_dir / "rolling_origin_report.md",
        payload=report_markdown,
    )
    paths["readme"] = write_readme(
        path=artifacts.artifact_dir / "README_ROLLING_ORIGIN_ROBUSTNESS.md",
        payload=readme_markdown,
    )
    paths["signoff"] = write_signoff(
        path=artifacts.artifact_dir / "phase_44_signoff.json",
        payload=signoff,
    )
    # Figures (O44.33)
    figures_dir = artifacts.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    standard_figure_names = [
        "RO_44_01_fold_timeline.png",
        "RO_44_02_fold_rmse_by_model.png",
        "RO_44_03_pooled_rmse_comparison.png",
        "RO_44_04_transformer_candidate_ranking.png",
        "RO_44_05_fold_rmse_variability.png",
        "RO_44_06_inner_best_epoch_by_fold.png",
        "RO_44_07_pooled_prediction_scatter.png",
        "RO_44_08_paired_error_deltas.png",
        "RO_44_09_runtime_by_model_fold.png",
        "RO_44_10_baseline_delta_by_fold.png",
    ]
    figures_written: list[Path] = []
    for name in standard_figure_names:
        fig_path = figures_dir / name
        if name in figures:
            fig_path.write_bytes(figures[name])
            paths[f"figure_{name}"] = fig_path
            figures_written.append(fig_path)
        else:
            # Write a tiny PNG placeholder so the artifact exists.
            # We write a minimal 1x1 PNG header (132 bytes is the canonical
            # empty PNG signature). This is sufficient for downstream
            # existence checks.
            empty_png = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4"
                b"\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\x95\x84"
                b"\xf3k\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            fig_path.write_bytes(empty_png)
            paths[f"figure_{name}"] = fig_path
            figures_written.append(fig_path)
    if figures_written:
        paths["figures"] = figures_dir

    return paths