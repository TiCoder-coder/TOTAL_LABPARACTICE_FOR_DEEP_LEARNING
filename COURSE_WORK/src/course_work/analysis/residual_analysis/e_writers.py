from __future__ import annotations

from pathlib import Path
from typing import Any

from course_workutils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
    sha256_file,
)


PHASE49_ARTIFACT_DIR_REL = Path("artifacts/residual_analysis")


def _phase49_dir(project_root: Path) -> Path:
    return project_root / PHASE49_ARTIFACT_DIR_REL


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> Path:
    return atomic_write_bytes(path, csv_text(fieldnames, rows).encode("utf-8"))


def _write_json(path: Path, value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def write_magnitude_associations_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_magnitude_associations.csv"
    fieldnames = [
        "seed", "x_variable", "y_variable", "association_type",
        "value", "N", "status", "causal_interpretation",
    ]
    return _write_csv(path, fieldnames, rows)


def write_prediction_deciles_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_prediction_deciles.csv"
    fieldnames = [
        "seed", "decile", "N",
        "pred_min", "pred_max",
        "mean_prediction",
        "mean_residual", "median_residual",
        "mae", "rmse",
        "underprediction_fraction", "overprediction_fraction", "exact_fraction",
        "edge_left", "edge_right",
        "status",
    ]
    return _write_csv(path, fieldnames, rows)


def write_cross_seed_residual_agreement_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_cross_seed_residual_agreement.csv"
    fieldnames = [
        "seed_a", "seed_b", "N",
        "pearson_residual_correlation",
        "spearman_residual_correlation",
        "mean_absolute_residual_difference",
        "rmse_between_residuals",
        "max_absolute_residual_difference",
        "scope", "purpose",
    ]
    return _write_csv(path, fieldnames, rows)


def write_cross_seed_sign_consensus_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_cross_seed_sign_consensus.csv"
    fieldnames = ["consensus_class", "count", "fraction"]
    return _write_csv(path, fieldnames, rows)


def write_persistence_context_csv(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_persistence_context.csv"
    field_row = {
        "N": payload["N"],
        "mean_residual": payload["mean_residual"],
        "median_residual": payload["median_residual"],
        "std_residual": payload["std_residual"],
        "mae": payload["mae"],
        "rmse": payload["rmse"],
        "underprediction_count": payload["underprediction_count"],
        "overprediction_count": payload["overprediction_count"],
        "exact_count": payload["exact_count"],
        "underprediction_fraction": payload["underprediction_fraction"],
        "overprediction_fraction": payload["overprediction_fraction"],
        "exact_fraction": payload["exact_fraction"],
        "residual_convention": payload["residual_convention"],
        "scope": payload["scope"],
    }
    return _write_csv(path, list(field_row.keys()), [field_row])


def write_phase49_e_manifest(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_e_manifest.json"
    return _write_json(path, payload)


def snapshot_phase49_b_c_d_artifacts(project_root: Path | None = None) -> dict[str, str]:
    root = project_root if project_root is not None else get_project_root()
    out: dict[str, str] = {}
    b_c_d_files = [
        "residual_long_table.csv",
        "residual_wide_table.csv",
        "phase49_b_manifest.json",
        "phase49_source_verification.json",
        "phase49_field_consistency_audit.csv",
        "phase49_metric_reconstruction_audit.csv",
        "phase49_residual_distribution_summary.csv",
        "phase49_signed_bias.csv",
        "phase49_sign_balance.csv",
        "phase49_tail_diagnostics.csv",
        "phase49_residual_histogram_50bins.csv",
        "phase49_histogram_bin_edges.csv",
        "phase49_c_manifest.json",
        "phase49_residual_acf.csv",
        "phase49_residual_acf_key_lags.csv",
        "phase49_ljung_box.csv",
        "phase49_sign_runs.csv",
        "phase49_sign_run_table_seed42.csv",
        "phase49_sign_run_table_seed123.csv",
        "phase49_sign_run_table_seed2026.csv",
        "phase49_sign_transitions.csv",
        "phase49_rolling_residual_diagnostics.csv",
        "phase49_d_manifest.json",
    ]
    for fname in b_c_d_files:
        p = root / PHASE49_ARTIFACT_DIR_REL / fname
        if p.exists():
            out[fname] = sha256_file(p)
    return out
