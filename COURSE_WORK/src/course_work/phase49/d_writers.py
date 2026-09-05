from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils.artifacts import (
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


def write_residual_acf_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_residual_acf.csv"
    fieldnames = [
        "seed", "lag_steps", "lag_minutes",
        "valid_pair_count", "acf_value",
        "segment_count", "status",
    ]
    return _write_csv(path, fieldnames, rows)


def write_acf_key_lags_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_residual_acf_key_lags.csv"
    fieldnames = [
        "seed", "lag_steps", "lag_minutes",
        "valid_pair_count", "acf_value",
        "segment_count", "status",
    ]
    return _write_csv(path, fieldnames, rows)


def write_ljung_box_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_ljung_box.csv"
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "seed", "segment_index", "n", "status",
            "Q_lag6", "p_value_lag6",
            "Q_lag36", "p_value_lag36",
            "Q_lag144", "p_value_lag144",
        ]
    return _write_csv(path, fieldnames, rows)


def write_sign_runs_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_sign_runs.csv"
    fieldnames = [
        "seed", "n_sign_segments", "run_count",
        "underprediction_run_count", "overprediction_run_count",
        "exact_zero_count", "mean_run_length", "median_run_length",
        "max_run_length", "p95_run_length",
        "breaks_at_zero", "breaks_at_gap", "breaks_at_sign_change",
    ]
    return _write_csv(path, fieldnames, rows)


def write_sign_run_table_csv(
    rows: list[dict[str, Any]],
    seed: str,
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"phase49_sign_run_table_seed{seed}.csv"
    fieldnames = ["seed", "run_index", "sign", "length"]
    field_rows = [
        {"seed": seed, "run_index": i, "sign": r["sign"], "length": r["length"]}
        for i, r in enumerate(rows)
    ]
    return _write_csv(path, fieldnames, field_rows)


def write_sign_transitions_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_sign_transitions.csv"
    fieldnames = [
        "seed", "from_sign", "to_sign", "count",
        "outgoing_total_from_a", "probability_given_from_a",
    ]
    return _write_csv(path, fieldnames, rows)


def write_rolling_diagnostics_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_rolling_residual_diagnostics.csv"
    fieldnames = [
        "window_index", "segment_index",
        "window_valid_count",
        "window_start_timestamp", "window_end_timestamp",
        "rolling_residual_mean_wh", "rolling_residual_std_wh",
        "rolling_mae_wh", "rolling_rmse_wh",
        "rolling_underprediction_fraction", "rolling_overprediction_fraction",
        "rolling_exact_fraction",
        "status",
    ]
    return _write_csv(path, fieldnames, rows)


def write_phase49_d_manifest(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_d_manifest.json"
    return _write_json(path, payload)


def snapshot_phase49_b_c_artifacts(project_root: Path | None = None) -> dict[str, str]:
    root = project_root if project_root is not None else get_project_root()
    out: dict[str, str] = {}
    for fname in [
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
    ]:
        p = root / PHASE49_ARTIFACT_DIR_REL / fname
        if p.exists():
            out[fname] = sha256_file(p)
    return out
