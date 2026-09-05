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


def write_distribution_summary_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_residual_distribution_summary.csv"
    fieldnames = ["seed", *[
        "n", "mean", "std", "min", "p01", "p05", "q1", "median", "q3",
        "p95", "p99", "max", "iqr", "mad", "skewness", "kurtosis_excess",
    ]]
    return _write_csv(path, fieldnames, rows)


def write_signed_bias_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_signed_bias.csv"
    fieldnames = [
        "seed", "n",
        "mean_residual_wh", "median_residual_wh", "mean_absolute_residual_wh",
        "mean_underprediction_residual_wh", "mean_overprediction_residual_wh",
        "underprediction_count", "overprediction_count", "exact_count",
        "underprediction_fraction", "overprediction_fraction", "exact_fraction",
        "fractions_sum_to_one",
    ]
    return _write_csv(path, fieldnames, rows)


def write_sign_balance_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_sign_balance.csv"
    fieldnames = [
        "seed", "n",
        "underprediction_count", "underprediction_fraction",
        "overprediction_count", "overprediction_fraction",
        "exact_count", "exact_fraction",
        "fractions_sum_to_one",
    ]
    return _write_csv(path, fieldnames, rows)


def write_tail_diagnostics_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_tail_diagnostics.csv"
    fieldnames = [
        "seed", "n",
        "abs_error_p90", "abs_error_p95", "abs_error_p99",
        "max_absolute_error",
        "fraction_above_p90", "fraction_above_p95", "fraction_above_p99",
    ]
    return _write_csv(path, fieldnames, rows)


def write_histogram_csv(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_residual_histogram_50bins.csv"
    fieldnames = [
        "bin_index", "bin_left", "bin_right", "bin_center",
        "seed", "count", "fraction",
    ]
    return _write_csv(path, fieldnames, rows)


def write_histogram_bin_edges_csv(
    edges: np.ndarray,
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_histogram_bin_edges.csv"
    rows = [
        {"edge_index": i, "edge_value": float(edges[i])}
        for i in range(edges.size)
    ]
    return _write_csv(path, ["edge_index", "edge_value"], rows)


def write_phase49_c_manifest(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = _phase49_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_c_manifest.json"
    return _write_json(path, payload)


def snapshot_phase49_b_artifacts(project_root: Path | None = None) -> dict[str, str]:
    root = project_root if project_root is not None else get_project_root()
    out: dict[str, str] = {}
    for fname in [
        "residual_long_table.csv",
        "residual_wide_table.csv",
        "phase49_b_manifest.json",
        "phase49_source_verification.json",
        "phase49_field_consistency_audit.csv",
        "phase49_metric_reconstruction_audit.csv",
    ]:
        p = root / PHASE49_ARTIFACT_DIR_REL / fname
        if p.exists():
            out[fname] = sha256_file(p)
    return out
