from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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
    payload = csv_text(fieldnames, rows).encode("utf-8")
    return atomic_write_bytes(path, payload)


def _write_json(path: Path, value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def write_residual_long_table(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "residual_long_table.csv"
    fieldnames = [
        "target_id",
        "target_timestamp",
        "seed",
        "y_true_wh",
        "y_pred_wh",
        "residual_wh",
        "absolute_error_wh",
        "squared_error_wh2",
        "residual_sign",
        "source_prediction_sha256",
    ]
    return _write_csv(path, fieldnames, rows)


def write_residual_wide_table(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "residual_wide_table.csv"
    fieldnames = [
        "target_id",
        "target_timestamp",
        "y_true_wh",
        "seed42_y_pred_wh",
        "seed42_residual_wh",
        "seed42_absolute_error_wh",
        "seed42_squared_error_wh2",
        "seed123_y_pred_wh",
        "seed123_residual_wh",
        "seed123_absolute_error_wh",
        "seed123_squared_error_wh2",
        "seed2026_y_pred_wh",
        "seed2026_residual_wh",
        "seed2026_absolute_error_wh",
        "seed2026_squared_error_wh2",
        "seed_mean_residual_wh",
        "seed_mean_residual_semantics",
    ]
    return _write_csv(path, fieldnames, rows)


def build_residual_wide_table_rows(
    long_table_by_seed: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seed42 = long_table_by_seed[42]
    seed123 = long_table_by_seed[123]
    seed2026 = long_table_by_seed[2026]
    n = len(seed42)
    if not (len(seed123) == n == len(seed2026)):
        raise ValueError("Per-seed long tables must have equal length N=2961")

    out: list[dict[str, Any]] = []
    for i in range(n):
        r42 = seed42[i]
        r123 = seed123[i]
        r2026 = seed2026[i]
        if not (
            r42["target_id"] == r123["target_id"] == r2026["target_id"]
            and r42["target_timestamp"] == r123["target_timestamp"] == r2026["target_timestamp"]
            and r42["y_true_wh"] == r123["y_true_wh"] == r2026["y_true_wh"]
        ):
            raise ValueError(
                f"Cross-seed misalignment at row {i}: "
                f"target_id 42={r42['target_id']!r} 123={r123['target_id']!r} 2026={r2026['target_id']!r}"
            )
        seed_mean_residual = (
            r42["residual_wh"] + r123["residual_wh"] + r2026["residual_wh"]
        ) / 3.0
        out.append(
            {
                "target_id": r42["target_id"],
                "target_timestamp": r42["target_timestamp"],
                "y_true_wh": r42["y_true_wh"],
                "seed42_y_pred_wh": r42["y_pred_wh"],
                "seed42_residual_wh": r42["residual_wh"],
                "seed42_absolute_error_wh": r42["absolute_error_wh"],
                "seed42_squared_error_wh2": r42["squared_error_wh2"],
                "seed123_y_pred_wh": r123["y_pred_wh"],
                "seed123_residual_wh": r123["residual_wh"],
                "seed123_absolute_error_wh": r123["absolute_error_wh"],
                "seed123_squared_error_wh2": r123["squared_error_wh2"],
                "seed2026_y_pred_wh": r2026["y_pred_wh"],
                "seed2026_residual_wh": r2026["residual_wh"],
                "seed2026_absolute_error_wh": r2026["absolute_error_wh"],
                "seed2026_squared_error_wh2": r2026["squared_error_wh2"],
                "seed_mean_residual_wh": seed_mean_residual,
                "seed_mean_residual_semantics": "SEED_MEAN_RESIDUAL_DESCRIPTIVE",
            }
        )
    return out


def write_phase49_field_consistency_audit(
    audit_rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase49_field_consistency_audit.csv"
    fieldnames = [
        "seed",
        "field",
        "n",
        "max_abs_difference",
        "mean_abs_difference",
        "tolerance",
        "status",
    ]
    return _write_csv(path, fieldnames, audit_rows)


def write_phase49_metric_reconstruction_audit(
    audit_rows: list[dict[str, Any]],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase49_metric_reconstruction_audit.csv"
    fieldnames = [
        "seed",
        "metric",
        "recomputed_value",
        "stored_phase47_value",
        "absolute_difference",
        "tolerance",
        "status",
    ]
    return _write_csv(path, fieldnames, audit_rows)


def write_source_verification_manifest(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase49_source_verification.json"
    return _write_json(path, payload)


def write_phase49_b_manifest(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out_dir = _phase49_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase49_b_manifest.json"
    return _write_json(path, payload)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def snapshot_existing_artifact_sha(
    path: Path,
    project_root: Path | None = None,
) -> str | None:
    root = project_root if project_root is not None else get_project_root()
    full = root / path if not path.is_absolute() else path
    if not full.exists():
        return None
    return sha256_file(full)
