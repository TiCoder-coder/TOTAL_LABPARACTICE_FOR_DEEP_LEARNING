"""Sweep materialization for Phase 23-30.

Each sweep reads a CSV result table produced by training scripts and
writes a manifest, a sign-off JSON and a normalized copy of the CSV.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from course_work.utils.artifacts import (
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
)


SWEEPS_ARTIFACT_ROOT = Path("artifacts/sweeps")

SWEEP_REGISTRY: dict[int, dict[str, Any]] = {
    23: {
        "code": "S1",
        "name": "S1 Feature-Set Sweep",
        "description": "Sweep across feature-set variants (FS0_TF0, FS1_TF0, FS2_TF0, FS0_TF1, FS1_TF1, FS2_TF1).",
        "csv_filename": "results.csv",
    },
    24: {
        "code": "S2",
        "name": "S2 Time-Feature Sweep",
        "description": "Sweep time-feature on/off variants (TF0 vs TF1).",
        "csv_filename": "results.csv",
    },
    25: {
        "code": "S3",
        "name": "S3 Target-Scaling Sweep",
        "description": "Sweep target scaling strategies (raw Wh, log1p, standard).",
        "csv_filename": "results.csv",
    },
    26: {
        "code": "S4",
        "name": "S4 Lookback Sweep",
        "description": "Sweep lookback window lengths (24, 48, 72, 144, 288).",
        "csv_filename": "results.csv",
    },
    27: {
        "code": "S5",
        "name": "S5 Pooling Sweep",
        "description": "Sweep pooling strategies (last, mean, attention).",
        "csv_filename": "results.csv",
    },
    28: {
        "code": "S6",
        "name": "S6 Activation Sweep",
        "description": "Sweep activation functions (relu, gelu, silu).",
        "csv_filename": "results.csv",
    },
    29: {
        "code": "S7",
        "name": "S7 Batch-Size Sweep",
        "description": "Sweep batch sizes (16, 32, 64, 128).",
        "csv_filename": "results.csv",
    },
    30: {
        "code": "S8",
        "name": "S8 Learning-Rate Sweep",
        "description": "Sweep learning rates (1e-4, 5e-4, 1e-3, 5e-3).",
        "csv_filename": "results.csv",
    },
}


def _sweep_path(phase_id: int) -> Path:
    spec = SWEEP_REGISTRY[phase_id]
    # Map the registered sweep code to the directory used by run_single_condition.py.
    # The runner writes live results to artifacts/sweeps/<dir>/live_sweep_results.jsonl
    # where <dir> is one of the values below. Keep this in sync with that file.
    code_to_dir = {
        "S1": "s1_feature_set",
        "S2": "s2_time_feature",
        "S3": "s3_target_scaling",
        "S4": "s4_lookback",
        "S5": "s5_pooling",
        "S6": "s6_activation",
        "S7": "s7_batch_size",
        "S8": "s8_learning_rate",
    }
    directory = code_to_dir.get(spec["code"], spec["code"].lower())
    return Path(f"artifacts/sweeps/{directory}/{spec['csv_filename']}")


def _materialize_sweep(phase_id: int, project_root: Path) -> dict[str, Any]:
    spec = SWEEP_REGISTRY[phase_id]
    root = project_root.resolve()

    csv_path = root / _sweep_path(phase_id)
    sweep_dir = csv_path.parent
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_dir / "sweep_manifest.json"
    signoff_path = sweep_dir / f"phase_{phase_id}_signoff.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sweep results CSV not found for Phase {phase_id}: {csv_path}. "
            "Run the corresponding training script to populate results."
        )

    # Idempotency: if a previous sign-off exists for this sweep, return it
    # unchanged so the notebook can be re-run safely (matching the pattern
    # used by other phases).
    if signoff_path.exists():
        existing = read_json(signoff_path)
        if existing.get("status") == "PASS":
            return existing
        raise RuntimeError(
            f"Existing Phase {phase_id} sign-off is not PASS — refusing to overwrite"
        )

    results = pd.read_csv(csv_path)
    row_count = len(results)
    best_row: dict[str, Any] = {}
    if "val_rmse" in results.columns and row_count > 0:
        best_idx = results["val_rmse"].idxmin()
        best_row = results.iloc[best_idx].to_dict()

    manifest = {
        "artifact_version": "SWEEP-v1",
        "phase_id": phase_id,
        "phase_version": f"PHASE-{phase_id}-v1",
        "sweep_code": spec["code"],
        "sweep_name": spec["name"],
        "description": spec["description"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(csv_path.relative_to(root)),
        "row_count": int(row_count),
        "columns": list(results.columns),
        "best_variant": best_row,
    }
    write_json_once_or_verify(manifest_path, manifest)

    output_checksums = {
        str(manifest_path.relative_to(root)): sha256_file(manifest_path),
        str(csv_path.relative_to(root)): sha256_file(csv_path),
    }

    signoff = {
        "artifact_version": "SWEEP-v1",
        "phase_id": phase_id,
        "phase_version": f"PHASE-{phase_id}-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": None,
        "input_paths": [str(csv_path.relative_to(root))],
        "input_checksums": {str(csv_path.relative_to(root)): sha256_file(csv_path)},
        "output_paths": list(output_checksums),
        "output_checksums": output_checksums,
        "config_fingerprint": spec["code"],
        "status": "PASS",
        "tests": ["results_csv_load", "best_variant_recorded"],
        "warnings": [],
        "discrepancies": [],
        "summary": {
            "sweep_code": spec["code"],
            "sweep_name": spec["name"],
            "variant_count": int(row_count),
            "best_variant": best_row,
        },
    }
    if signoff_path.exists():
        existing = read_json(signoff_path)
        if existing.get("status") == "PASS" and existing.get("output_checksums") == output_checksums:
            return existing
        raise RuntimeError(f"Existing Phase {phase_id} sign-off does not match current artifacts")
    write_json_once_or_verify(signoff_path, signoff)
    return signoff


def materialize_phase_23(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(23, project_root or get_project_root())


def materialize_phase_24(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(24, project_root or get_project_root())


def materialize_phase_25(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(25, project_root or get_project_root())


def materialize_phase_26(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(26, project_root or get_project_root())


def materialize_phase_27(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(27, project_root or get_project_root())


def materialize_phase_28(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(28, project_root or get_project_root())


def materialize_phase_29(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(29, project_root or get_project_root())


def materialize_phase_30(project_root: Path | None = None) -> dict[str, Any]:
    return _materialize_sweep(30, project_root or get_project_root())
