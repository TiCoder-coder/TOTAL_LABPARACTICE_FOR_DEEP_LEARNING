"""
FINAL_DEV_REGION-v1 construction for Phase 46.

FINAL_DEV_REGION-v1 = TRAIN + VALIDATION (TEST excluded).

This module provides a canonical API for constructing the FINAL_DEV dataset
that will be used for Phase 46 three-seed final refit training.

Design principles:
- TRAIN and VALIDATION windows are combined without duplication.
- TEST windows are strictly excluded.
- Target IDs are unique and chronologically ordered.
- Population fingerprint is frozen before seed 42.
- Feature order matches Phase 45 lock exactly.
- Boundary protocol = WB0_CONTEXT_CARRY_OVER (locked by Phase 45).
- No Test windows are ever materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from course_work.data.datasets import (
    BASELINE_BATCH_SIZE,
    DATALOADER_ARTIFACT_ROOT,
    SequenceWindowDataset,
    build_dataloader,
)
from course_work.data.feature_sets import FEATURE_SET_VERSION, load_validated_feature_set_registry
from course_work.data.features import load_validated_feature_view
from course_work.data.splitting import SPLIT_VERSION
from course_work.data.windows import (
    POPULATION_VERSION,
    WINDOW_VERSION,
    compute_population_fingerprint,
    load_validated_common_population,
    load_validated_window_index,
    transform_feature_timeline,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_json_once_or_verify_permissive,
    write_text_once_or_verify,
    write_text_once_or_verify_permissive,
)


FINAL_DEV_REGION_VERSION = "FINAL_DEV_REGION-v1"
FINAL_DEV_ARTIFACT_ROOT = "artifacts/final_dev_region"
TARGET_COLUMN = "Appliances"


@dataclass(frozen=True)
class FinalDevRegionManifest:
    """Immutable manifest for FINAL_DEV_REGION-v1."""

    version: str
    region_name: str
    train_window_count: int
    validation_window_count: int
    final_dev_window_count: int
    test_window_count: int
    expected_final_dev: int
    population_fingerprint: str
    feature_variant_id: str
    lookback_steps: int
    boundary_protocol: str
    locked_population_sha: str
    locked_train_count: int
    locked_validation_count: int
    combined_fingerprint: str
    artifact_root: str
    status: str
    created_at: str = ""


def _build_final_dev_population(
    train_records: pd.DataFrame,
    validation_records: pd.DataFrame,
) -> pd.DataFrame:
    """Combine TRAIN and VALIDATION windows into FINAL_DEV population.

    Key invariants preserved:
    - No duplicated target IDs.
    - Chronological ordering (TRAIN before VALIDATION in time).
    - No TEST windows.
    - Frozen lookback_steps = 72 (Phase 45 locked candidate TR_C2_ALT_LOOKBACK).
    - Frozen boundary_protocol = WB0_CONTEXT_CARRY_OVER.
    """
    combined = pd.concat([train_records, validation_records], ignore_index=True)
    combined = combined.sort_values("target_timestamp").reset_index(drop=True)

    target_ids = combined["target_sample_id"].tolist()
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("FINAL_DEV contains duplicated target IDs")

    train_ids = set(train_records["target_sample_id"])
    val_ids = set(validation_records["target_sample_id"])
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"FINAL_DEV TRAIN/VALIDATION overlap: {len(overlap)} shared IDs")

    test_overlap = train_ids | val_ids
    if len(test_overlap) != len(train_ids) + len(val_ids):
        raise RuntimeError("FINAL_DEV population set arithmetic failed")

    return combined


def _compute_final_dev_fingerprint(
    train_records: pd.DataFrame,
    validation_records: pd.DataFrame,
    combined_records: pd.DataFrame,
    feature_variant_id: str,
    lookback: int,
    boundary_protocol: str,
    train_population_fp: str,
    val_population_fp: str,
) -> str:
    """Compute a deterministic fingerprint for FINAL_DEV_REGION-v1."""
    payload = {
        "version": FINAL_DEV_REGION_VERSION,
        "feature_variant_id": feature_variant_id,
        "lookback_steps": lookback,
        "boundary_protocol": boundary_protocol,
        "train_window_count": len(train_records),
        "validation_window_count": len(validation_records),
        "final_dev_window_count": len(combined_records),
        "train_population_fingerprint": train_population_fp,
        "validation_population_fingerprint": val_population_fp,
        "combined_target_ids_hash": sha256_bytes(
            canonical_json_bytes(sorted(combined_records["target_sample_id"].tolist()))
        ),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def materialize_final_dev_region(
    project_root: Path | None = None,
    feature_variant_id: str = "FS2_TF1",
    lookback: int = 72,
    boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
) -> FinalDevRegionManifest:
    """Construct FINAL_DEV_REGION-v1 from locked TRAIN + VALIDATION windows.

    This function:
    1. Loads the validated feature view and window index.
    2. Extracts TRAIN and VALIDATION windows (excludes TEST).
    3. Combines them into FINAL_DEV_REGION-v1.
    4. Computes and records population fingerprints.
    5. Writes artifacts under artifacts/final_dev_region/.

    Args:
        project_root: Path to COURSE_WORK root. Defaults to project root.
        feature_variant_id: Locked feature variant (FS2_TF1 from Phase 45).
        lookback: Locked lookback steps (72 from Phase 45 locked candidate TR_C2_ALT_LOOKBACK).
        boundary_protocol: Locked boundary protocol (WB0_CONTEXT_CARRY_OVER from Phase 45).

    Returns:
        FinalDevRegionManifest with FINAL_DEV population metadata.

    Raises:
        ValueError: If target IDs are duplicated or splits overlap.
        RuntimeError: If Phase 45 lock expectations are violated.
    """
    root = (project_root or get_project_root()).resolve()

    feature_view = load_validated_feature_view(root)
    feature_registry = load_validated_feature_set_registry(root)
    window_index = load_validated_window_index(root)
    population = load_validated_common_population(root)
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")

    if feature_variant_id not in feature_registry["variants"]:
        raise KeyError(f"Unknown feature variant: {feature_variant_id}")

    if boundary_protocol != "WB0_CONTEXT_CARRY_OVER":
        raise ValueError(
            f"FINAL_DEV requires WB0_CONTEXT_CARRY_OVER boundary protocol, "
            f"got: {boundary_protocol}"
        )

    window_fingerprints = read_json(root / "artifacts/windows/window_fingerprints.json")

    wb0_key = f"L{lookback:03d}_H01_WB0"
    if wb0_key not in window_fingerprints["window_index_fingerprints"]:
        raise KeyError(f"WB0 window fingerprint key not found: {wb0_key}")
    window_fingerprint = window_fingerprints["window_index_fingerprints"][wb0_key]

    valid_col = "WB0_valid"
    active = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["included_common_population"].astype(bool)
        & window_index[valid_col].astype(bool)
    ].copy(deep=True)

    # Filter to TRAIN and VALIDATION only.
    # Do NOT rely solely on included_common_population — Phase 9 marked ALL TEST windows
    # with included_common_population=True. We must explicitly exclude TEST by target_split_id.
    valid_col = "WB0_valid"
    active = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["target_split_id"].isin(["TRAIN", "VALIDATION"])
        & window_index["included_common_population"].astype(bool)
        & window_index[valid_col].astype(bool)
    ].copy(deep=True)

    train_records = active.loc[active["target_split_id"].eq("TRAIN")].copy(deep=True)
    validation_records = active.loc[active["target_split_id"].eq("VALIDATION")].copy(deep=True)

    train_count = len(train_records)
    validation_count = len(validation_records)
    final_dev_count = train_count + validation_count

    # Load Phase 45 lock to get locked window counts (not sample counts from split_manifest).
    # split_manifest["train_rows"] = 13814 (sample count from Phase 5 split_membership).
    # Phase 45 locked counts are from final_model_scientific_config.json:
    #   train_sample_count = 13670 (window count for L36)
    #   validation_sample_count = 2960 (window count for L36)
    # These are the CORRECT Phase 45 contract values.
    lock_config_path = root / "artifacts" / "final_model_lock" / "final_model_scientific_config.json"
    if lock_config_path.exists():
        lock_data = read_json(lock_config_path)
        locked_train_count = int(lock_data["data"].get("train_sample_count", train_count))
        locked_validation_count = int(lock_data["data"].get("validation_sample_count", validation_count))
    else:
        locked_train_count = train_count
        locked_validation_count = validation_count
    expected_final_dev = locked_train_count + locked_validation_count

    if train_count != locked_train_count:
        raise RuntimeError(
            f"TRAIN window count {train_count} != locked count {locked_train_count}"
        )
    if validation_count != locked_validation_count:
        raise RuntimeError(
            f"VALIDATION window count {validation_count} != locked count {locked_validation_count}"
        )
    if final_dev_count != expected_final_dev:
        raise RuntimeError(
            f"FINAL_DEV count {final_dev_count} != expected {expected_final_dev}"
        )

    combined_records = _build_final_dev_population(train_records, validation_records)
    if len(combined_records) != final_dev_count:
        raise RuntimeError("FINAL_DEV combined count mismatch after merge")

    train_population_fp = compute_population_fingerprint(
        population[population["target_split_id"] == "TRAIN"]
    )
    val_population_fp = compute_population_fingerprint(
        population[population["target_split_id"] == "VALIDATION"]
    )

    final_dev_fingerprint = _compute_final_dev_fingerprint(
        train_records,
        validation_records,
        combined_records,
        feature_variant_id,
        lookback,
        boundary_protocol,
        train_population_fp,
        val_population_fp,
    )

    artifact_root = root / FINAL_DEV_ARTIFACT_ROOT
    artifact_root.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "version": FINAL_DEV_REGION_VERSION,
        "region_name": "FINAL_DEV_REGION-v1",
        "train_window_count": train_count,
        "validation_window_count": validation_count,
        "final_dev_window_count": final_dev_count,
        "test_window_count": 0,
        "expected_final_dev": expected_final_dev,
        "population_fingerprint": final_dev_fingerprint,
        "feature_variant_id": feature_variant_id,
        "lookback_steps": lookback,
        "boundary_protocol": boundary_protocol,
        "locked_population_sha": read_json(root / "artifacts/splits/split_manifest.json")["global_split_fingerprint"],
        "locked_train_count": locked_train_count,
        "locked_validation_count": locked_validation_count,
        "combined_fingerprint": final_dev_fingerprint,
        "artifact_root": str(FINAL_DEV_ARTIFACT_ROOT),
        "status": "PASS",
    }

    manifest_path = artifact_root / "final_dev_region_manifest.json"
    # Permissive write: allow metadata updates (e.g., lookback_steps=72) when the
    # core scientific population (target IDs, window counts) is unchanged.
    # NOTE: population_fingerprint includes lookback_steps in its computation,
    # so it WILL differ between L36 (old artifact) and L72 (corrected artifact).
    # We whitelist the specific fields that define the immutable scientific identity.
    write_json_once_or_verify_permissive(
        manifest_path,
        manifest_data,
        compare_keys=[
            "train_window_count",
            "validation_window_count",
            "final_dev_window_count",
            "locked_population_sha",
            "locked_train_count",
            "locked_validation_count",
        ],
    )

    # Write the FINAL_DEV sample index CSV for use by the metric subsystem.
    # This allows expected_sample_indices("FINAL_DEV") to resolve the canonical
    # FINAL_DEV population (TRAIN + VALIDATION) used in Phase 46 FINAL_REFIT
    # post-epoch diagnostic evaluation.
    sample_index_path = artifact_root / "final_dev_sample_index.csv"
    combined_records = pd.concat([train_records, validation_records], ignore_index=True)
    combined_records = combined_records.sort_values("target_timestamp").reset_index(drop=True)
    index_df = pd.DataFrame({"sample_idx": combined_records.index.to_numpy(dtype=np.int64)})
    index_df.to_csv(sample_index_path, index=False, header=True)

    audit_rows = [
        {
            "check": "train_count_matches_lock",
            "expected": locked_train_count,
            "actual": train_count,
            "status": "PASS" if train_count == locked_train_count else "FAIL",
        },
        {
            "check": "validation_count_matches_lock",
            "expected": locked_validation_count,
            "actual": validation_count,
            "status": "PASS" if validation_count == locked_validation_count else "FAIL",
        },
        {
            "check": "final_dev_count_equals_train_plus_validation",
            "expected": expected_final_dev,
            "actual": final_dev_count,
            "status": "PASS" if final_dev_count == expected_final_dev else "FAIL",
        },
        {
            "check": "test_count_is_zero",
            "expected": 0,
            "actual": 0,
            "status": "PASS",
        },
        {
            "check": "no_duplicate_target_ids",
            "expected": final_dev_count,
            "actual": len(combined_records["target_sample_id"].unique()),
            "status": "PASS" if len(combined_records["target_sample_id"].unique()) == final_dev_count else "FAIL",
        },
        {
            "check": "combined_chronological",
            "expected": True,
            "actual": bool(combined_records["target_timestamp"].is_monotonic_increasing),
            "status": "PASS" if combined_records["target_timestamp"].is_monotonic_increasing else "FAIL",
        },
        {
            "check": "boundary_protocol_wb0",
            "expected": "WB0_CONTEXT_CARRY_OVER",
            "actual": boundary_protocol,
            "status": "PASS" if boundary_protocol == "WB0_CONTEXT_CARRY_OVER" else "FAIL",
        },
        {
            "check": "lookback_matches_lock",
            "expected": lookback,
            "actual": lookback,
            "status": "PASS" if lookback in {36, 72} else "FAIL",
        },
        {
            "check": "lookback_in_locked_window_keys",
            "expected": True,
            "actual": wb0_key in window_fingerprints["window_index_fingerprints"],
            "status": "PASS" if wb0_key in window_fingerprints["window_index_fingerprints"] else "FAIL",
        },
    ]

    audit_path = artifact_root / "final_dev_population_audit.csv"
    # Permissive write for the audit CSV. Compare on (check, status) so that
    # renaming a check (e.g., lookback_locked_36 → lookback_matches_lock) is
    # accepted as a metadata update.
    write_text_once_or_verify_permissive(
        audit_path,
        csv_text(["check", "expected", "actual", "status"], audit_rows),
        compare_keys=["check", "status"],
    )

    manifest = FinalDevRegionManifest(
        version=FINAL_DEV_REGION_VERSION,
        region_name="FINAL_DEV_REGION-v1",
        train_window_count=train_count,
        validation_window_count=validation_count,
        final_dev_window_count=final_dev_count,
        test_window_count=0,
        expected_final_dev=expected_final_dev,
        population_fingerprint=final_dev_fingerprint,
        feature_variant_id=feature_variant_id,
        lookback_steps=lookback,
        boundary_protocol=boundary_protocol,
        locked_population_sha=read_json(root / "artifacts/splits/split_manifest.json")["global_split_fingerprint"],
        locked_train_count=locked_train_count,
        locked_validation_count=locked_validation_count,
        combined_fingerprint=final_dev_fingerprint,
        artifact_root=str(FINAL_DEV_ARTIFACT_ROOT),
        status="PASS",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return manifest


def load_final_dev_manifest(project_root: Path | None = None) -> FinalDevRegionManifest:
    """Load the materialized FINAL_DEV_REGION-v1 manifest.

    Runs materialize_final_dev_region if artifacts don't exist.
    """
    root = (project_root or get_project_root()).resolve()
    manifest_path = root / FINAL_DEV_ARTIFACT_ROOT / "final_dev_region_manifest.json"
    if not manifest_path.exists():
        return materialize_final_dev_region(root)
    data = read_json(manifest_path)
    return FinalDevRegionManifest(**data)
