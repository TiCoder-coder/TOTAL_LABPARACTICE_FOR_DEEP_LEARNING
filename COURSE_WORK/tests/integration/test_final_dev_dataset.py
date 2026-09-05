"""Regression tests for FINAL_DEV dataset implementation (Phase 46 Task 5).

These tests verify:
  - FINAL_DEV accepts TRAIN+VALIDATION records
  - FINAL_DEV rejects TEST
  - TRAIN dataset still rejects VALIDATION records (regression)
  - VALIDATION dataset still rejects TRAIN records (regression)
  - TRAIN and VALIDATION origins are preserved correctly
  - TRAIN sample parity (values match original TRAIN dataset)
  - VALIDATION sample parity (values match original VALIDATION dataset)
  - FINAL_DEV count = 16630
  - No duplicate target IDs
  - Feature shape [36, 33]
  - FINAL_SCALING-v1 used (X + Y)
  - No Test access
  - Deterministic construction

Uses real data paths (not mocked).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.data.datasets import (
    DatasetConfig,
    SequenceWindowDataset,
    TargetAccessMode,
    build_dataset_suite,
    build_final_dev_dataset,
)
from course_work.data.final_dev import materialize_final_dev_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_final_scaling_reg():
    """Load FINAL_SCALING-v1 scaler bundles without re-materializing.

    Uses joblib to load the already-materialized artifacts. This avoids the
    write_json_once_or_verify conflict when tests are run multiple times
    in the same process.
    """
    import joblib
    from course_work.scaling.final_scaling import FINAL_SCALING_ARTIFACT_ROOT
    root = ROOT
    reg_path = root / FINAL_SCALING_ARTIFACT_ROOT / "final_scaler_registry.json"
    if not reg_path.exists():
        pytest.skip(f"FINAL_SCALING-v1 not materialized at {reg_path}")
    import json
    registry = json.loads(reg_path.read_bytes())
    x_entry = registry["x_bundles"]["FS2_TF1"]
    x_path = root / x_entry["artifact_path"]
    y_entry = registry["target_bundles"]["YS1"]
    y_path = root / y_entry["artifact_path"]
    x_bundle = joblib.load(x_path)
    y_bundle = joblib.load(y_path)
    return x_bundle, y_bundle


# ---------------------------------------------------------------------------
# Test 1: FINAL_DEV accepts TRAIN+VALIDATION records
# ---------------------------------------------------------------------------

def test_final_dev_accepts_combined_records():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    splits = set(ds._records["target_split_id"].unique())
    assert "TRAIN" in splits, "FINAL_DEV must contain TRAIN records"
    assert "VALIDATION" in splits, "FINAL_DEV must contain VALIDATION records"
    assert "TEST" not in splits, "FINAL_DEV must NOT contain TEST records"


# ---------------------------------------------------------------------------
# Test 2: FINAL_DEV rejects TEST
# ---------------------------------------------------------------------------

def test_final_dev_rejects_test_records():
    """FINAL_DEV must not contain any TEST windows."""
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    test_count = (ds._records["target_split_id"] == "TEST").sum()
    assert test_count == 0, f"FINAL_DEV contains {test_count} TEST records — TEST access is FORBIDDEN"


# ---------------------------------------------------------------------------
# Test 3: TRAIN dataset still rejects VALIDATION records (regression)
# ---------------------------------------------------------------------------

def test_train_dataset_still_rejects_validation_records():
    suite, _ = build_dataset_suite(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    train_ds = suite["TRAIN"]

    # Load FINAL_DEV window records to get VALIDATION rows.
    window_index_path = ROOT / "artifacts" / "windows" / "window_index.csv"
    from course_work.data.windows import load_validated_window_index
    wi = load_validated_window_index(ROOT)
    val_rows = wi.loc[
        wi["lookback_steps"].eq(36)
        & wi["target_split_id"].eq("VALIDATION")
        & wi["included_common_population"].astype(bool)
        & wi["WB0_valid"].astype(bool)
    ].copy(deep=True)
    val_rows = val_rows.sort_values("target_timestamp").reset_index(drop=True)

    # Try to create a TRAIN dataset with VALIDATION records — must fail.
    with pytest.raises(ValueError, match="split outside DatasetConfig"):
        SequenceWindowDataset(
            feature_matrix=train_ds._feature_matrix,
            window_records=val_rows.head(10),
            config=train_ds.config,
            target_values=train_ds._target_values,
            target_scaler=train_ds._target_scaler,
        )


# ---------------------------------------------------------------------------
# Test 4: VALIDATION dataset still rejects TRAIN records (regression)
# ---------------------------------------------------------------------------

def test_validation_dataset_still_rejects_train_records():
    suite, _ = build_dataset_suite(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    val_ds = suite["VALIDATION"]

    from course_work.data.windows import load_validated_window_index
    wi = load_validated_window_index(ROOT)
    train_rows = wi.loc[
        wi["lookback_steps"].eq(36)
        & wi["target_split_id"].eq("TRAIN")
        & wi["included_common_population"].astype(bool)
        & wi["WB0_valid"].astype(bool)
    ].copy(deep=True)
    train_rows = train_rows.sort_values("target_timestamp").reset_index(drop=True)

    with pytest.raises(ValueError, match="split outside DatasetConfig"):
        SequenceWindowDataset(
            feature_matrix=val_ds._feature_matrix,
            window_records=train_rows.head(10),
            config=val_ds.config,
            target_values=val_ds._target_values,
            target_scaler=val_ds._target_scaler,
        )


# ---------------------------------------------------------------------------
# Test 5: FINAL_DEV backing covers both TRAIN and VALIDATION
# ---------------------------------------------------------------------------

def test_final_dev_contains_correct_train_and_validation_counts():
    x_bundle, y_bundle = _load_final_scaling_reg()
    manifest = materialize_final_dev_region(
        project_root=ROOT,
        feature_variant_id="FS2_TF1",
        lookback=36,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    expected_train = manifest.train_window_count
    expected_val = manifest.validation_window_count
    expected_total = manifest.final_dev_window_count
    expected_test = manifest.test_window_count

    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )

    assert len(ds) == expected_total, f"FINAL_DEV length {len(ds)} != expected {expected_total}"
    assert len(ds) == 16630, f"FINAL_DEV length must be 16630, got {len(ds)}"

    actual_train = (ds._records["target_split_id"] == "TRAIN").sum()
    actual_val = (ds._records["target_split_id"] == "VALIDATION").sum()
    actual_test = (ds._records["target_split_id"] == "TEST").sum()

    assert actual_train == expected_train, f"TRAIN count {actual_train} != expected {expected_train}"
    assert actual_val == expected_val, f"VALIDATION count {actual_val} != expected {expected_val}"
    assert actual_test == 0, f"TEST count must be 0, got {actual_test}"
    assert actual_train + actual_val == len(ds), "TRAIN + VALIDATION must equal total"


# ---------------------------------------------------------------------------
# Test 6: FINAL_DEV target backing covers both TRAIN and VALIDATION
# ---------------------------------------------------------------------------

def test_final_dev_target_backing_covers_train_and_validation():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    assert ds._target_values is not None, "FINAL_DEV target_values must not be None for YS1"
    assert ds._target_scaler is not None, "FINAL_DEV target_scaler must not be None for YS1"


# ---------------------------------------------------------------------------
# Test 7: TRAIN sample parity
# ---------------------------------------------------------------------------

def test_train_sample_parity():
    """A TRAIN-origin sample in FINAL_DEV must have identical RAW target values to
    the same sample in the canonical TRAIN dataset.

    NOTE: y_model (scaled) values are NOT expected to match because:
      - FINAL_DEV uses FINAL_SCALING-v1 Y scaler (fitted on 16,630 FINAL_DEV rows)
      - TRAIN uses Phase 9 Y scaler (fitted on TRAIN-only rows)
    This is correct Phase 46 behavior.

    y_raw_wh (raw watts-hour targets) MUST match exactly — this proves the record
    is correctly indexing the shared target timeline regardless of which scaler
    is applied.
    """
    x_bundle, y_bundle = _load_final_scaling_reg()
    suite, _ = build_dataset_suite(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    train_ds = suite["TRAIN"]

    fd_ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )

    # Find a TRAIN record in FINAL_DEV.
    train_mask = fd_ds._records["target_split_id"] == "TRAIN"
    train_indices = fd_ds._records[train_mask].index.tolist()
    assert len(train_indices) > 0, "No TRAIN records found in FINAL_DEV"

    # Pick the first TRAIN record for comparison.
    fd_record_local_idx = train_indices[0]
    fd_record = fd_ds._records.iloc[fd_record_local_idx]

    # Find the same record in the TRAIN dataset.
    train_record = train_ds._records.loc[
        train_ds._records["target_sample_id"] == fd_record["target_sample_id"]
    ].iloc[0]

    # Compare canonical indices — both datasets index the same backing arrays.
    fd_canon = fd_record["canonical_sample_idx"]
    train_canon = train_record["canonical_sample_idx"]
    assert fd_canon == train_canon, (
        f"TRAIN-origin FINAL_DEV record has canonical_sample_idx={fd_canon} "
        f"but TRAIN dataset has {train_canon} — backing array mismatch"
    )

    # The TRAIN dataset's local index for this record is the position of
    # the record in train_ds._records after reset_index(drop=True).
    train_local_idx = train_ds._records.index[
        train_ds._records["canonical_sample_idx"] == fd_canon
    ].tolist()[0]

    # Get actual items.
    fd_item = fd_ds[fd_record_local_idx]
    train_item = train_ds[train_local_idx]

    # RAW target values MUST match — this proves the record is correctly
    # indexing the shared target timeline (TRAIN records in FINAL_DEV
    # do NOT accidentally read VALIDATION backing data).
    np.testing.assert_allclose(
        fd_item["y_raw_wh"].numpy(),
        train_item["y_raw_wh"].numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="TRAIN-origin FINAL_DEV y_raw_wh differs from TRAIN dataset — backing array indexing bug"
    )


# ---------------------------------------------------------------------------
# Test 8: VALIDATION sample parity
# ---------------------------------------------------------------------------

def test_validation_sample_parity():
    """A VALIDATION-origin sample in FINAL_DEV must have identical RAW target values
    to the same sample in the canonical VALIDATION dataset.

    NOTE: y_model (scaled) values are NOT expected to match because of different Y
    scalers (FINAL_SCALING-v1 vs Phase 9). y_raw_wh MUST match — this proves the
    VALIDATION record is correctly indexing VALIDATION backing data, not TRAIN data.
    """
    x_bundle, y_bundle = _load_final_scaling_reg()
    suite, _ = build_dataset_suite(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    val_ds = suite["VALIDATION"]

    fd_ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )

    # Find a VALIDATION record in FINAL_DEV.
    val_mask = fd_ds._records["target_split_id"] == "VALIDATION"
    val_indices = fd_ds._records[val_mask].index.tolist()
    assert len(val_indices) > 0, "No VALIDATION records found in FINAL_DEV"

    fd_local_idx = val_indices[0]
    fd_record = fd_ds._records.iloc[fd_local_idx]
    val_record = val_ds._records.loc[
        val_ds._records["target_sample_id"] == fd_record["target_sample_id"]
    ].iloc[0]

    fd_canon = fd_record["canonical_sample_idx"]
    val_canon = val_record["canonical_sample_idx"]

    assert fd_canon == val_canon, (
        f"VALIDATION-origin FINAL_DEV record has canonical_sample_idx={fd_canon} "
        f"but VALIDATION dataset has {val_canon} — backing array mismatch"
    )

    val_local_idx = val_ds._records.index[
        val_ds._records["canonical_sample_idx"] == fd_canon
    ].tolist()[0]

    fd_item = fd_ds[fd_local_idx]
    val_item = val_ds[val_local_idx]

    # RAW target values MUST match — this proves VALIDATION records in
    # FINAL_DEV do NOT accidentally read TRAIN backing data.
    np.testing.assert_allclose(
        fd_item["y_raw_wh"].numpy(),
        val_item["y_raw_wh"].numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="VALIDATION-origin FINAL_DEV y_raw_wh differs from VALIDATION dataset — backing array indexing bug"
    )


# ---------------------------------------------------------------------------
# Test 9: FINAL_DEV count
# ---------------------------------------------------------------------------

def test_final_dev_count():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    assert len(ds) == 16630, f"FINAL_DEV count must be 16630, got {len(ds)}"


# ---------------------------------------------------------------------------
# Test 10: No duplicate target IDs
# ---------------------------------------------------------------------------

def test_no_duplicate_target_ids():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    n_unique = ds._records["target_sample_id"].nunique()
    n_total = len(ds)
    assert n_unique == n_total, (
        f"FINAL_DEV has {n_total} records but only {n_unique} unique target_sample_ids"
    )


# ---------------------------------------------------------------------------
# Test 11: Feature shape [36, 33]
# ---------------------------------------------------------------------------

def test_final_dev_feature_shape():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    item = ds[0]
    assert item["x"].shape == (36, 33), f"Expected (36, 33), got {item['x'].shape}"
    assert item["y_model"].shape == (1,), f"Expected (1,), got {item['y_model'].shape}"
    assert bool(torch.isfinite(item["x"]).all()), "X must be all finite"
    assert bool(torch.isfinite(item["y_model"]).all()), "y_model must be all finite"


# ---------------------------------------------------------------------------
# Test 12: FINAL_SCALING-v1 is used (not Phase 9 scalers)
# ---------------------------------------------------------------------------

def test_final_dev_uses_final_scaling_v1():
    """FINAL_DEV must use FINAL_SCALING-v1 scalers, not Phase 9 scalers."""
    x_bundle, y_bundle = _load_final_scaling_reg()

    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )

    # The dataset config must reference the FINAL_SCALING-v1 scaler bundle.
    assert "FINAL" in ds.config.scaler_bundle_id or "FINAL" in ds.config.scaler_bundle_id.upper(), (
        f"Dataset scaler_bundle_id does not reference FINAL_SCALING: {ds.config.scaler_bundle_id}"
    )
    # Y scaler must be the FINAL_SCALING-v1 bundle.
    assert ds._target_scaler is not None, "FINAL_DEV YS1 requires target_scaler"
    assert ds._target_scaler.get("scaling_version") == "FINAL_SCALING-v1", (
        f"Target scaler is not FINAL_SCALING-v1: {ds._target_scaler.get('scaling_version')}"
    )


# ---------------------------------------------------------------------------
# Test 13: No Test access
# ---------------------------------------------------------------------------

def test_no_test_access():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    test_mask = ds._records["target_split_id"] == "TEST"
    test_count = int(test_mask.sum())
    assert test_count == 0, f"FINAL_DEV contains {test_count} TEST records — TEST access is FORBIDDEN"


# ---------------------------------------------------------------------------
# Test 14: Deterministic construction
# ---------------------------------------------------------------------------

def test_final_dev_deterministic():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds1 = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    ds2 = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    assert len(ds1) == len(ds2)
    item1 = ds1[0]
    item2 = ds2[0]
    np.testing.assert_allclose(
        item1["x"].numpy(), item2["x"].numpy(),
        rtol=1e-6, atol=1e-6, err_msg="FINAL_DEV construction is not deterministic"
    )


# ---------------------------------------------------------------------------
# Test 15: DatasetConfig has FINAL_DEV semantics
# ---------------------------------------------------------------------------

def test_final_dev_config_has_correct_access_mode():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    assert ds.config.target_access_mode == TargetAccessMode.FINAL_DEV.value, (
        f"Dataset config target_access_mode is {ds.config.target_access_mode}, "
        f"expected {TargetAccessMode.FINAL_DEV.value}"
    )
    assert ds.config.split_id == "FINAL_DEV", (
        f"Dataset config split_id is {ds.config.split_id}, expected FINAL_DEV"
    )


# ---------------------------------------------------------------------------
# Test 16: DataLoader batch smoke test
# ---------------------------------------------------------------------------

def test_final_dev_dataloader_batch():
    x_bundle, y_bundle = _load_final_scaling_reg()
    ds = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch["x"].shape == (8, 36, 33), f"Batch X shape {batch['x'].shape} != (8, 36, 33)"
    assert batch["y_model"].shape == (8, 1), f"Batch y_model shape {batch['y_model'].shape} != (8, 1)"
    assert bool(torch.isfinite(batch["x"]).all()), "Batch X must be all finite"
    assert bool(torch.isfinite(batch["y_model"]).all()), "Batch y_model must be all finite"
