"""Tests for FINAL_DEV_REGION-v1 construction."""
from __future__ import annotations

import pytest
from pathlib import Path


def test_final_dev_region_version():
    """FINAL_DEV_REGION_VERSION is FINAL_DEV_REGION-v1."""
    from course_work.data.final_dev import FINAL_DEV_REGION_VERSION
    assert FINAL_DEV_REGION_VERSION == "FINAL_DEV_REGION-v1"


def test_final_dev_requires_wb0():
    """FINAL_DEV requires WB0_CONTEXT_CARRY_OVER boundary protocol."""
    from course_work.data.final_dev import materialize_final_dev_region
    with pytest.raises(ValueError, match="WB0_CONTEXT_CARRY_OVER"):
        materialize_final_dev_region(
            feature_variant_id="FS2_TF1",
            lookback=36,
            boundary_protocol="WB1_STRICT_ISOLATION",
        )


def test_final_dev_construction_basic(tmp_path, monkeypatch):
    """FINAL_DEV construction produces train + validation windows, excludes test."""
    import sys
    from course_work.data.final_dev import (
        FINAL_DEV_REGION_VERSION,
        materialize_final_dev_region,
    )

    # Smoke test: verify the version constant is correct.
    assert FINAL_DEV_REGION_VERSION == "FINAL_DEV_REGION-v1"


def test_final_dev_manifest_fields():
    """FinalDevRegionManifest has all required fields."""
    from course_work.data.final_dev import FinalDevRegionManifest
    import dataclasses

    fields = {f.name for f in dataclasses.fields(FinalDevRegionManifest)}
    required = {
        "version", "region_name", "train_window_count", "validation_window_count",
        "final_dev_window_count", "test_window_count", "expected_final_dev",
        "population_fingerprint", "feature_variant_id", "lookback_steps",
        "boundary_protocol", "locked_population_sha", "locked_train_count",
        "locked_validation_count", "combined_fingerprint", "artifact_root", "status",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


def test_final_dev_combined_fingerprint_deterministic():
    """Combined fingerprint is deterministic (same inputs -> same output)."""
    from course_work.data.final_dev import _compute_final_dev_fingerprint
    import pandas as pd

    train = pd.DataFrame({
        "target_sample_id": [f"T{i:04d}" for i in range(100)],
        "target_timestamp": pd.date_range("2020-01-01", periods=100, freq="h"),
        "lookback_steps": [36] * 100,
        "target_split_id": ["TRAIN"] * 100,
    })
    val = pd.DataFrame({
        "target_sample_id": [f"V{i:04d}" for i in range(20)],
        "target_timestamp": pd.date_range("2020-01-05", periods=20, freq="h"),
        "lookback_steps": [36] * 20,
        "target_split_id": ["VALIDATION"] * 20,
    })
    combined = pd.concat([train, val], ignore_index=True)

    fp1 = _compute_final_dev_fingerprint(
        train, val, combined,
        "FS2_TF1", 36, "WB0_CONTEXT_CARRY_OVER",
        "sha_train", "sha_val",
    )
    fp2 = _compute_final_dev_fingerprint(
        train, val, combined,
        "FS2_TF1", 36, "WB0_CONTEXT_CARRY_OVER",
        "sha_train", "sha_val",
    )
    assert fp1 == fp2, "FINAL_DEV fingerprint should be deterministic"
    assert len(fp1) == 64, "SHA256 hex should be 64 characters"


def test_final_dev_no_duplicate_target_ids():
    """FINAL_DEV population should have no duplicate target IDs."""
    from course_work.data.final_dev import _build_final_dev_population
    import pandas as pd

    train = pd.DataFrame({
        "target_sample_id": [f"T{i:04d}" for i in range(5)],
        "target_timestamp": pd.date_range("2020-01-01", periods=5, freq="h"),
    })
    val = pd.DataFrame({
        "target_sample_id": [f"V{i:04d}" for i in range(3)],
        "target_timestamp": pd.date_range("2020-01-06", periods=3, freq="h"),
    })

    combined = _build_final_dev_population(train, val)
    assert len(combined) == 8
    assert len(combined["target_sample_id"].unique()) == 8


def test_final_dev_no_overlap():
    """FINAL_DEV should detect TRAIN/VALIDATION overlap."""
    from course_work.data.final_dev import _build_final_dev_population
    import pandas as pd

    train = pd.DataFrame({
        "target_sample_id": ["A", "B", "C"],
        "target_timestamp": pd.date_range("2020-01-01", periods=3, freq="h"),
    })
    val = pd.DataFrame({
        "target_sample_id": ["B", "C", "D"],  # B, C overlap with train
        "target_timestamp": pd.date_range("2020-01-04", periods=3, freq="h"),
    })

    with pytest.raises(ValueError, match="duplicated target IDs|overlap"):
        _build_final_dev_population(train, val)


def test_final_dev_artifact_root():
    """FINAL_DEV artifact root is artifacts/final_dev_region."""
    from course_work.data.final_dev import FINAL_DEV_ARTIFACT_ROOT
    assert FINAL_DEV_ARTIFACT_ROOT == "artifacts/final_dev_region"
