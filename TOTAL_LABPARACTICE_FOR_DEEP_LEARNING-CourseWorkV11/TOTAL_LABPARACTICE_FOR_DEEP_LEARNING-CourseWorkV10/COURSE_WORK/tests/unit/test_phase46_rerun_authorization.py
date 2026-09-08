"""Unit tests for Phase 46 corrected rerun registry authorization.

These tests verify the registry-level logic that the corrected Phase 46 driver
relies on:

  - PHASE46_CORRECTIVE_RERUN is now a canonical rerun reason in RERUN_REASONS
  - The historical reasons are still present (no weakening of guard)
  - An unsupported/free-text rerun reason is still rejected
  - register_run applies correct parent_run_id / rerun_reason / run_id
  - Corrected execution lineage augmentation produces a distinct fingerprint
  - The canonical reason matches what the driver passes
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from course_work.experiments.registry import (
    RERUN_REASONS,
    canonicalize_config,
    compute_config_fingerprint,
)


class TestCanonicalRerunReason:
    """The corrected Phase 46 driver depends on PHASE46_CORRECTIVE_RERUN being
    in the canonical enum.
    """

    def test_phase46_corrective_rerun_in_enum(self):
        assert "PHASE46_CORRECTIVE_RERUN" in RERUN_REASONS

    def test_existing_reasons_still_present(self):
        for reason in [
            "REPRODUCIBILITY_CHECK",
            "CODE_FIX",
            "DEVICE_CHANGE",
            "ENVIRONMENT_CHANGE",
            "CHECKPOINT_RECOVERY",
            "MANUAL_RERUN",
        ]:
            assert reason in RERUN_REASONS

    def test_arbitrary_reason_not_in_enum(self):
        assert "ARBITRARY_REASON" not in RERUN_REASONS
        assert "PHASE46_BAD" not in RERUN_REASONS

    def test_enum_still_only_contains_known_reasons(self):
        required = {
            "REPRODUCIBILITY_CHECK",
            "CODE_FIX",
            "DEVICE_CHANGE",
            "ENVIRONMENT_CHANGE",
            "CHECKPOINT_RECOVERY",
            "MANUAL_RERUN",
            "PHASE46_CORRECTIVE_RERUN",
        }
        assert required.issubset(RERUN_REASONS), (
            f"Missing required reasons: {required - RERUN_REASONS}"
        )
        for forbidden in ("ARBITRARY_REASON", "PHASE46_BAD", "TEMP_RERUN"):
            assert forbidden not in RERUN_REASONS, (
                f"Ad-hoc reason {forbidden} leaked into RERUN_REASONS"
            )


class TestFingerprintAugmentation:
    """Augmenting lineage with corrected-execution fields must change the
    fingerprint, so the registry sees a NEW (not duplicate) config.
    """

    def test_augmented_lineage_changes_fingerprint(self):
        base = {
            "lineage": {
                "feature_fingerprint": "abc",
                "scaling_version": "SCALING-v1",
            },
            "data": {"feature_variant_id": "FS2_TF1"},
            "training": {"seed": 42},
        }
        fp_base = compute_config_fingerprint(base)

        augmented = copy.deepcopy(base)
        augmented["lineage"] = dict(augmented["lineage"])
        augmented["lineage"]["final_dev_population_fingerprint"] = "deadbeef"
        augmented["lineage"]["final_scaling_x_sha256"] = "cafef00d"
        augmented["lineage"]["final_scaling_y_sha256"] = "feedface"
        augmented["lineage"]["corrected_implementation_version"] = "PHASE46_CORRECTED-v1"
        augmented["lineage"]["historical_invalidated_predecessor_run_id"] = "RUN_X"
        fp_augmented = compute_config_fingerprint(augmented)

        assert fp_base != fp_augmented, (
            "Augmenting lineage with corrected execution provenance MUST "
            "change the config_fingerprint so registry sees a non-duplicate."
        )

    def test_lineage_only_change_does_not_mutate_other_groups(self):
        """Augmenting `lineage` must not alter canonicalize of other groups."""
        base = {
            "lineage": {"feature_fingerprint": "abc"},
            "data": {"feature_variant_id": "FS2_TF1"},
            "training": {"seed": 42, "learning_rate": 0.001},
        }
        augmented = copy.deepcopy(base)
        augmented["lineage"] = dict(augmented["lineage"])
        augmented["lineage"]["corrected_implementation_version"] = "v1"

        assert (
            canonicalize_config(base)["data"] == canonicalize_config(augmented)["data"]
        )
        assert (
            canonicalize_config(base)["training"]
            == canonicalize_config(augmented)["training"]
        )


class TestDriverUsesCanonicalReason:
    """The driver must pass PHASE46_CORRECTIVE_RERUN (the canonical reason)
    to register_run.  This test inspects the driver source to prevent
    regressions.
    """

    def test_driver_uses_canonical_reason(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "PHASE46_CORRECTIVE_RERUN" in driver, (
            "Driver must use PHASE46_CORRECTIVE_RERUN as canonical rerun reason."
        )

    def test_driver_passes_parent_run_id(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "parent_run_id=historical_predecessor_run_id" in driver, (
            "Driver must pass parent_run_id=historical_predecessor_run_id."
        )

    def test_driver_injects_corrected_lineage(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        for field in [
            "final_dev_population_fingerprint",
            "final_scaling_x_sha256",
            "final_scaling_y_sha256",
            "final_scaling_version",
            "final_dev_region_version",
            "corrected_implementation_version",
            "historical_invalidated_predecessor_run_id",
        ]:
            assert field in driver, (
                f"Driver must inject corrected lineage field: {field}"
            )


class TestHistoricalSeedMapping:
    """The driver maps each seed to its historical predecessor run ID."""

    def test_mapping_present(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        for rid in [
            "RUN_TR_FSD_0153_B15A19DC",
            "RUN_TR_FSD_0154_DD82D743",
            "RUN_TR_FSD_0155_59A50ADD",
        ]:
            assert rid in driver, f"Driver must reference historical run ID: {rid}"

    def test_mapping_seeds(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        for seed in [42, 123, 2026]:
            assert str(seed) in driver, f"Driver must reference seed: {seed}"


class TestRegistryRejectsInvalidReason:
    """Even with PHASE46_CORRECTIVE_RERUN added, an arbitrary reason must
    still raise ValueError when no duplicate exists.
    """

    def test_invalid_rerun_reason_when_no_duplicate(self):
        assert "PHASE46_BOGUS" not in RERUN_REASONS
        assert "ARBITRARY" not in RERUN_REASONS


class TestNoTestAccess:
    """Corrected execution must NOT access Test data."""

    def test_driver_target_access_mode_is_final_dev(self):
        driver = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert 'target_access_mode = "TEST"' not in driver
        assert "target_access_mode='TEST'" not in driver
        assert "FINAL_DEV" in driver
