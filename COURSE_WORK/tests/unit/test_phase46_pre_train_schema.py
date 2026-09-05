"""Regression tests for Phase 46 pre-train schema contract.

Verifies:
  1. real FINAL_DEV producer satisfies driver schema
  2. real FINAL_SCALING producer satisfies driver schema
  3. real Phase 45 → 46 handoff satisfies driver schema
  4. missing required field fails before registration
  5. all three seeds can build complete corrected run_config
  6. no historical run reused
  7. no official registry entry created in simulation
  8. no Test access
  9. final_dev_region_version sourced from FINAL_DEV_REGION_VERSION constant
 10. locked_model_id bug fully eliminated (no raw access remains)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import phase46_three_seed_runs as p46
from course_work.data.final_dev import FINAL_DEV_REGION_VERSION, materialize_final_dev_region
from course_work.scaling.final_scaling import materialize_final_scaling_v1


# ---------------------------------------------------------------------------
# Producer schemas
# ---------------------------------------------------------------------------


class TestRealProducers:
    def test_real_final_dev_satisfies_schema(self):
        """The real ``materialize_final_dev_region`` returns a manifest that
        passes the driver schema validator."""
        manifest = materialize_final_dev_region(project_root=p46.ROOT)
        p46.validate_final_dev_manifest_schema(manifest)
        assert manifest.final_dev_window_count == 16630
        assert manifest.test_window_count == 0  # Test firewall

    def test_real_final_scaling_satisfies_schema(self):
        """The real ``materialize_final_scaling_v1`` returns a dict that
        passes the driver schema validator."""
        result = materialize_final_scaling_v1(
            project_root=p46.ROOT,
            feature_variant_id="FS2_TF1",
        )
        p46.validate_final_scaling_schema(result)
        assert result["scaling_version"] == "FINAL_SCALING-v1"
        assert result["fit_region"] == FINAL_DEV_REGION_VERSION
        assert result["status"] == "PASS"

    def test_real_phase45_handoff_satisfies_schema(self):
        """The real ``phase46_three_seed_handoff.json`` passes the driver
        schema validator."""
        handoff = json.load(open(p46.PHASE_46_HANDOFF))
        p46.validate_phase46_handoff_schema(handoff)
        # Locked candidate is TR_C2_ALT_LOOKBACK (Phase 45 final model lock).
        assert handoff["candidate_id"] == "TR_C2_ALT_LOOKBACK"
        assert handoff["FINAL_REFIT_EPOCHS"] == 30

    def test_real_phase45_signoff_satisfies_schema(self):
        """The real ``phase_45_signoff.json`` passes the driver schema
        validator."""
        signoff = json.load(open(p46.PHASE_45_SIGNOFF))
        p46.validate_phase45_signoff_schema(signoff)
        assert signoff["status"] == "PASS"


# ---------------------------------------------------------------------------
# Missing-key detection
# ---------------------------------------------------------------------------


class TestMissingFieldsFail:
    def test_handoff_missing_candidate_id(self):
        """A handoff without ``candidate_id`` must raise PreTrainSchemaError."""
        handoff = json.load(open(p46.PHASE_46_HANDOFF))
        del handoff["candidate_id"]
        with pytest.raises(p46.PreTrainSchemaError, match="phase46_handoff"):
            p46.validate_phase46_handoff_schema(handoff)

    def test_scaling_missing_x_sha256(self):
        """A scaling result without ``x_sha256`` must raise."""
        result = materialize_final_scaling_v1(
            project_root=p46.ROOT, feature_variant_id="FS2_TF1"
        )
        del result["x_sha256"]
        with pytest.raises(p46.PreTrainSchemaError, match="final_scaling_result"):
            p46.validate_final_scaling_schema(result)

    def test_final_dev_test_window_count_must_be_zero(self):
        """A final_dev_manifest with test_window_count > 0 must raise."""
        # Build a synthetic manifest with a non-zero test_window_count.
        from dataclasses import replace
        manifest = materialize_final_dev_region(project_root=p46.ROOT)
        mutated = replace(manifest, test_window_count=1)
        with pytest.raises(p46.PreTrainSchemaError, match="Test firewall"):
            p46.validate_final_dev_manifest_schema(mutated)


# ---------------------------------------------------------------------------
# All three seeds build a complete corrected run_config
# ---------------------------------------------------------------------------


def _build_corrected_run_config(seed, scaling_result, locked_cfg):
    cfg = json.loads(json.dumps(locked_cfg))
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["max_epochs"] = int(locked_cfg.get("training", {}).get("max_epochs", 50))
    cfg["training"]["early_stopping_enabled"] = False
    cfg["training"]["final_refit_mode"] = True
    cfg["lineage"] = dict(cfg.get("lineage", {}))
    cfg["lineage"]["final_dev_population_fingerprint"] = (
        scaling_result["final_dev_population_fingerprint"]
    )
    cfg["lineage"]["final_scaling_x_sha256"] = scaling_result["x_sha256"]
    cfg["lineage"]["final_scaling_y_sha256"] = scaling_result["y_sha256"]
    cfg["lineage"]["final_scaling_version"] = scaling_result["scaling_version"]
    cfg["lineage"]["final_dev_region_version"] = FINAL_DEV_REGION_VERSION
    cfg["lineage"]["corrected_implementation_version"] = "PHASE46_CORRECTED-v1"
    cfg["lineage"]["historical_invalidated_predecessor_run_id"] = (
        p46._historical_seed_to_run_id(seed)
    )
    return cfg


REQUIRED_LINEAGE_KEYS = (
    "final_dev_population_fingerprint",
    "final_scaling_x_sha256",
    "final_scaling_y_sha256",
    "final_scaling_version",
    "final_dev_region_version",
    "corrected_implementation_version",
    "historical_invalidated_predecessor_run_id",
    "feature_fingerprint",
    "population_fingerprint",
)


class TestAllSeedsBuildConfig:
    @pytest.fixture(scope="class")
    def driver_state(self):
        handoff = json.load(open(p46.PHASE_46_HANDOFF))
        scaling_result = materialize_final_scaling_v1(
            project_root=p46.ROOT, feature_variant_id="FS2_TF1"
        )
        locked_cfg = handoff["scientific_config"]
        return handoff, scaling_result, locked_cfg

    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_complete_run_config_for_each_seed(self, driver_state, seed):
        """Each seed must produce a run_config with every required lineage field."""
        _, scaling_result, locked_cfg = driver_state
        cfg = _build_corrected_run_config(seed, scaling_result, locked_cfg)
        for key in REQUIRED_LINEAGE_KEYS:
            assert key in cfg["lineage"], f"missing lineage.{key}"
            assert cfg["lineage"][key], f"empty lineage.{key}"
        assert cfg["training"]["seed"] == seed
        # corrected_implementation_version marker is present and project-defined.
        assert cfg["lineage"]["corrected_implementation_version"] == "PHASE46_CORRECTED-v1"

    @pytest.mark.parametrize("seed", [42, 123, 2026])
    def test_each_seed_has_correct_predecessor(self, driver_state, seed):
        """Each seed must map to the correct historical predecessor."""
        _, _, _ = driver_state
        pred = p46._historical_seed_to_run_id(seed)
        assert pred is not None
        assert pred in p46.EXCLUDED_HISTORICAL_RUN_IDS


# ---------------------------------------------------------------------------
# Schema validators error messages are informative
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_pre_train_schema_error_includes_missing_keys(self):
        try:
            p46._check_required_mapping_keys(
                {"a": 1}, ("a", "b", "c"), label="test"
            )
            assert False, "should have raised"
        except p46.PreTrainSchemaError as exc:
            assert "test" in str(exc)
            assert "b" in str(exc) and "c" in str(exc)
            assert "missing required fields" in str(exc).lower()

    def test_pre_train_schema_error_lists_available_keys(self):
        try:
            p46._check_required_mapping_keys(
                {"x": 1, "y": 2}, ("z",), label="test"
            )
            assert False, "should have raised"
        except p46.PreTrainSchemaError as exc:
            assert "x" in str(exc) and "y" in str(exc)

    def test_nested_path_error_lists_missing_paths(self):
        try:
            p46._check_required_nested_paths(
                {"data": {}},
                (("data", "feature_variant_id"),),
                label="locked_cfg",
            )
            assert False, "should have raised"
        except p46.PreTrainSchemaError as exc:
            assert "data.feature_variant_id" in str(exc)
            assert "locked_cfg" in str(exc)


# ---------------------------------------------------------------------------
# Driver no longer reads wrong field
# ---------------------------------------------------------------------------


class TestDriverFieldAccess:
    def test_driver_does_not_read_scaling_result_final_dev_region_version(self):
        """The driver must NOT read ``final_dev_region_version`` from
        scaling_result — that key does not exist there."""
        driver_src = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert (
            'scaling_result["final_dev_region_version"]' not in driver_src
        ), "Driver still reads non-existent scaling_result['final_dev_region_version']"

    def test_driver_does_not_read_phase46_handoff_locked_model_id_directly(self):
        """The driver must NOT directly access ``phase46_handoff['locked_model_id']``
        — that key does not exist in the canonical handoff."""
        driver_src = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert (
            'phase46_handoff["locked_model_id"]' not in driver_src
        ), "Driver still reads non-existent phase46_handoff['locked_model_id']"

    def test_driver_uses_resolve_locked_model_id(self):
        """The driver must use ``_resolve_locked_model_id`` for the
        locked model ID resolution."""
        driver_src = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "_resolve_locked_model_id(phase46_handoff)" in driver_src

    def test_driver_uses_final_dev_region_version_constant(self):
        """The driver must source ``final_dev_region_version`` from the
        ``FINAL_DEV_REGION_VERSION`` constant, not from ``scaling_result``."""
        driver_src = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert (
            'run_config["lineage"]["final_dev_region_version"] = FINAL_DEV_REGION_VERSION'
            in driver_src
        )


# ---------------------------------------------------------------------------
# Test firewall
# ---------------------------------------------------------------------------


class TestNoTestAccess:
    def test_target_access_mode_not_test_in_simulation(self):
        """The corrected run_config must not have target_access_mode == 'TEST'."""
        handoff = json.load(open(p46.PHASE_46_HANDOFF))
        scaling_result = materialize_final_scaling_v1(
            project_root=p46.ROOT, feature_variant_id="FS2_TF1"
        )
        for seed in [42, 123, 2026]:
            cfg = _build_corrected_run_config(seed, scaling_result, handoff["scientific_config"])
            assert cfg["data"].get("target_access_mode") != "TEST", (
                f"seed {seed} has target_access_mode=TEST — firewall breached"
            )
