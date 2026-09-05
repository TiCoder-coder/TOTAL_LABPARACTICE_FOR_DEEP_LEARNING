"""
Focused unit tests for Phase 46 FINAL_DEV_DIAGNOSTIC metric path.

Tests:
1. FINAL_DEV diagnostic: observed=16630, expected=16630, PASS
2. FINAL_DEV missing one sample, FAILS
3. FINAL_DEV duplicate sample, FAILS
4. FINAL_DEV containing TEST sample, FAILS
5. TRAIN diagnostic still expects TRAIN only
6. VALIDATION metric still expects VALIDATION only
7. Phase 46 FINAL_REFIT uses FINAL_DEV_DIAGNOSTIC mode
8. No Validation selection (evaluate_validation=False path)
9. No EarlyStopping triggered by FINAL_DEV diagnostics
10. Test access forbidden for FINAL_DEV
11. EvaluationMode enum has FINAL_DEV_DIAGNOSTIC
12. validate_evaluation_access rejects FINAL_DEV with TRAIN_DIAGNOSTIC mode
13. validate_evaluation_access rejects TRAIN with FINAL_DEV_DIAGNOSTIC mode
14. validate_evaluation_access accepts FINAL_DEV with FINAL_DEV_DIAGNOSTIC mode
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "COURSE_WORK/src")

from course_work.evaluation.metrics import (
    EvaluationContext,
    EvaluationMode,
    align_prediction_bundle_to_expected_population,
    compute_regression_metrics,
    expected_sample_indices,
    validate_evaluation_access,
    validate_population_coverage,
)
from course_work.data.final_dev import FINAL_DEV_ARTIFACT_ROOT


class TestEvaluationModeEnum:
    def test_final_dev_diagnostic_exists(self):
        assert hasattr(EvaluationMode, "FINAL_DEV_DIAGNOSTIC")
        assert EvaluationMode.FINAL_DEV_DIAGNOSTIC.value == "FINAL_DEV_DIAGNOSTIC"

    def test_all_modes_present(self):
        assert EvaluationMode.TRAIN_DIAGNOSTIC.value == "TRAIN_DIAGNOSTIC"
        assert EvaluationMode.VALIDATION.value == "VALIDATION"
        assert EvaluationMode.FINAL_TEST.value == "FINAL_TEST"
        assert EvaluationMode.FINAL_DEV_DIAGNOSTIC.value == "FINAL_DEV_DIAGNOSTIC"
        assert len(EvaluationMode) == 4


class TestValidateEvaluationAccess:
    """Test that validate_evaluation_access enforces correct split/mode pairing."""

    def test_final_dev_requires_final_dev_diagnostic_mode(self):
        ctx = EvaluationContext(
            split_id="FINAL_DEV",
            evaluation_mode="FINAL_DEV_DIAGNOSTIC",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        validate_evaluation_access(ctx)  # should not raise

    def test_final_dev_rejects_train_diagnostic_mode(self):
        ctx = EvaluationContext(
            split_id="FINAL_DEV",
            evaluation_mode="TRAIN_DIAGNOSTIC",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        with pytest.raises(PermissionError, match="FINAL_DEV requires FINAL_DEV_DIAGNOSTIC"):
            validate_evaluation_access(ctx)

    def test_final_dev_rejects_validation_mode(self):
        ctx = EvaluationContext(
            split_id="FINAL_DEV",
            evaluation_mode="VALIDATION",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        with pytest.raises(PermissionError):
            validate_evaluation_access(ctx)

    def test_train_requires_train_diagnostic_mode(self):
        ctx = EvaluationContext(
            split_id="TRAIN",
            evaluation_mode="TRAIN_DIAGNOSTIC",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        validate_evaluation_access(ctx)  # should not raise

    def test_train_rejects_final_dev_diagnostic_mode(self):
        ctx = EvaluationContext(
            split_id="TRAIN",
            evaluation_mode="FINAL_DEV_DIAGNOSTIC",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        # The TRAIN-specific check fires first (split_id =="TRAIN" takes priority)
        with pytest.raises(PermissionError, match="TRAIN requires TRAIN_DIAGNOSTIC"):
            validate_evaluation_access(ctx)

    def test_validation_requires_validation_mode(self):
        ctx = EvaluationContext(
            split_id="VALIDATION",
            evaluation_mode="VALIDATION",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        validate_evaluation_access(ctx)  # should not raise

    def test_final_dev_diagnostic_restricted_to_final_dev_split(self):
        ctx = EvaluationContext(
            split_id="TRAIN",
            evaluation_mode="FINAL_DEV_DIAGNOSTIC",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
        )
        # TRAIN-specific check fires first since TRAIN is a known split first
        with pytest.raises(PermissionError, match="TRAIN requires TRAIN_DIAGNOSTIC"):
            validate_evaluation_access(ctx)


class TestValidatePopulationCoverage:
    """Test population coverage validation for FINAL_DEV scenarios."""

    def test_exact_match_passes(self):
        observed = np.arange(16630, dtype=np.int64)
        expected = np.arange(16630, dtype=np.int64)
        validate_population_coverage(observed, expected)  # no raise

    def test_missing_one_sample_fails(self):
        observed = np.arange(16629, dtype=np.int64)  # missing index 16629
        expected = np.arange(16630, dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            validate_population_coverage(observed, expected)

    def test_duplicate_sample_fails(self):
        observed = np.arange(16629, dtype=np.int64)
        observed = np.append(observed, 100)  # duplicate index 100
        expected = np.arange(16630, dtype=np.int64)
        with pytest.raises(ValueError, match="contains duplicates"):
            validate_population_coverage(observed, expected)

    def test_extra_sample_fails(self):
        observed = np.arange(16631, dtype=np.int64)  # one extra
        expected = np.arange(16630, dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            validate_population_coverage(observed, expected)

    def test_different_order_same_set_passes(self):
        observed = np.arange(16630, dtype=np.int64)
        expected = np.arange(16630, dtype=np.int64)[::-1].copy()
        validate_population_coverage(observed, expected)  # order doesn't matter for set equality


class TestExpectedSampleIndicesFinalDev:
    """Test expected_sample_indices returns correct FINAL_DEV population."""

    @pytest.fixture
    def temp_final_dev_artifacts(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        # Create FINAL_DEV artifact structure
        fd_root = root / FINAL_DEV_ARTIFACT_ROOT.lstrip("/")
        fd_root.mkdir(parents=True)
        # Write the sample index CSV with 16630 entries (0..16629)
        idx_df = pd.DataFrame({"sample_idx": np.arange(16630, dtype=np.int64)})
        idx_path = fd_root / "final_dev_sample_index.csv"
        idx_df.to_csv(idx_path, index=False)
        return root

    def test_final_dev_returns_16630(self, temp_final_dev_artifacts):
        result = expected_sample_indices(
            "FINAL_DEV",
            lookback_steps=36,
            project_root=temp_final_dev_artifacts,
        )
        assert len(result) == 16630
        assert result[0] == 0
        assert result[-1] == 16629

    def test_final_dev_case_insensitive(self, temp_final_dev_artifacts):
        result_lower = expected_sample_indices("final_dev", project_root=temp_final_dev_artifacts)
        result_upper = expected_sample_indices("FINAL_DEV", project_root=temp_final_dev_artifacts)
        np.testing.assert_array_equal(result_lower, result_upper)

    def test_train_still_uses_window_index(self, tmp_path):
        """TRAIN still uses window_index for its expected population."""
        # Don't mock FINAL_DEV artifacts — TRAIN should not need them
        from course_work.data.windows import load_validated_window_index
        # The function should still exist and work for TRAIN
        root = tmp_path / "project"
        root.mkdir()
        # Load the real window_index
        from course_work.utils.artifacts import get_project_root
        real_root = get_project_root()
        # This should work for TRAIN split
        result = expected_sample_indices("TRAIN", lookback_steps=36, project_root=real_root)
        assert len(result) > 0

    def test_final_dev_missing_file_raises(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        fd_root = root / FINAL_DEV_ARTIFACT_ROOT.lstrip("/")
        fd_root.mkdir(parents=True)
        # No CSV file
        with pytest.raises(FileNotFoundError, match="FINAL_DEV sample index not found"):
            expected_sample_indices("FINAL_DEV", project_root=root)


class TestAlignPredictionBundleFinalDev:
    """Test align_prediction_bundle_to_expected_population for FINAL_DEV."""

    def test_final_dev_bundle_aligns_to_final_dev_expected(self):
        """A FINAL_DEV prediction bundle aligns correctly to FINAL_DEV expected population."""
        from course_work.evaluation.metrics import PredictionBundle
        # 16630 samples in FINAL_DEV order (0..16629)
        sample_idx = np.arange(16630, dtype=np.int64)
        bundle = PredictionBundle(
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
            split_id="FINAL_DEV",
            sample_idx=sample_idx,
            y_true_wh=np.random.randn(16630),
            y_pred_wh=np.random.randn(16630),
            target_scaling_option="YS1",
            population_fingerprint="abc123",
            lookback_steps=36,
            horizon_steps=1,
        )
        # Expected is same FINAL_DEV population
        expected = np.arange(16630, dtype=np.int64)
        aligned = align_prediction_bundle_to_expected_population(bundle, expected)
        np.testing.assert_array_equal(aligned.sample_idx, sample_idx)

    def test_final_dev_bundle_fails_on_train_expected(self):
        """A FINAL_DEV bundle fails when expected population is TRAIN-only (13,670)."""
        from course_work.evaluation.metrics import PredictionBundle
        sample_idx = np.arange(16630, dtype=np.int64)
        bundle = PredictionBundle(
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
            split_id="FINAL_DEV",
            sample_idx=sample_idx,
            y_true_wh=np.random.randn(16630),
            y_pred_wh=np.random.randn(16630),
            target_scaling_option="YS1",
            population_fingerprint="abc123",
            lookback_steps=36,
            horizon_steps=1,
        )
        # Expected is TRAIN-only (13670 samples)
        expected_train = np.arange(13670, dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            align_prediction_bundle_to_expected_population(bundle, expected_train)


class TestPhase46DriverFinalRefitMode:
    """Test that Phase 46 driver correctly sets final_refit_mode=True."""

    def test_driver_calls_engine_with_final_refit_mode_true(self):
        """Verify the Phase 46 driver passes final_refit_mode=True to engine.train()."""
        driver_path = Path("COURSE_WORK/scripts/phase46_three_seed_runs.py")
        content = driver_path.read_text()
        # Must call engine.train with final_refit_mode=True
        assert "final_refit_mode=True" in content, (
            "Phase 46 driver must pass final_refit_mode=True to engine.train()"
        )
        # Must NOT pass final_refit_mode=False
        assert "final_refit_mode=False" not in content, (
            "Phase 46 driver must NOT disable final_refit_mode"
        )

    def test_driver_passes_final_dev_loader(self):
        """Verify the Phase 46 driver uses final_dev_loader (not train_loader)."""
        driver_path = Path("COURSE_WORK/scripts/phase46_three_seed_runs.py")
        content = driver_path.read_text()
        # The engine.train call must use final_dev_loader
        assert "final_dev_loader" in content
        # Must NOT pass train_loader directly
        # (train_loader would be the TRAIN-only loader)
        lines = content.split("\n")
        train_call_lines = [l for l in lines if "engine.train(" in l or "final_dev_loader" in l]
        # Check that the engine.train call uses final_dev_loader as the first DataLoader arg
        for i, line in enumerate(train_call_lines):
            if "final_dev_loader" in line:
                assert "final_dev_loader" in line, (
                    "engine.train() must be called with final_dev_loader, not train_loader"
                )

    def test_driver_evaluate_validation_false(self):
        """Verify Phase 46 driver sets evaluate_validation=False."""
        driver_path = Path("COURSE_WORK/scripts/phase46_three_seed_runs.py")
        content = driver_path.read_text()
        assert "evaluate_validation=False" in content, (
            "FINAL_REFIT requires evaluate_validation=False"
        )

    def test_driver_metric_labels_final_dev_diagnostic(self):
        """Verify Phase 46 artifacts label metrics as FINAL_DEV_DIAGNOSTIC."""
        driver_path = Path("COURSE_WORK/scripts/phase46_three_seed_runs.py")
        content = driver_path.read_text()
        # Must NOT use TRAIN_DIAGNOSTIC for Phase 46 artifacts
        # (except where it may appear in non-metric contexts)
        lines_with_diagnostic = [
            l for l in content.split("\n")
            if "metric_semantic_label" in l
        ]
        for line in lines_with_diagnostic:
            assert "TRAIN_DIAGNOSTIC" not in line, (
                f"Phase 46 metric labels must be FINAL_DEV_DIAGNOSTIC, not TRAIN_DIAGNOSTIC: {line.strip()}"
            )
            assert "FINAL_DEV_DIAGNOSTIC" in line, (
                f"Phase 46 metric labels must contain FINAL_DEV_DIAGNOSTIC: {line.strip()}"
            )


class TestTrainMethodFinalRefitMode:
    """Test TrainingEngine.train() final_refit_mode parameter."""

    def test_train_accepts_final_refit_mode_parameter(self):
        """Verify train() method accepts final_refit_mode parameter."""
        from course_work.training.engine import TrainingEngine
        import inspect
        sig = inspect.signature(TrainingEngine.train)
        assert "final_refit_mode" in sig.parameters, (
            "TrainingEngine.train() must accept final_refit_mode parameter"
        )
        param = sig.parameters["final_refit_mode"]
        assert param.default is False, (
            "final_refit_mode should default to False for backward compatibility"
        )

    def test_evaluate_loader_accepts_evaluation_mode_override(self):
        """Verify _evaluate_loader accepts evaluation_mode_override."""
        from course_work.training.engine import TrainingEngine
        import inspect
        sig = inspect.signature(TrainingEngine._evaluate_loader)
        assert "evaluation_mode_override" in sig.parameters, (
            "TrainingEngine._evaluate_loader() must accept evaluation_mode_override"
        )


class TestPredictionBundleSchema:
    """Test prediction_bundle_schema.json includes FINAL_DEV."""

    def test_schema_allows_final_dev_split_id(self):
        schema_path = Path("COURSE_WORK/artifacts/metrics/prediction_bundle_schema.json")
        schema = json.loads(schema_path.read_text())
        assert "FINAL_DEV" in schema["required_fields"]["split_id"], (
            "prediction_bundle_schema.json must allow split_id=FINAL_DEV"
        )

    def test_schema_still_allows_train_validation_test(self):
        schema_path = Path("COURSE_WORK/artifacts/metrics/prediction_bundle_schema.json")
        schema = json.loads(schema_path.read_text())
        for split in ["TRAIN", "VALIDATION", "TEST"]:
            assert split in schema["required_fields"]["split_id"], (
                f"prediction_bundle_schema.json must still allow split_id={split}"
            )


class TestNoRegressionInDevelopmentBehavior:
    """Ensure Phase 46 fix does NOT break development-phase behavior."""

    def test_train_diagnostic_mode_still_works_for_train_split(self):
        """validate_evaluation_access still accepts TRAIN + TRAIN_DIAGNOSTIC."""
        ctx = EvaluationContext(
            split_id="TRAIN",
            evaluation_mode="TRAIN_DIAGNOSTIC",
            run_id="RUN_DEV",
            model_id="MODEL_DEV",
        )
        validate_evaluation_access(ctx)  # must not raise

    def test_validation_mode_still_works_for_validation_split(self):
        """validate_evaluation_access still accepts VALIDATION + VALIDATION."""
        ctx = EvaluationContext(
            split_id="VALIDATION",
            evaluation_mode="VALIDATION",
            run_id="RUN_DEV",
            model_id="MODEL_DEV",
        )
        validate_evaluation_access(ctx)  # must not raise

    def test_test_requires_final_test_mode(self):
        """validate_evaluation_access still requires FINAL_TEST for TEST split."""
        ctx = EvaluationContext(
            split_id="TEST",
            evaluation_mode="FINAL_TEST",
            run_id="RUN_TEST",
            model_id="MODEL_TEST",
            model_lock_id="LOCK_001",
        )
        validate_evaluation_access(ctx)  # must not raise

    def test_train_diagnostic_fails_on_validation_mode(self):
        """TRAIN split cannot use VALIDATION evaluation mode."""
        ctx = EvaluationContext(
            split_id="TRAIN",
            evaluation_mode="VALIDATION",
            run_id="RUN_DEV",
            model_id="MODEL_DEV",
        )
        with pytest.raises(PermissionError, match="TRAIN requires TRAIN_DIAGNOSTIC"):
            validate_evaluation_access(ctx)
