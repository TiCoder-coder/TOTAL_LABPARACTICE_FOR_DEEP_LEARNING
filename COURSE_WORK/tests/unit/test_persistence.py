import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from course_work.baselines.persistence import (
    PersistenceConfig,
    build_persistence_prediction_frame,
    build_persistence_prediction_bundle,
    build_persistence_manifest,
    build_persistence_run_config,
    evaluate_persistence_validation,
    complete_persistence_run,
    prepare_validation_persistence_data,
    persistence_contract_payload,
    predict_persistence,
    register_and_start_persistence_run,
    validate_h1_alignment,
    validate_lookback_invariance,
    validate_persistence_inputs,
    validate_persistence_predictions,
    validate_population,
    verify_phase_14_inputs,
    write_and_register_persistence_artifacts,
)
from course_work.experiments.registry import ExperimentRegistry


class PersistenceCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def window_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_id": ["W1", "W2", "W3"],
                "target_sample_id": ["T1", "T2", "T3"],
                "lookback_steps": [144, 144, 144],
                "horizon_steps": [1, 1, 1],
                "timeline_input_end": [1, 2, 3],
                "timeline_target": [2, 3, 4],
                "input_end_timestamp": pd.to_datetime(["2020-01-01 00:10:00", "2020-01-01 00:20:00", "2020-01-01 00:30:00"]),
                "target_timestamp": pd.to_datetime(["2020-01-01 00:20:00", "2020-01-01 00:30:00", "2020-01-01 00:40:00"]),
                "input_end_raw_row_index": [1, 2, 3],
                "target_raw_row_index": [2, 3, 4],
                "continuity_segment_id": ["S1", "S1", "S1"],
                "target_split_id": ["VALIDATION", "VALIDATION", "VALIDATION"],
                "WB0_valid": [True, True, True],
                "included_common_population": [True, True, True],
            },
            index=[10, 11, 12],
        )

    def test_contract_is_parameter_free_task_level_h1(self) -> None:
        contract = persistence_contract_payload(PersistenceConfig())
        self.assertEqual(contract["model_id"], "PERSISTENCE_LAST_VALUE")
        self.assertEqual(contract["baseline_scope"], "TASK_LEVEL")
        self.assertEqual(contract["forecast_horizon_steps"], 1)
        self.assertEqual(contract["source_lag_minutes"], 10)
        self.assertEqual(contract["trainable_parameters"], 0)
        self.assertFalse(contract["requires_training"])
        self.assertIsNone(contract["seed"])
        self.assertFalse(contract["test_access_authorized"])

    def test_constant_increasing_decreasing_and_varying_series(self) -> None:
        cases = (
            np.asarray([5.0, 5.0, 5.0, 5.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([4.0, 3.0, 2.0, 1.0]),
            np.asarray([8.0, 2.0, 9.0, 3.0]),
        )
        for values in cases:
            with self.subTest(values=values.tolist()):
                actual = predict_persistence([1, 2, 3], [0, 1, 2], values)
                np.testing.assert_array_equal(actual, values[[0, 1, 2]])

    def test_prediction_is_deterministic_and_returns_independent_float64_vector(self) -> None:
        values = np.asarray([10, 20, 30, 40], dtype=np.int64)
        first = predict_persistence([1, 2, 3], [0, 1, 2], values)
        second = predict_persistence([1, 2, 3], [0, 1, 2], values)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float64)
        first[0] = -1.0
        self.assertEqual(values[0], 10)

    def test_prediction_rejects_wrong_lag_bounds_duplicates_and_non_finite_values(self) -> None:
        values = np.asarray([10.0, 20.0, 30.0, 40.0])
        with self.assertRaises(ValueError):
            predict_persistence([2, 3], [0, 1], values)
        with self.assertRaises(IndexError):
            predict_persistence([3, 4], [2, 3], values)
        with self.assertRaises(ValueError):
            predict_persistence([1, 1], [0, 0], values)
        with self.assertRaises(ValueError):
            predict_persistence([1, 2], [0, 1], [10.0, np.nan, 30.0])

    def test_h1_alignment_and_input_contract_pass(self) -> None:
        frame = self.window_frame()
        inputs = validate_persistence_inputs(frame, [10.0, 20.0, 30.0, 40.0, 50.0])
        alignment = validate_h1_alignment(frame)
        self.assertEqual(inputs["sample_count"], 3)
        self.assertEqual(alignment["horizon_minutes"], 10)
        self.assertEqual(alignment["status"], "PASS")

    def test_h1_alignment_rejects_gap_wrong_split_and_target_leakage(self) -> None:
        gap = self.window_frame()
        gap.loc[11, "target_timestamp"] = pd.Timestamp("2020-01-01 00:50:00")
        with self.assertRaises(ValueError):
            validate_h1_alignment(gap)
        wrong_split = self.window_frame()
        wrong_split.loc[11, "target_split_id"] = "TEST"
        with self.assertRaises(PermissionError):
            validate_h1_alignment(wrong_split)
        leakage = self.window_frame()
        leakage.loc[11, "timeline_input_end"] = leakage.loc[11, "timeline_target"]
        with self.assertRaises(ValueError):
            validate_h1_alignment(leakage)

    def test_population_requires_exact_unique_order_and_fingerprint(self) -> None:
        result = validate_population([10, 11, 12], [10, 11, 12], "POP", "POP")
        self.assertEqual(result["sample_count"], 3)
        with self.assertRaises(ValueError):
            validate_population([10, 12, 11], [10, 11, 12], "POP", "POP")
        with self.assertRaises(ValueError):
            validate_population([10, 10, 12], [10, 11, 12], "POP", "POP")
        with self.assertRaises(ValueError):
            validate_population([10, 11, 12], [10, 11, 12], "OTHER", "POP")

    def test_prediction_validation_requires_exact_raw_source_values(self) -> None:
        result = validate_persistence_predictions([10.0, 20.0], [10.0, 20.0])
        self.assertTrue(result["predictions_equal_raw_source"])
        with self.assertRaises(ValueError):
            validate_persistence_predictions([10.0, 21.0], [10.0, 20.0])

    def test_phase_14_upstream_contracts_are_valid(self) -> None:
        context = verify_phase_14_inputs(self.root)
        self.assertEqual(context["window_manifest"]["population_version"], "WINDOWPOP-v1")
        self.assertEqual(context["metric_manifest"]["metric_version"], "METRICS-v1")
        self.assertEqual(context["registry_manifest"]["experiment_registry_version"], "EXPERIMENTS-v1")

    def test_real_validation_population_is_complete_aligned_and_test_bounded(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        self.assertEqual(prepared.validation_sample_count, 2960)
        self.assertEqual(prepared.population_fingerprint, "a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987")
        self.assertEqual(len(prepared.raw_timeline), prepared.maximum_validation_raw_row_index + 1)
        self.assertLess(prepared.maximum_validation_raw_row_index, prepared.minimum_test_raw_row_index)
        self.assertTrue(prepared.window_metadata["target_split_id"].eq("VALIDATION").all())
        np.testing.assert_array_equal(prepared.y_pred_wh, prepared.source_values_wh)

    def test_real_prediction_frame_has_locked_schema_and_population(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        frame = build_persistence_prediction_frame(prepared, "PERSISTENCE_TEST_ONLY")
        self.assertEqual(
            frame.columns.tolist(),
            [
                "run_id",
                "sample_idx",
                "window_id",
                "target_timestamp",
                "source_timestamp",
                "source_raw_row_index",
                "target_raw_row_index",
                "y_true_wh",
                "y_pred_wh",
                "residual_wh",
                "absolute_error_wh",
                "squared_error_wh",
            ],
        )
        self.assertEqual(len(frame), 2960)
        self.assertFalse(frame["sample_idx"].duplicated().any())
        np.testing.assert_array_equal(frame["target_raw_row_index"] - frame["source_raw_row_index"], np.ones(2960, dtype=np.int64))
        np.testing.assert_allclose(frame["squared_error_wh"], np.square(frame["residual_wh"]), rtol=0.0, atol=0.0)

    def test_persistence_run_registers_before_running_with_locked_config(self) -> None:
        config = build_persistence_run_config(self.root)
        with tempfile.TemporaryDirectory(prefix="phase14-run-test-") as directory:
            temporary_root = Path(directory)
            registry = ExperimentRegistry(self.root, temporary_root / "registry", temporary_root / "runs")
            started = register_and_start_persistence_run(registry, config)
            self.assertEqual(started["status"], "RUNNING")
            self.assertEqual(started["experiment_family"], "PERSISTENCE_BASELINE")
            self.assertEqual(started["execution_type"], "EVALUATION")
            self.assertFalse(started["test_access_authorized"])
            self.assertIsNotNone(started["started_at"])
            self.assertIsNone(started["config"]["data"]["feature_variant_id"])
            self.assertIsNone(started["config"]["reproducibility"]["seed"])

    def test_validation_metrics_use_shared_contract_and_raw_wh_identity(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        bundle = build_persistence_prediction_bundle(prepared, "PERSISTENCE_TEST_ONLY")
        evaluation = evaluate_persistence_validation(prepared, bundle.run_id, self.root)
        self.assertEqual(bundle.target_scaling_option, "YS0")
        self.assertEqual(evaluation.metric_result.metric_version, "METRICS-v1")
        self.assertEqual(evaluation.metric_result.split_id, "VALIDATION")
        self.assertEqual(evaluation.metric_result.n_samples, 2960)
        self.assertEqual(evaluation.metric_result.target_unit, "Wh")
        self.assertGreaterEqual(evaluation.metric_result.mae_wh, 0.0)
        self.assertGreaterEqual(evaluation.metric_result.rmse_wh, 0.0)
        self.assertTrue(all(row["status"] == "PASS" for row in evaluation.audit_rows))
        self.assertTrue(all(row["status"] == "PASS" for row in evaluation.unit_test_rows))

    def test_real_predictions_are_invariant_across_common_lookbacks(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        result = validate_lookback_invariance(prepared, self.root)
        self.assertEqual(result["lookbacks"], {"L36": True, "L72": True, "L144": True})

    def test_artifacts_and_metrics_register_with_checksums_before_completion(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        config = build_persistence_run_config(self.root)
        with tempfile.TemporaryDirectory(prefix="phase14-artifact-test-") as directory:
            temporary_root = Path(directory)
            registry = ExperimentRegistry(self.root, temporary_root / "registry", temporary_root / "runs")
            started = register_and_start_persistence_run(registry, config)
            evaluation = evaluate_persistence_validation(prepared, started["run_id"], self.root)
            result = write_and_register_persistence_artifacts(
                registry,
                temporary_root / "registry/baselines/persistence",
                prepared,
                evaluation,
                started["run_id"],
                "2026-08-16T00:00:00+00:00",
            )
            record = registry.get_run(started["run_id"])
            self.assertEqual(record["status"], "RUNNING")
            self.assertEqual(len(record["metrics"]), 3)
            self.assertEqual({row["metric_name"] for row in record["metrics"]}, {"mae_wh", "rmse_wh", "r2"})
            self.assertIn("PREDICTIONS", {row["artifact_type"] for row in record["artifacts"]})
            self.assertIn("METRICS", {row["artifact_type"] for row in record["artifacts"]})
            self.assertEqual(len(result["checksums"]), 7)
            self.assertTrue(all(path.is_file() for path in result["paths"].values()))
            self.assertEqual(result["metrics_payload"]["metric_result"]["n_samples"], 2960)

    def test_isolated_run_completes_and_manifest_preserves_contract(self) -> None:
        prepared = prepare_validation_persistence_data(self.root)
        config = build_persistence_run_config(self.root)
        context = verify_phase_14_inputs(self.root)
        with tempfile.TemporaryDirectory(prefix="phase14-completion-test-") as directory:
            temporary_root = Path(directory)
            registry = ExperimentRegistry(self.root, temporary_root / "registry", temporary_root / "runs")
            started = register_and_start_persistence_run(registry, config)
            evaluation = evaluate_persistence_validation(prepared, started["run_id"], self.root)
            artifacts = write_and_register_persistence_artifacts(
                registry,
                temporary_root / "registry/baselines/persistence",
                prepared,
                evaluation,
                started["run_id"],
                "2026-08-16T00:00:00+00:00",
            )
            completed = complete_persistence_run(registry, started["run_id"], evaluation)
            manifest = build_persistence_manifest(temporary_root, context, completed, prepared, evaluation, artifacts)
            self.assertEqual(completed["status"], "COMPLETED")
            self.assertIsNone(completed["best_epoch"])
            self.assertEqual(manifest["run_id"], completed["run_id"])
            self.assertEqual(manifest["audit_status"], "PASS")
            self.assertFalse(manifest["test_access_authorized"])
            self.assertFalse(manifest["test_targets_materialized"])
            self.assertEqual(manifest["validation_sample_count"], 2960)


if __name__ == "__main__":
    unittest.main()
