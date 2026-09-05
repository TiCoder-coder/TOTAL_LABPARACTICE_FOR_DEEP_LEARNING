import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from course_work.data.scaling import load_validated_target_scaler
from course_work.evaluation.metrics import (
    EvaluationContext,
    EvaluationMode,
    PredictionBundle,
    aggregate_sample_weighted_loss,
    align_prediction_bundle_to_expected_population,
    compare_to_baseline,
    compute_mae_wh,
    compute_r2,
    compute_regression_metrics,
    compute_residual_arrays,
    compute_rmse_wh,
    convert_predictions_to_wh,
    normalize_regression_vector,
    validate_evaluation_access,
)


class SharedMetricsTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    def build_bundle(self) -> PredictionBundle:
        return PredictionBundle(
            run_id="RUN-v1",
            model_id="MODEL-v1",
            split_id="VALIDATION",
            sample_idx=np.array([3, 1, 2]),
            y_true_wh=np.array([30.0, 10.0, 20.0]),
            y_pred_wh=np.array([27.0, 12.0, 18.0]),
            target_scaling_option="YS0",
            population_fingerprint="population",
            lookback_steps=144,
            horizon_steps=1,
        )

    def test_reference_metrics_match_hand_calculation(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 27.0])
        self.assertAlmostEqual(compute_mae_wh(y_true, y_pred), 7.0 / 3.0)
        self.assertAlmostEqual(compute_rmse_wh(y_true, y_pred), math.sqrt(17.0 / 3.0))
        r2, status = compute_r2(y_true, y_pred)
        self.assertAlmostEqual(r2, 0.915)
        self.assertEqual(status, "DEFINED")

    def test_perfect_and_negative_r2_are_preserved(self) -> None:
        perfect, perfect_status = compute_r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        negative, negative_status = compute_r2([1.0, 2.0, 3.0], [10.0, 10.0, 10.0])
        self.assertEqual(perfect, 1.0)
        self.assertEqual(perfect_status, "DEFINED")
        self.assertLess(negative, 0.0)
        self.assertEqual(negative_status, "DEFINED")

    def test_r2_undefined_states_are_explicit(self) -> None:
        constant, constant_status = compute_r2([5.0, 5.0], [4.0, 6.0])
        single, single_status = compute_r2([5.0], [4.0])
        self.assertTrue(math.isnan(constant))
        self.assertEqual(constant_status, "UNDEFINED_CONSTANT_TARGET")
        self.assertTrue(math.isnan(single))
        self.assertEqual(single_status, "UNDEFINED_TOO_FEW_SAMPLES")

    def test_vector_normalization_accepts_only_single_output_shapes(self) -> None:
        expected = np.array([1.0, 2.0])
        self.assertTrue(np.array_equal(normalize_regression_vector(expected), expected))
        self.assertTrue(np.array_equal(normalize_regression_vector(expected[:, None]), expected))
        self.assertTrue(np.array_equal(normalize_regression_vector(torch.tensor(expected)), expected))
        with self.assertRaises(ValueError):
            normalize_regression_vector(np.ones((2, 2)))

    def test_metric_inputs_reject_length_and_finite_violations(self) -> None:
        with self.assertRaises(ValueError):
            compute_mae_wh([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            compute_rmse_wh([1.0, np.nan], [1.0, 2.0])
        with self.assertRaises(ValueError):
            compute_r2([1.0, 2.0], [1.0, np.inf])

    def test_population_alignment_restores_expected_order(self) -> None:
        aligned = align_prediction_bundle_to_expected_population(self.build_bundle(), [1, 2, 3])
        self.assertTrue(np.array_equal(aligned.sample_idx, [1, 2, 3]))
        self.assertTrue(np.array_equal(aligned.y_true_wh, [10.0, 20.0, 30.0]))
        self.assertTrue(np.array_equal(aligned.y_pred_wh, [12.0, 18.0, 27.0]))

    def test_population_alignment_rejects_duplicates_and_partial_split(self) -> None:
        duplicate = replace(self.build_bundle(), sample_idx=np.array([1, 1, 2]))
        with self.assertRaises(ValueError):
            align_prediction_bundle_to_expected_population(duplicate, [1, 2, 3])
        with self.assertRaises(ValueError):
            align_prediction_bundle_to_expected_population(self.build_bundle(), [1, 2])

    def test_regression_metrics_use_full_aligned_validation_population(self) -> None:
        result = compute_regression_metrics(
            y_true_wh=[30.0, 10.0, 20.0],
            y_pred_wh=[27.0, 12.0, 18.0],
            sample_idx=[3, 1, 2],
            split_id="VALIDATION",
            evaluation_mode=EvaluationMode.VALIDATION.value,
            population_fingerprint="population",
            run_id="RUN-v1",
            model_id="MODEL-v1",
            expected_sample_idx=[1, 2, 3],
        )
        self.assertEqual(result.n_samples, 3)
        self.assertEqual(result.selection_metric_name, "rmse_wh")
        self.assertAlmostEqual(result.selection_metric_value, math.sqrt(17.0 / 3.0))
        self.assertEqual(result.status, "PASS")

    def test_evaluation_modes_enforce_test_firewall(self) -> None:
        with self.assertRaises(PermissionError):
            validate_evaluation_access(EvaluationContext("TEST", EvaluationMode.VALIDATION.value, "RUN", "MODEL"))
        with self.assertRaises(PermissionError):
            validate_evaluation_access(EvaluationContext("TEST", EvaluationMode.FINAL_TEST.value, "RUN", "MODEL"))
        validate_evaluation_access(EvaluationContext("TEST", EvaluationMode.FINAL_TEST.value, "RUN", "MODEL", "LOCK-v1"))

    def test_target_scale_conversion_supports_ys0_and_frozen_ys1(self) -> None:
        values = np.array([10.0, 60.0, 250.0])
        self.assertTrue(np.array_equal(convert_predictions_to_wh(values, "YS0"), values))
        bundle = load_validated_target_scaler(self.root)
        scaled = bundle["scaler"].transform(values.reshape(-1, 1)).reshape(-1)
        restored = convert_predictions_to_wh(scaled, "YS1", self.root, bundle)
        self.assertTrue(np.allclose(restored, values))

    def test_residual_convention_is_true_minus_prediction(self) -> None:
        arrays = compute_residual_arrays([10.0, 20.0], [12.0, 18.0])
        self.assertTrue(np.array_equal(arrays["residual"], [-2.0, 2.0]))
        self.assertTrue(np.array_equal(arrays["prediction_error"], [2.0, -2.0]))
        self.assertTrue(np.array_equal(arrays["absolute_error"], [2.0, 2.0]))
        self.assertTrue(np.array_equal(arrays["squared_error"], [4.0, 4.0]))

    def test_epoch_loss_is_weighted_by_sample_count(self) -> None:
        self.assertAlmostEqual(aggregate_sample_weighted_loss([2.0, 4.0], [2, 1]), 8.0 / 3.0)
        with self.assertRaises(ValueError):
            aggregate_sample_weighted_loss([1.0, 2.0], [1])
        with self.assertRaises(ValueError):
            aggregate_sample_weighted_loss([1.0], [0])

    def test_baseline_comparison_requires_matching_contract_and_population(self) -> None:
        baseline = compute_regression_metrics(
            [10.0, 20.0, 30.0],
            [15.0, 15.0, 15.0],
            [1, 2, 3],
            "VALIDATION",
            EvaluationMode.VALIDATION.value,
            "population",
            "BASELINE",
            "PERSISTENCE",
            [1, 2, 3],
        )
        model = compute_regression_metrics(
            [10.0, 20.0, 30.0],
            [12.0, 18.0, 27.0],
            [1, 2, 3],
            "VALIDATION",
            EvaluationMode.VALIDATION.value,
            "population",
            "MODEL",
            "TRANSFORMER",
            [1, 2, 3],
        )
        comparison = compare_to_baseline(baseline, model)
        self.assertGreater(comparison["delta_rmse_wh"], 0.0)
        with self.assertRaises(ValueError):
            compare_to_baseline(baseline, replace(model, population_fingerprint="other"))


if __name__ == "__main__":
    unittest.main()
