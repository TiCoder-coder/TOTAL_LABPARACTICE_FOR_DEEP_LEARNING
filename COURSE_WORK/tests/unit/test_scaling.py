import unittest

import numpy as np
import pandas as pd

from course_work.data.feature_sets import compute_feature_fingerprint, get_feature_list
from course_work.data.scaling import (
    PASSTHROUGH_FEATURES,
    SCALING_VERSION,
    build_scaling_policy,
    compute_scaler_statistics_fingerprint,
    fit_x_scaler_bundle,
    fit_y_scaler,
    inverse_transform_continuous_features,
    inverse_transform_target,
    transform_feature_variant,
    transform_target,
    validate_scaler_bundle,
)


class TrainOnlyScalingTest(unittest.TestCase):
    def build_train_frame(self, variant_id: str, row_count: int = 12) -> pd.DataFrame:
        features = get_feature_list(variant_id)
        index = np.arange(row_count, dtype=np.float64)
        values = {}
        for position, feature in enumerate(features):
            if feature == "hour_sin":
                values[feature] = np.sin(2.0 * np.pi * index / row_count)
            elif feature == "hour_cos":
                values[feature] = np.cos(2.0 * np.pi * index / row_count)
            elif feature == "dow_sin":
                values[feature] = np.sin(2.0 * np.pi * index / 7.0)
            elif feature == "dow_cos":
                values[feature] = np.cos(2.0 * np.pi * index / 7.0)
            elif feature == "weekend":
                values[feature] = (index % 7 >= 5).astype(np.float64)
            else:
                values[feature] = index * (position + 1.0) + position
        frame = pd.DataFrame(values, columns=features)
        frame["split_id"] = "TRAIN"
        frame["timestamp"] = pd.date_range("2026-01-01", periods=row_count, freq="10min")
        return frame

    def fit_bundle(self, variant_id: str = "FS1_TF1") -> tuple[pd.DataFrame, dict]:
        frame = self.build_train_frame(variant_id)
        features = get_feature_list(variant_id)
        fingerprint = compute_feature_fingerprint(variant_id, features)
        bundle = fit_x_scaler_bundle(frame, variant_id, fingerprint, "train-fingerprint", "global-fingerprint")
        return frame, bundle

    def test_policy_covers_six_variants_and_excludes_metadata(self) -> None:
        policy = build_scaling_policy()
        self.assertEqual(policy["scaling_version"], SCALING_VERSION)
        self.assertEqual(len(policy["variants"]), 6)
        for variant_id, variant in policy["variants"].items():
            self.assertEqual(variant["full_feature_order"], get_feature_list(variant_id))
            self.assertFalse(set(variant["scaled_feature_order"]).intersection(policy["metadata_only"]))
            self.assertEqual(
                variant["pass_through_feature_order"],
                [feature for feature in get_feature_list(variant_id) if feature in PASSTHROUGH_FEATURES],
            )

    def test_x_fit_rejects_non_train_rows(self) -> None:
        frame = self.build_train_frame("FS0_TF0")
        frame.loc[frame.index[-1], "split_id"] = "VALIDATION"
        features = get_feature_list("FS0_TF0")
        fingerprint = compute_feature_fingerprint("FS0_TF0", features)
        with self.assertRaises(ValueError):
            fit_x_scaler_bundle(frame, "FS0_TF0", fingerprint, "train", "global")

    def test_transform_preserves_order_passthrough_and_standardizes_train(self) -> None:
        frame, bundle = self.fit_bundle("FS1_TF1")
        features = bundle["full_feature_order"]
        transformed = transform_feature_variant(
            frame[features],
            bundle,
            "FS1_TF1",
            bundle["feature_fingerprint"],
            "global-fingerprint",
        )
        self.assertEqual(list(transformed.columns), features)
        scaled = transformed[bundle["scaled_feature_order"]].to_numpy()
        self.assertTrue(np.allclose(scaled.mean(axis=0), 0.0, atol=1e-10, rtol=0.0))
        self.assertTrue(np.allclose(scaled.std(axis=0, ddof=0), 1.0, atol=1e-10, rtol=0.0))
        self.assertTrue(np.array_equal(
            transformed[bundle["pass_through_feature_order"]].to_numpy(),
            frame[bundle["pass_through_feature_order"]].to_numpy(),
        ))
        recovered = inverse_transform_continuous_features(transformed[bundle["scaled_feature_order"]], bundle)
        self.assertTrue(np.allclose(
            recovered.to_numpy(),
            frame[bundle["scaled_feature_order"]].to_numpy(),
            atol=1e-10,
            rtol=1e-10,
        ))

    def test_transform_rejects_feature_order_variant_and_split_mismatch(self) -> None:
        frame, bundle = self.fit_bundle("FS1_TF1")
        features = bundle["full_feature_order"]
        with self.assertRaises(ValueError):
            transform_feature_variant(
                frame[list(reversed(features))],
                bundle,
                "FS1_TF1",
                bundle["feature_fingerprint"],
                "global-fingerprint",
            )
        with self.assertRaises(ValueError):
            validate_scaler_bundle(bundle, "FS0_TF1", bundle["feature_fingerprint"], "global-fingerprint")
        with self.assertRaises(ValueError):
            validate_scaler_bundle(bundle, "FS1_TF1", bundle["feature_fingerprint"], "different-split")

    def test_zero_variance_is_flagged_and_remains_finite(self) -> None:
        frame = self.build_train_frame("FS0_TF0")
        frame["lights"] = 5.0
        features = get_feature_list("FS0_TF0")
        fingerprint = compute_feature_fingerprint("FS0_TF0", features)
        bundle = fit_x_scaler_bundle(frame, "FS0_TF0", fingerprint, "train", "global")
        self.assertIn("lights", bundle["zero_variance_features"])
        position = bundle["scaled_feature_order"].index("lights")
        self.assertEqual(float(bundle["scaler"].scale_[position]), 1.0)
        transformed = transform_feature_variant(frame[features], bundle, "FS0_TF0", fingerprint, "global")
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertTrue(np.all(transformed["lights"].to_numpy() == 0.0))

    def test_target_options_are_train_guarded_and_roundtrip(self) -> None:
        values = np.array([10.0, 20.0, 40.0, 80.0])
        frame = pd.DataFrame({
            "Appliances": values,
            "split_id": ["TRAIN"] * len(values),
            "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="10min"),
        })
        y_bundle = fit_y_scaler(frame, "train", "global")
        scaled = transform_target(values, "YS1", y_bundle)
        recovered = inverse_transform_target(scaled, "YS1", y_bundle)
        self.assertTrue(np.allclose(recovered, values, atol=1e-10, rtol=1e-10))
        self.assertTrue(np.array_equal(transform_target(values, "YS0"), values))
        self.assertTrue(np.array_equal(inverse_transform_target(values, "YS0"), values))
        invalid = frame.copy(deep=True)
        invalid.loc[invalid.index[-1], "split_id"] = "TEST"
        with self.assertRaises(ValueError):
            fit_y_scaler(invalid, "train", "global")

    def test_scaler_statistics_are_deterministic(self) -> None:
        _, first = self.fit_bundle("FS2_TF1")
        _, second = self.fit_bundle("FS2_TF1")
        self.assertEqual(first["statistics_fingerprint"], second["statistics_fingerprint"])
        self.assertEqual(
            compute_scaler_statistics_fingerprint(first["scaled_feature_order"], first["scaler"]),
            first["statistics_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
