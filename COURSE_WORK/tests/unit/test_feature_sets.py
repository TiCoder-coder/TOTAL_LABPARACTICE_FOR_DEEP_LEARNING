import unittest
from pathlib import Path

import numpy as np

from course_work.data.feature_sets import (
    BASELINE_VARIANT,
    EXPECTED_FEATURE_COUNTS,
    EXOGENOUS,
    HISTORICAL_TARGET,
    METADATA,
    RANDOM_CONTROLS,
    TIME_FEATURES,
    build_feature_variants,
    compute_feature_fingerprint,
    get_feature_list,
    load_phase_7_registries,
    pairwise_contracts_pass,
    validate_feature_variants,
)
from course_work.data.features import load_validated_feature_view


class FeatureSetVariantsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.variants = build_feature_variants()

    def test_all_six_variants_have_expected_counts_and_order(self) -> None:
        self.assertEqual({variant_id: len(features) for variant_id, features in self.variants.items()}, EXPECTED_FEATURE_COUNTS)
        for variant_id, features in self.variants.items():
            self.assertEqual(features[: len(EXOGENOUS)], EXOGENOUS)
            if variant_id.endswith("TF1"):
                self.assertEqual(features[-len(TIME_FEATURES) :], TIME_FEATURES)

    def test_target_random_control_time_and_metadata_isolation(self) -> None:
        for variant_id, features in self.variants.items():
            feature_set_id, time_feature_id = variant_id.split("_")
            self.assertEqual(HISTORICAL_TARGET[0] in features, feature_set_id in {"FS1", "FS2"})
            self.assertEqual(all(feature in features for feature in RANDOM_CONTROLS), feature_set_id == "FS2")
            self.assertEqual(all(feature in features for feature in TIME_FEATURES), time_feature_id == "TF1")
            self.assertFalse(set(features).intersection(METADATA))
            self.assertEqual(len(features), len(set(features)))

    def test_pairwise_ablation_contracts_pass(self) -> None:
        self.assertTrue(pairwise_contracts_pass(self.variants))
        self.assertEqual(set(self.variants["FS1_TF0"]) - set(self.variants["FS0_TF0"]), set(HISTORICAL_TARGET))
        self.assertEqual(set(self.variants["FS2_TF0"]) - set(self.variants["FS1_TF0"]), set(RANDOM_CONTROLS))
        for feature_set_id in ["FS0", "FS1", "FS2"]:
            self.assertEqual(set(self.variants[f"{feature_set_id}_TF1"]) - set(self.variants[f"{feature_set_id}_TF0"]), set(TIME_FEATURES))

    def test_registry_and_fingerprints_are_deterministic_and_defensive(self) -> None:
        second = build_feature_variants()
        self.assertEqual(self.variants, second)
        for variant_id, features in self.variants.items():
            self.assertEqual(compute_feature_fingerprint(variant_id, features), compute_feature_fingerprint(variant_id, second[variant_id]))
        candidate = get_feature_list(BASELINE_VARIANT)
        candidate.append("invalid")
        self.assertNotIn("invalid", get_feature_list(BASELINE_VARIANT))
        with self.assertRaisesRegex(KeyError, "Unknown feature variant"):
            get_feature_list("FS9_TF9")

    def test_actual_features_v1_passes_all_variant_audits(self) -> None:
        feature_view = load_validated_feature_view(self.root)
        registries = load_phase_7_registries(self.root)
        audit = validate_feature_variants(feature_view, registries, self.variants)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual({row["status"] for row in audit["leakage_rows"]}, {"PASS"})
        self.assertEqual({row["status"] for row in audit["order_rows"]}, {"PASS"})

    def test_matrix_probe_preserves_registered_channel_semantics(self) -> None:
        feature_view = load_validated_feature_view(self.root)
        probe = feature_view.iloc[:5]
        for variant_id, features in self.variants.items():
            matrix = probe[list(features)].to_numpy(dtype=float)
            self.assertEqual(matrix.shape, (5, EXPECTED_FEATURE_COUNTS[variant_id]))
            self.assertTrue(np.isfinite(matrix).all())
            for position in [0, len(features) // 2, len(features) - 1]:
                self.assertEqual(matrix[0, position], float(probe.iloc[0][features[position]]))


if __name__ == "__main__":
    unittest.main()
