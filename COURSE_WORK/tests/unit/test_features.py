import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from course_work.data.features import (
    ENGINEERED_FEATURES,
    METADATA_COLUMNS,
    add_time_features,
    build_feature_leakage_audit,
    build_feature_registry,
    build_feature_view,
    validate_feature_invariants,
)
from course_work.data.schema import load_raw_csv
from course_work.data.temporal import build_temporal_view
from course_work.utils.artifacts import read_json


class FeatureEngineeringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.schema_manifest = read_json(cls.root / "artifacts/schema/schema_manifest.json")
        cls.raw = load_raw_csv(cls.root / "data/raw_data/energydata_complete.csv")
        cls.temporal = build_temporal_view(cls.raw)

    def test_time_features_use_minute_resolution_and_cyclic_norm(self) -> None:
        timestamps = pd.to_datetime([
            "2026-01-05 00:00:00",
            "2026-01-05 00:10:00",
            "2026-01-05 23:50:00",
            "2026-01-06 00:00:00",
        ])
        source = pd.DataFrame({"timestamp": timestamps})
        output = add_time_features(source)
        self.assertEqual(list(source.columns), ["timestamp"])
        self.assertAlmostEqual(output.loc[1, "hour_sin"], np.sin(2.0 * np.pi * 10.0 / 1440.0))
        self.assertTrue(np.allclose(output["hour_sin"].pow(2) + output["hour_cos"].pow(2), 1.0))
        self.assertTrue(np.allclose(output["dow_sin"].pow(2) + output["dow_cos"].pow(2), 1.0))
        boundary_distance = np.linalg.norm(output.loc[2, ["hour_sin", "hour_cos"]].to_numpy() - output.loc[3, ["hour_sin", "hour_cos"]].to_numpy())
        self.assertLess(boundary_distance, 0.05)

    def test_weekend_mapping(self) -> None:
        timestamps = pd.to_datetime([
            "2026-01-09 12:00:00",
            "2026-01-10 12:00:00",
            "2026-01-11 12:00:00",
            "2026-01-12 12:00:00",
        ])
        output = add_time_features(pd.DataFrame({"timestamp": timestamps}))
        self.assertEqual(output["weekend"].tolist(), [0, 1, 1, 0])

    def test_feature_engineering_is_deterministic_and_rejects_reapplication(self) -> None:
        source = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=8, freq="10min")})
        first = add_time_features(source)
        second = add_time_features(source)
        pd.testing.assert_frame_equal(first, second)
        with self.assertRaisesRegex(ValueError, "already exist"):
            add_time_features(first)

    def test_canonical_feature_view_preserves_upstream_values_and_order(self) -> None:
        feature_view = build_feature_view(self.temporal, self.schema_manifest)
        expected_columns = [
            *METADATA_COLUMNS,
            "Appliances",
            *self.schema_manifest["regular_feature_columns"],
            *self.schema_manifest["random_control_columns"],
            *ENGINEERED_FEATURES,
        ]
        self.assertEqual(list(feature_view.columns), expected_columns)
        self.assertEqual(len(feature_view), len(self.temporal))
        self.assertTrue(feature_view["timestamp"].equals(self.temporal["timestamp_parsed"]))
        for column in ["Appliances", *self.schema_manifest["regular_feature_columns"], *self.schema_manifest["random_control_columns"]]:
            self.assertTrue(feature_view[column].equals(self.temporal[column]))
        audit = validate_feature_invariants(self.temporal, feature_view, self.schema_manifest)
        self.assertEqual({row["status"] for row in audit}, {"PASS"})

    def test_registry_and_leakage_contract(self) -> None:
        feature_view = build_feature_view(self.temporal.iloc[:20], self.schema_manifest)
        registry = pd.DataFrame(build_feature_registry(feature_view, self.schema_manifest)).set_index("column_name")
        leakage = pd.DataFrame(build_feature_leakage_audit(feature_view, self.schema_manifest)).set_index("feature_name")
        self.assertFalse(bool(registry.loc["raw_row_index", "model_eligible"]))
        self.assertFalse(bool(registry.loc["timestamp", "model_eligible"]))
        self.assertEqual(registry.loc["Appliances", "availability"], "TARGET_HISTORY_CANDIDATE")
        self.assertEqual(registry.loc["rv1", "availability"], "RANDOM_CONTROL")
        self.assertEqual(registry.loc["hour_sin", "availability"], "KNOWN_CALENDAR")
        self.assertEqual(set(leakage["status"]), {"PASS"})
        self.assertFalse(leakage["depends_on_future_target"].any())
        self.assertFalse(leakage["depends_on_full_dataset_statistics"].any())

    def test_invalid_timestamp_input_fails(self) -> None:
        source = pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"]})
        with self.assertRaisesRegex(ValueError, "datetime"):
            add_time_features(source)


if __name__ == "__main__":
    unittest.main()
