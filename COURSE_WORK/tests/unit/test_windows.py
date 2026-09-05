import unittest

import numpy as np
import pandas as pd

from course_work.data.windows import (
    LOOKBACK_OPTIONS,
    WINDOW_INDEX_COLUMNS,
    build_common_target_population,
    build_native_window_index,
    build_relative_lag_axis,
    build_timeline_positions,
    compute_population_fingerprint,
    compute_window_bounds,
    compute_window_fingerprint,
    filter_common_window_index,
    make_target_sample_id,
    make_window_id,
    materialize_window,
    validate_temporal_window,
)


class WindowBuilderTest(unittest.TestCase):
    @staticmethod
    def timeline(
        timestamps: pd.DatetimeIndex,
        split_ids: list[str] | None = None,
        segments: list[str] | None = None,
    ) -> pd.DataFrame:
        size = len(timestamps)
        return pd.DataFrame({
            "timeline_position": np.arange(size),
            "raw_row_index": np.arange(size),
            "timestamp": timestamps,
            "continuity_segment_id": segments or ["SEG-0001"] * size,
            "split_id": split_ids or ["TRAIN"] * size,
        })

    def test_bounds_identifiers_and_relative_lags(self) -> None:
        self.assertEqual(compute_window_bounds(144, 144), (0, 143, 144))
        self.assertEqual(make_target_sample_id(144), "TGT_00000144")
        self.assertEqual(make_window_id(144, 1, 144), "WIN_L144_H01_TGT_00000144")
        lags = build_relative_lag_axis(144)
        self.assertEqual(len(lags), 144)
        self.assertEqual(int(lags[0]), 1440)
        self.assertEqual(int(lags[-1]), 10)

    def test_contiguous_native_index_has_correct_geometry(self) -> None:
        timeline = self.timeline(pd.date_range("2026-01-01", periods=10, freq="10min"))
        valid, rejected = build_native_window_index(timeline, 3)
        self.assertEqual(len(valid), 7)
        self.assertEqual(len(rejected), 3)
        first = valid.iloc[0]
        self.assertEqual(first["timeline_input_start"], 0)
        self.assertEqual(first["timeline_input_end"], 2)
        self.assertEqual(first["timeline_target"], 3)
        self.assertLess(first["timeline_input_end"], first["timeline_target"])
        self.assertEqual(rejected["reason"].unique().tolist(), ["INSUFFICIENT_HISTORY"])

    def test_temporal_rejection_taxonomy(self) -> None:
        input_gap = self.timeline(pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:30:00",
            "2026-01-01 00:40:00",
        ]))
        target_gap = self.timeline(pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:20:00",
            "2026-01-01 00:40:00",
        ]))
        duplicate = self.timeline(pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:20:00",
        ]))
        off_grid = self.timeline(pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:25:00",
            "2026-01-01 00:35:00",
        ]))
        self.assertEqual(validate_temporal_window(input_gap, 3, 3)[1], "INPUT_GAP")
        self.assertEqual(validate_temporal_window(target_gap, 3, 3)[1], "TARGET_GAP")
        self.assertEqual(validate_temporal_window(duplicate, 3, 3)[1], "DUPLICATE_TIMESTAMP")
        self.assertEqual(validate_temporal_window(off_grid, 3, 3)[1], "OFF_GRID_TIMESTAMP")

    def test_continuity_segment_mismatch_is_rejected(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=4, freq="10min")
        timeline = self.timeline(timestamps, segments=["SEG-1", "SEG-1", "SEG-2", "SEG-2"])
        valid, reason, _ = validate_temporal_window(timeline, 3, 3)
        self.assertFalse(valid)
        self.assertEqual(reason, "INPUT_GAP")

    def test_wb0_carry_over_and_wb1_eligibility(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=12, freq="10min")
        splits = ["TRAIN"] * 8 + ["VALIDATION"] * 2 + ["TEST"] * 2
        valid, _ = build_native_window_index(self.timeline(timestamps, splits), 3)
        first_validation = valid.loc[valid["target_split_id"].eq("VALIDATION")].iloc[0]
        first_test = valid.loc[valid["target_split_id"].eq("TEST")].iloc[0]
        self.assertTrue(first_validation["WB0_valid"])
        self.assertFalse(first_validation["WB1_valid"])
        self.assertTrue(first_validation["crosses_split_boundary"])
        self.assertTrue(first_test["WB0_valid"])
        self.assertFalse(first_test["WB1_valid"])

    def test_future_split_input_for_train_fails(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=5, freq="10min")
        timeline = self.timeline(timestamps, ["VALIDATION", "VALIDATION", "VALIDATION", "TRAIN", "TRAIN"])
        with self.assertRaises(RuntimeError):
            build_native_window_index(timeline, 3)

    def test_common_population_aligns_every_lookback(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=220, freq="10min")
        splits = ["TRAIN"] * 170 + ["VALIDATION"] * 25 + ["TEST"] * 25
        timeline = self.timeline(timestamps, splits)
        native = {lookback: build_native_window_index(timeline, lookback)[0] for lookback in LOOKBACK_OPTIONS}
        population = build_common_target_population(native, timeline)
        common_index = filter_common_window_index(native, population)
        expected = set(population["target_sample_id"])
        for lookback in LOOKBACK_OPTIONS:
            actual = set(common_index.loc[common_index["lookback_steps"].eq(lookback), "target_sample_id"])
            self.assertEqual(actual, expected)
        self.assertEqual(population.iloc[0]["timeline_target"], max(LOOKBACK_OPTIONS))
        self.assertEqual(list(common_index.columns), WINDOW_INDEX_COLUMNS)

    def test_fingerprints_are_deterministic(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=220, freq="10min")
        timeline = self.timeline(timestamps)
        native_a = {lookback: build_native_window_index(timeline, lookback)[0] for lookback in LOOKBACK_OPTIONS}
        native_b = {lookback: build_native_window_index(timeline, lookback)[0] for lookback in LOOKBACK_OPTIONS}
        population_a = build_common_target_population(native_a, timeline)
        population_b = build_common_target_population(native_b, timeline)
        index_a = filter_common_window_index(native_a, population_a)
        index_b = filter_common_window_index(native_b, population_b)
        self.assertEqual(compute_population_fingerprint(population_a), compute_population_fingerprint(population_b))
        self.assertEqual(compute_window_fingerprint(index_a), compute_window_fingerprint(index_b))

    def test_lazy_materialization_preserves_sequence_shape_and_direction(self) -> None:
        matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
        record = {
            "window_id": "WIN_L003_H01_TGT_00000003",
            "target_sample_id": "TGT_00000003",
            "lookback_steps": 3,
            "horizon_steps": 1,
            "timeline_input_start": 0,
            "timeline_input_end": 2,
            "timeline_target": 3,
            "input_start_timestamp": "2026-01-01 00:00:00",
            "input_end_timestamp": "2026-01-01 00:20:00",
            "target_timestamp": "2026-01-01 00:30:00",
            "target_split_id": "TRAIN",
        }
        target = np.arange(10, dtype=np.float64)
        result = materialize_window(matrix, record, target)
        self.assertEqual(result["X_seq"].shape, (3, 3))
        self.assertEqual(result["X_seq"].dtype, np.float32)
        np.testing.assert_array_equal(result["X_seq"], matrix[:3])
        self.assertEqual(result["y_model"].shape, (1,))
        self.assertEqual(result["y_raw_wh"].shape, (1,))
        self.assertEqual(float(result["y_raw_wh"][0]), 3.0)

    def test_test_targets_are_locked_by_default(self) -> None:
        matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
        record = {
            "window_id": "WIN_L003_H01_TGT_00000003",
            "target_sample_id": "TGT_00000003",
            "lookback_steps": 3,
            "horizon_steps": 1,
            "timeline_input_start": 0,
            "timeline_input_end": 2,
            "timeline_target": 3,
            "input_start_timestamp": "2026-01-01 00:00:00",
            "input_end_timestamp": "2026-01-01 00:20:00",
            "target_timestamp": "2026-01-01 00:30:00",
            "target_split_id": "TEST",
        }
        locked = materialize_window(matrix, record, np.arange(10, dtype=np.float64))
        self.assertIsNone(locked["y_model"])
        self.assertIsNone(locked["y_raw_wh"])
        unlocked = materialize_window(
            matrix,
            record,
            np.arange(10, dtype=np.float64),
            allow_test_targets=True,
        )
        self.assertEqual(unlocked["y_model"].shape, (1,))

    def test_timeline_alignment_rejects_mismatched_rows(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=4, freq="10min")
        feature_view = pd.DataFrame({
            "raw_row_index": [0, 1, 2, 3],
            "timestamp": timestamps,
            "continuity_segment_id": ["SEG-1"] * 4,
        })
        membership = pd.DataFrame({
            "raw_row_index": [0, 1, 3, 2],
            "timestamp": timestamps,
            "continuity_segment_id": ["SEG-1"] * 4,
            "split_id": ["TRAIN"] * 4,
        })
        with self.assertRaises(ValueError):
            build_timeline_positions(feature_view, membership)


if __name__ == "__main__":
    unittest.main()
