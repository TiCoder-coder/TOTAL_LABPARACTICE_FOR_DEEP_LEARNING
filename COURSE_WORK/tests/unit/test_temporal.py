import unittest
from pathlib import Path

import pandas as pd

from course_work.data.schema import dataframe_fingerprint, load_raw_csv
from course_work.data.temporal import (
    audit_temporal_dataframe,
    build_temporal_view,
    is_temporally_valid_window,
)


class TemporalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_path = Path(__file__).resolve().parents[2] / "data/raw_data/energydata_complete.csv"
        cls.dataframe = load_raw_csv(raw_path)

    def test_canonical_timeline_passes(self) -> None:
        audit = audit_temporal_dataframe(self.dataframe)
        manifest = audit["manifest"]
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(manifest["parse_failure_count"], 0)
        self.assertEqual(manifest["gap_count"], 0)
        self.assertEqual(manifest["duplicate_timestamp_count"], 0)
        self.assertEqual(manifest["continuity_segment_count"], 1)
        self.assertEqual(manifest["continuity_ratio"], 1.0)

    def test_temporal_view_does_not_mutate_raw_dataframe(self) -> None:
        candidate = self.dataframe.copy(deep=True)
        before = dataframe_fingerprint(candidate)
        view = build_temporal_view(candidate)
        self.assertEqual(dataframe_fingerprint(candidate), before)
        self.assertIn("raw_row_index", view.columns)
        self.assertIn("continuity_segment_id", view.columns)

    def test_gap_creates_missing_timestamp_and_new_segment(self) -> None:
        candidate = self.dataframe.iloc[:8].drop(index=[3]).reset_index(drop=True)
        audit = audit_temporal_dataframe(candidate)
        self.assertEqual(audit["status"], "PASS_WITH_WARNING")
        self.assertEqual(audit["manifest"]["gap_count"], 1)
        self.assertEqual(audit["manifest"]["missing_timestamp_count"], 1)
        self.assertEqual(audit["manifest"]["continuity_segment_count"], 2)

    def test_conflicting_duplicate_is_a_failure(self) -> None:
        candidate = pd.concat([self.dataframe.iloc[:5], self.dataframe.iloc[[2]]], ignore_index=True)
        candidate.loc[5, "Appliances"] = candidate.loc[5, "Appliances"] + 10
        audit = audit_temporal_dataframe(candidate)
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["manifest"]["conflicting_duplicate_group_count"], 1)

    def test_window_safety_accepts_only_exact_grid(self) -> None:
        valid = pd.date_range("2016-01-01 00:00:00", periods=5, freq="10min")
        self.assertEqual(is_temporally_valid_window(valid, lookback=4), (True, "VALID"))
        invalid = valid.delete(2).append(pd.DatetimeIndex([valid[-1] + pd.Timedelta(minutes=10)]))
        self.assertEqual(is_temporally_valid_window(invalid, lookback=4), (False, "INPUT_GAP"))


if __name__ == "__main__":
    unittest.main()
