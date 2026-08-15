import unittest
from pathlib import Path

import pandas as pd

from course_work.data.eda import (
    SELECTED_LAGS,
    build_eda_view,
    numeric_summary,
    prepare_eda_analysis,
    segment_aware_lag_correlation,
)
from course_work.data.schema import dataframe_fingerprint, load_raw_csv
from course_work.data.temporal import build_temporal_view
from course_work.utils.artifacts import read_json


class EdaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.raw_path = cls.root / "data/raw_data/energydata_complete.csv"
        cls.raw = load_raw_csv(cls.raw_path)
        cls.temporal = build_temporal_view(cls.raw)
        cls.eda = build_eda_view(cls.temporal)
        cls.schema_manifest = read_json(cls.root / "artifacts/schema/schema_manifest.json")

    def test_eda_view_preserves_lineage_and_raw_values(self) -> None:
        before = dataframe_fingerprint(self.raw)
        self.assertEqual(len(self.eda), len(self.raw))
        self.assertTrue(self.eda["raw_row_index"].is_unique)
        self.assertEqual(self.eda["Appliances"].tolist(), self.raw["Appliances"].tolist())
        self.assertEqual(dataframe_fingerprint(self.raw), before)

    def test_numeric_summary_has_required_schema(self) -> None:
        summary = numeric_summary(self.eda, self.schema_manifest)
        required = {"variable", "role", "group", "count", "mean", "std", "min", "q01", "q05", "q25", "median", "q75", "q90", "q95", "q99", "max", "skewness"}
        self.assertEqual(set(summary.columns), required)
        self.assertEqual(len(summary), 28)

    def test_selected_lags_are_segment_aware_and_finite(self) -> None:
        for lag in SELECTED_LAGS:
            correlation, pairs = segment_aware_lag_correlation(self.eda, "Appliances", lag)
            self.assertTrue(pd.notna(correlation))
            self.assertEqual(pairs, len(self.eda) - lag)

    def test_gap_prevents_cross_segment_lag_pair(self) -> None:
        candidate = self.raw.iloc[:8].drop(index=[3]).reset_index(drop=True)
        view = build_eda_view(build_temporal_view(candidate))
        _, pairs = segment_aware_lag_correlation(view, "Appliances", 1)
        self.assertEqual(pairs, len(view) - 2)

    def test_complete_analysis_preserves_raw_dataframe(self) -> None:
        analysis = prepare_eda_analysis(self.root)
        self.assertEqual(analysis["raw_dataframe_fingerprint_before"], analysis["raw_dataframe_fingerprint_after"])
        self.assertEqual(len(analysis["hypotheses"]), 10)
        self.assertEqual(set(analysis["hypotheses"]["decision_status"]), {"UNTESTED"})
        self.assertEqual(len(analysis["representative_day"]), 144)
        self.assertEqual(len(analysis["representative_week"]), 1008)


if __name__ == "__main__":
    unittest.main()
