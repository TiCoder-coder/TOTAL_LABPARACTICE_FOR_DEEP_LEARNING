import unittest
from pathlib import Path

import pandas as pd

from course_work.data.schema import (
    EXPECTED_COLUMNS,
    audit_raw_dataframe,
    dataframe_fingerprint,
    expected_schema,
    load_raw_csv,
    serialized_header,
)


class SchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = Path(__file__).resolve().parents[2] / "data/raw_data/energydata_complete.csv"
        cls.dataframe = load_raw_csv(cls.raw_path)

    def test_expected_schema_is_complete(self) -> None:
        schema = expected_schema()
        self.assertEqual(list(schema), EXPECTED_COLUMNS)
        self.assertEqual(schema["Appliances"]["role"], "target")
        self.assertEqual(schema["date"]["role"], "time")
        self.assertEqual(schema["rv1"]["role"], "random_control")
        self.assertNotEqual(schema["T6"]["feature_group"], schema["T_out"]["feature_group"])

    def test_canonical_raw_schema_passes(self) -> None:
        audit = audit_raw_dataframe(self.dataframe, serialized_header(self.raw_path))
        self.assertEqual(audit["critical_failures"], [])
        self.assertEqual(audit["row_count"], 19735)
        self.assertEqual(audit["column_count"], 29)
        self.assertEqual(audit["ordered_columns"], EXPECTED_COLUMNS)
        self.assertEqual(audit["timestamp_parse_failure_count"], 0)

    def test_audit_does_not_mutate_dataframe(self) -> None:
        candidate = self.dataframe.copy(deep=True)
        before = dataframe_fingerprint(candidate)
        audit_raw_dataframe(candidate, list(candidate.columns))
        self.assertEqual(dataframe_fingerprint(candidate), before)

    def test_missing_target_is_a_critical_failure(self) -> None:
        candidate = self.dataframe.drop(columns=["Appliances"])
        audit = audit_raw_dataframe(candidate, list(candidate.columns))
        self.assertIn("missing_columns", audit["critical_failures"])
        self.assertIn("target_or_timestamp_cardinality", audit["critical_failures"])

    def test_non_numeric_target_is_a_critical_failure(self) -> None:
        candidate = self.dataframe.copy(deep=True)
        candidate["Appliances"] = candidate["Appliances"].astype(object)
        candidate.loc[0, "Appliances"] = "invalid"
        audit = audit_raw_dataframe(candidate, list(candidate.columns))
        self.assertIn("numeric_coercion", audit["critical_failures"])


if __name__ == "__main__":
    unittest.main()
