import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from course_work.data.acquisition import (
    extract_expected_member,
    smoke_test_csv,
    validate_archive,
)


class AcquisitionTest(unittest.TestCase):
    def test_valid_archive_and_csv_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "valid.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("energydata_complete.csv", "date,Appliances\n2016-01-11 17:00:00,60\n2016-01-11 17:10:00,60\n2016-01-11 17:20:00,50\n2016-01-11 17:30:00,50\n2016-01-11 17:40:00,60\n")
            validation = validate_archive(archive)
            extracted = extract_expected_member(archive, root / "out.csv")
            self.assertTrue(validation["archive_integrity_ok"])
            self.assertTrue(validation["paths_safe"])
            self.assertTrue(smoke_test_csv(extracted)["csv_smoke_test_ok"])

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("../energydata_complete.csv", "date,Appliances\n")
            with self.assertRaises(ValueError):
                validate_archive(archive)

    def test_missing_expected_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "missing.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("other.csv", "value\n1\n")
            with self.assertRaises(ValueError):
                validate_archive(archive)

    def test_incomplete_csv_preview_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            path.write_text("date,Appliances\n2016-01-11 17:00:00,60\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke_test_csv(path)


if __name__ == "__main__":
    unittest.main()
