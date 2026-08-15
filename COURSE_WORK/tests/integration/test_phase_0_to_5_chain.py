import unittest
from pathlib import Path

from course_work.contracts.coursework import materialize_phase_0
from course_work.data.acquisition import materialize_phase_2
from course_work.data.schema import materialize_phase_3
from course_work.data.temporal import materialize_phase_4
from course_work.reporting.eda import FIGURE_FILENAMES, materialize_phase_5
from course_work.utils.artifacts import read_json, sha256_file
from course_work.utils.environment import materialize_phase_1


class PhaseChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_phase_chain_signoffs_are_valid(self) -> None:
        signoffs = [
            materialize_phase_0(self.root),
            materialize_phase_1(self.root),
            materialize_phase_2(self.root),
            materialize_phase_3(self.root),
            materialize_phase_4(self.root),
            materialize_phase_5(self.root),
        ]
        self.assertEqual([item["phase_id"] for item in signoffs], list(range(6)))
        self.assertEqual([item["status"] for item in signoffs], ["PASS", "PASS", "PASS", "PASS_WITH_WARNING", "PASS", "PASS"])

    def test_every_signed_output_checksum_matches(self) -> None:
        signoff_paths = [
            "artifacts/contracts/phase_0_signoff.json",
            "artifacts/environment/phase_1_signoff.json",
            "artifacts/acquisition/phase_2_signoff.json",
            "artifacts/schema/phase_3_signoff.json",
            "artifacts/temporal/phase_4_signoff.json",
            "artifacts/eda/phase_5_signoff.json",
        ]
        for relative_signoff in signoff_paths:
            signoff = read_json(self.root / relative_signoff)
            for relative_output, expected_checksum in signoff["output_checksums"].items():
                self.assertEqual(sha256_file(self.root / relative_output), expected_checksum)

    def test_raw_data_is_unchanged(self) -> None:
        raw = self.root / "data/raw_data/energydata_complete.csv"
        self.assertEqual(sha256_file(raw), "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d")

    def test_eda_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/eda/eda_manifest.json")
        figures = self.root / "artifacts/eda/figures"
        self.assertEqual({path.name for path in figures.glob("*.png")}, set(FIGURE_FILENAMES))
        self.assertEqual(manifest["split_distribution_analysis_status"], "DEFERRED_TO_PHASE_8")
        self.assertEqual(manifest["processing_actions"]["rows_removed"], 0)
        self.assertFalse(manifest["processing_actions"]["outliers_removed"])
        self.assertFalse(manifest["processing_actions"]["interpolation_applied"])
        self.assertFalse(manifest["processing_actions"]["feature_selection_applied"])
        self.assertFalse(manifest["processing_actions"]["model_tuning_applied"])

    def test_phase_6_remains_unstarted(self) -> None:
        feature_module = self.root / "src/course_work/data/features.py"
        self.assertEqual(feature_module.stat().st_size, 0)
        self.assertFalse((self.root / "artifacts/features").exists())


if __name__ == "__main__":
    unittest.main()
