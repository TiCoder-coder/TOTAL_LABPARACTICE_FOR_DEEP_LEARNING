import json
import unittest
from pathlib import Path

from course_work.reporting.phase_summary import (
    LOG_FILENAMES,
    SOURCE_SPECS,
    render_phase_summary,
)
from course_work.utils.artifacts import sha256_file


class PhasePresentationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_rendering_all_phases_preserves_canonical_sources(self) -> None:
        paths = {
            relative_path
            for phase_sources in SOURCE_SPECS.values()
            for _, relative_path in phase_sources
        }
        before = {path: sha256_file(self.root / path) for path in paths}
        for phase_id in range(15):
            rendered = render_phase_summary(phase_id, self.root)
            self.assertIn("cw-phase-summary", rendered.data)
            if phase_id == 6:
                self.assertNotIn("<table", rendered.data)
            else:
                self.assertIn("<table", rendered.data)
                self.assertIn("cw-phase-table", rendered.data)
            self.assertNotIn("<pre>", rendered.data)
            self.assertNotIn("Technical details", rendered.data)
            self.assertNotIn("Warnings and discrepancies", rendered.data)
        after = {path: sha256_file(self.root / path) for path in paths}
        self.assertEqual(after, before)

    def test_exactly_one_processing_log_exists_per_phase(self) -> None:
        log_root = self.root / "docs/save_log_in_processing"
        expected = set(LOG_FILENAMES.values())
        actual = {path.name for path in log_root.glob("phase_*_log.json")}
        self.assertEqual(actual, expected)
        for phase_id, filename in LOG_FILENAMES.items():
            with (log_root / filename).open("r", encoding="utf-8") as stream:
                log = json.load(stream)
            self.assertEqual(log["phase_id"], phase_id)
            self.assertNotIn("/Users/", json.dumps(log, ensure_ascii=False))
            for source in log["source_artifacts"]:
                self.assertEqual(sha256_file(self.root / source["path"]), source["sha256"])


if __name__ == "__main__":
    unittest.main()
