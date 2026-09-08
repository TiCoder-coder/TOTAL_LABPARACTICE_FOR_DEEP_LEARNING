import json
import tempfile
import unittest
from pathlib import Path

from course_work.contracts.coursework import (
    EXPECTED_OPTION_IDS,
    coursework_contract_fingerprint,
    load_coursework_contract,
    materialize_phase_0,
    validate_coursework_contract,
)


class CourseworkContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_coursework_contract()

    def test_canonical_contract_is_valid(self) -> None:
        self.assertEqual(validate_coursework_contract(self.contract), ())

    def test_option_registry_is_complete(self) -> None:
        registry = self.contract["option_registry"]
        for group, expected in EXPECTED_OPTION_IDS.items():
            self.assertEqual({item["id"] for item in registry[group]}, expected)

    def test_split_is_chronological_and_complete(self) -> None:
        split = self.contract["split"]
        self.assertEqual(split["type"], "chronological")
        self.assertAlmostEqual(split["train"] + split["validation"] + split["test"], 1.0)

    def test_target_and_horizon_are_unambiguous(self) -> None:
        problem = self.contract["problem"]
        self.assertEqual(problem["target"], "Appliances")
        self.assertEqual(problem["target_unit"], "Wh")
        self.assertEqual(problem["forecast_horizon_steps"], 1)
        self.assertEqual(problem["forecast_horizon_minutes"], 10)

    def test_materialization_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs/base/coursework_contract.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(self.contract), encoding="utf-8")
            first = materialize_phase_0(root)
            second = materialize_phase_0(root)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(first["config_fingerprint"], coursework_contract_fingerprint(self.contract))


if __name__ == "__main__":
    unittest.main()
