import unittest
from pathlib import Path

import torch

from course_work.utils.environment import (
    device_smoke_test,
    environment_inventory,
    resolve_kernel_contract,
    select_device,
)


class EnvironmentTest(unittest.TestCase):
    def test_device_selection_follows_available_backend(self) -> None:
        selected = select_device()
        if torch.cuda.is_available():
            self.assertEqual(selected.type, "cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.assertEqual(selected.type, "mps")
        else:
            self.assertEqual(selected.type, "cpu")

    def test_kernel_matches_interpreter(self) -> None:
        notebook = Path(__file__).resolve().parents[2] / "notebook_course_work/CourseWork.ipynb"
        self.assertTrue(resolve_kernel_contract(notebook)["matches_interpreter"])

    def test_environment_inventory_has_required_fields(self) -> None:
        inventory = environment_inventory()
        self.assertEqual(inventory["environment_id"], "ENV-v1")
        self.assertEqual(inventory["default_dtype"], "torch.float32")
        self.assertTrue(inventory["kernel"]["matches_interpreter"])

    def test_device_smoke_test_passes(self) -> None:
        report = device_smoke_test()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["all_finite"])


if __name__ == "__main__":
    unittest.main()
