import unittest
from copy import deepcopy
from pathlib import Path

import torch

from course_work.utils.environment import (
    device_smoke_test,
    dependency_freeze,
    environment_identity_differences,
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

    def test_runtime_context_drift_does_not_change_environment_identity(self) -> None:
        recorded = environment_inventory()
        current = deepcopy(recorded)
        current["working_directory"] = "/different/runtime/directory"
        current["project_root"] = "/different/project/location"
        current["cpu_count"] = (recorded["cpu_count"] or 0) + 1
        current["mps_available"] = not recorded["mps_available"]
        current["mps_device_name"] = "mps" if current["mps_available"] else None
        current["selected_device"] = "mps" if current["mps_available"] else "cpu"
        current["kernel"]["kernel_spec_path"] = "/different/kernel/spec/path"
        self.assertEqual(environment_identity_differences(recorded, current), ())

    def test_package_version_drift_changes_environment_identity(self) -> None:
        recorded = environment_inventory()
        current = deepcopy(recorded)
        current["package_versions"]["torch"] = "different"
        self.assertEqual(
            environment_identity_differences(recorded, current),
            ("package_versions",),
        )

    def test_dependency_freeze_excludes_local_editable_project(self) -> None:
        freeze = dependency_freeze()
        self.assertNotIn("course-work", freeze.lower())
        self.assertNotIn("course_work", freeze.lower())
        self.assertNotIn("-e ", freeze)


if __name__ == "__main__":
    unittest.main()
