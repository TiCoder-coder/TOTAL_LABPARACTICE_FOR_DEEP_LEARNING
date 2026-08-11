"""Structural tests for the staged notebook hyperparameter pipeline."""

import ast
import json
from pathlib import Path
import unittest


PRACTICE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PRACTICE_ROOT / "practice_1.ipynb"


class NotebookHyperparameterPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.sources = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
        ]

    def test_pipeline_cells_are_valid_python(self) -> None:
        for cell_index in (69, 70, 72, 73, 74, 76, 77):
            with self.subTest(cell_index=cell_index):
                ast.parse(self.sources[cell_index])

    def test_search_stages_are_ordered_and_validation_only(self) -> None:
        search_source = "\n".join(self.sources[69:75])
        stage_markers = (
            "generate_learning_rate_trials(",
            "generate_architecture_trials(",
            "refinement_learning_rates(",
            "generate_confirmation_trials(",
        )
        marker_positions = [search_source.index(marker) for marker in stage_markers]
        self.assertEqual(marker_positions, sorted(marker_positions))
        self.assertNotIn("test_loader", search_source)
        self.assertNotIn("test_dataset", search_source)
        self.assertIn("validation_loader", search_source)

    def test_every_search_trial_has_isolated_recovery_log_plot_and_artifact(self) -> None:
        runner_source = self.sources[70]
        self.assertIn("Search trial IDs must be unique", runner_source)
        self.assertIn("SEARCH_RECOVERY_DIR / f'{trial_id}.resume.pth'", runner_source)
        self.assertIn("/ trial_stage\n", runner_source)
        self.assertIn("save_path=SEARCH_PLOT_OUTPUT_DIR / f'{trial_id}.png'", runner_source)
        self.assertIn("SEARCH_TRIAL_OUTPUT_DIR / f'{trial_id}.pth'", runner_source)
        self.assertIn("atomic_torch_save(", runner_source)

    def test_confirmation_controls_final_selection_and_exports(self) -> None:
        confirmation_source = self.sources[74]
        final_source = self.sources[77]
        self.assertIn("aggregate_confirmation_results(", confirmation_source)
        self.assertIn("selected_hyperparameter_summary", confirmation_source)
        self.assertIn("export_search_outputs(", confirmation_source)
        self.assertIn("final_config = dict(selected_config)", final_source)
        self.assertIn("selected_result['best_epoch']", final_source)
        self.assertIn("save_path=", final_source)


if __name__ == "__main__":
    unittest.main()
