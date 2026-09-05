import ast
import io
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path

import nbformat


class NotebookBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.notebook_path = cls.root / "notebook_course_work/CourseWork.ipynb"
        cls.notebook = nbformat.read(cls.notebook_path, as_version=4)
        cls.code_cells = [cell for cell in cls.notebook.cells if cell.cell_type == "code"]
        cls.code = "\n\n".join(cell.source for cell in cls.code_cells)

    def test_notebook_is_valid_and_phase_sections_are_ordered(self) -> None:
        nbformat.validate(self.notebook)
        markdown = "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "markdown"
        )
        headings = [
            "## Phase 1 - Environment",
            "## Phase 2 - Data Acquisition",
            "## Phase 3 - Schema Audit",
            "## Phase 4 - Temporal Integrity Audit",
            "## Phase 5 - Chronological Split",
            "## Phase 6 - Exploratory Data Analysis",
            "## Phase 7 - Feature Engineering",
            "## Phase 8 - Feature-Set Variants",
            "## Phase 9 - Train-Only Scaling",
            "## Phase 10 - Window Builder",
            "## Phase 11 - DataLoaders",
            "## Phase 12 - Shared Metrics",
            "## Phase 13 - Experiment Registry",
            "## Phase 14 - Persistence Baseline",
        ]
        positions = [markdown.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("## Phase 0 - Coursework Contract", markdown)

    def test_notebook_respects_phase_6_direct_eda_exception(self) -> None:
        non_phase_6_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if not cell.id.startswith("phase-6-")
        )
        tree = ast.parse(non_phase_6_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        non_phase_6_forbidden_fragments = (
            "pd.read_csv",
            "pd.to_datetime",
            ".interpolate(",
            ".fillna(",
            ".drop(",
            ".corr(",
            ".rolling(",
            "plt.",
            "seaborn",
            "sns.",
            "to_csv",
            "to_json",
            "json.dump",
            "hashlib",
            "sha256",
        )
        self.assertFalse(any(fragment in non_phase_6_code for fragment in non_phase_6_forbidden_fragments))
        notebook_forbidden_fragments = (
            "fetch_ucirepo",
            "to_csv",
            "to_json",
            "json.dump",
            "open(",
            "torch.",
            "DataLoader",
            "optimizer",
            ".fit(",
            ".backward(",
        )
        self.assertFalse(any(fragment in self.code for fragment in notebook_forbidden_fragments))
        for phase in range(1, 15):
            self.assertIn(f"materialize_phase_{phase}", self.code)
            self.assertIn(f"render_phase_summary({phase}, PROJECT_ROOT)", self.code)
            self.assertNotIn(f"display(phase_{phase}_signoff)", self.code)
            self.assertNotIn(f"display(phase_{phase}_manifest)", self.code)
        self.assertNotIn("materialize_phase_0", self.code)
        self.assertNotIn("render_phase_summary(0, PROJECT_ROOT)", self.code)
        self.assertNotIn("display(read_json(", self.code)

    def test_phase_5_is_orchestration_only(self) -> None:
        phase_5_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-5-")
        )
        tree = ast.parse(phase_5_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "train_test_split",
            ".sample(",
            ".shuffle(",
            "np.random",
            "random.",
            "math.floor",
            "pd.read_csv",
            ".iloc[",
            ".loc[",
            "to_csv",
            "to_json",
            "json.dump",
            "plt.",
            "StandardScaler",
        )
        self.assertFalse(any(fragment in phase_5_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_5(PROJECT_ROOT)", phase_5_code)

    def test_phase_7_is_orchestration_only(self) -> None:
        phase_7_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-7-")
        )
        tree = ast.parse(phase_7_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "np.sin",
            "np.cos",
            "pd.to_datetime",
            ".dt.",
            "to_csv",
            "to_json",
            "json.dump",
            "rolling(",
            "shift(",
            "StandardScaler",
        )
        self.assertFalse(any(fragment in phase_7_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_7(PROJECT_ROOT)", phase_7_code)

    def test_phase_8_is_orchestration_only(self) -> None:
        phase_8_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-8-")
        )
        tree = ast.parse(phase_8_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "FEATURE_COMPONENTS",
            "FEATURE_SET_REGISTRY",
            "compute_feature_fingerprint",
            "hashlib",
            "sha256",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
            "select_dtypes",
            ".to_numpy(",
        )
        self.assertFalse(any(fragment in phase_8_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_8(PROJECT_ROOT)", phase_8_code)

    def test_phase_9_is_orchestration_only(self) -> None:
        phase_9_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-9-")
        )
        tree = ast.parse(phase_9_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "StandardScaler",
            ".fit(",
            ".fit_transform(",
            ".transform(",
            "joblib",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
            "mean(",
            "std(",
        )
        self.assertFalse(any(fragment in phase_9_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_9(PROJECT_ROOT)", phase_9_code)
        self.assertIn("render_phase_summary(9, PROJECT_ROOT)", phase_9_code)

    def test_phase_10_is_orchestration_only(self) -> None:
        phase_10_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-10-")
        )
        tree = ast.parse(phase_10_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "compute_window_bounds",
            "build_native_window_index",
            "build_common_target_population",
            "materialize_window(",
            "pd.read_csv",
            ".iloc[",
            ".loc[",
            "to_csv",
            "to_json",
            "json.dump",
            "StandardScaler",
            ".fit(",
            ".transform(",
            "DataLoader",
        )
        self.assertFalse(any(fragment in phase_10_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_10(PROJECT_ROOT)", phase_10_code)
        self.assertIn("render_phase_summary(10, PROJECT_ROOT)", phase_10_code)

    def test_phase_11_is_orchestration_only(self) -> None:
        phase_11_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-11-")
        )
        tree = ast.parse(phase_11_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "SequenceWindowDataset",
            "DataLoader(",
            "build_dataloader(",
            "build_dataset_suite(",
            "torch.Generator",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
        )
        self.assertFalse(any(fragment in phase_11_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_11(PROJECT_ROOT)", phase_11_code)
        self.assertIn("render_phase_summary(11, PROJECT_ROOT)", phase_11_code)

    def test_phase_12_is_orchestration_only(self) -> None:
        phase_12_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-12-")
        )
        tree = ast.parse(phase_12_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "mean_absolute_error",
            "root_mean_squared_error",
            "r2_score",
            "compute_regression_metrics(",
            "convert_predictions_to_wh(",
            "np.",
            "torch.",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
        )
        self.assertFalse(any(fragment in phase_12_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_12(PROJECT_ROOT)", phase_12_code)
        self.assertIn("render_phase_summary(12, PROJECT_ROOT)", phase_12_code)

    def test_phase_13_is_orchestration_only(self) -> None:
        phase_13_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-13-")
        )
        tree = ast.parse(phase_13_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "ExperimentRegistry(",
            "register_run(",
            "start_run(",
            "complete_run(",
            "register_artifact(",
            "register_metric(",
            "compute_config_fingerprint(",
            "validate_sweep_consistency(",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
        )
        self.assertFalse(any(fragment in phase_13_code for fragment in forbidden_fragments))
        self.assertIn("materialize_phase_13(PROJECT_ROOT)", phase_13_code)
        self.assertIn("render_phase_summary(13, PROJECT_ROOT)", phase_13_code)

    def test_phase_14_is_orchestration_only(self) -> None:
        phase_14_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-14-")
        )
        tree = ast.parse(phase_14_code)
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        self.assertFalse(any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)))
        forbidden_fragments = (
            "predict_persistence(",
            "prepare_validation_persistence_data(",
            "compute_regression_metrics(",
            "ExperimentRegistry(",
            "register_run(",
            "start_run(",
            "complete_run(",
            "register_artifact(",
            "register_metric(",
            "pd.read_csv",
            "to_csv",
            "to_json",
            "json.dump",
        )
        self.assertFalse(any(fragment in phase_14_code for fragment in forbidden_fragments))
        self.assertEqual(phase_14_code.count("materialize_phase_14(PROJECT_ROOT)"), 1)
        self.assertEqual(phase_14_code.count("render_phase_summary(14, PROJECT_ROOT)"), 1)

    def test_phase_6_direct_eda_coverage_is_complete(self) -> None:
        phase_6_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-6-")
        )
        required_fragments = (
            "data_overview_memory_mib",
            "render_dataframe_table(",
            "Representative Data Samples",
            "Descriptive Statistics by Variable",
            "df.describe(",
            "df.dtypes.reindex(data_overview_statistics.index)",
            "missing_counts",
            "missing_percentage",
            "sns.heatmap(",
            "day_of_week",
            "is_weekend",
            "hour_counts",
            "day_counts",
            "df[\"Appliances\"].skew()",
            "daily_average",
            "first_week",
            "correlation_matrix",
            "top_correlations",
            "sns.scatterplot(",
            "calculate_segment_cross_correlation",
            ".boxplot(",
            "summarize_iqr_outliers",
            "df_eda_smoothed_demo",
            ".interpolate(",
            "Canonical DataFrame Unchanged",
            "display_figure",
        )
        self.assertFalse(any(fragment not in phase_6_code for fragment in required_fragments))
        for figure_index in range(1, 17):
            self.assertIn(f"EDA_{figure_index:02d}_", phase_6_code)

    def test_notebook_code_has_no_comments_or_icons(self) -> None:
        tokens = tokenize.generate_tokens(io.StringIO(self.code).readline)
        self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
        self.assertFalse(any(ord(character) >= 0x1F000 for character in self.code))

    def test_execution_state_is_clean_or_complete(self) -> None:
        counts = [cell.execution_count for cell in self.code_cells]
        if all(count is None for count in counts):
            self.assertTrue(all(not cell.outputs for cell in self.code_cells))
            return
        self.assertTrue(all(isinstance(count, int) for count in counts))
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(counts), len(set(counts)))
        for cell in self.code_cells:
            self.assertFalse(any(output.output_type == "error" for output in cell.outputs))
            for output in cell.outputs:
                if output.output_type == "stream":
                    self.assertNotIn("Warning", output.text)
                    self.assertNotIn("Traceback", output.text)

    def test_notebook_has_no_embedded_machine_specific_path(self) -> None:
        source = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertNotIn("/Users/", source)

    def test_package_imports_from_notebook_directory_without_path_bootstrap(self) -> None:
        command = [
            sys.executable,
            "-I",
            "-c",
            "from pathlib import Path; import course_work; print(Path(course_work.__file__).resolve())",
        ]
        completed = subprocess.run(
            command,
            cwd=self.notebook_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            Path(completed.stdout.strip()),
            (self.root / "src/course_work/__init__.py").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
