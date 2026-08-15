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
            "## Phase 0 - Coursework Contract",
            "## Phase 1 - Environment",
            "## Phase 2 - Data Acquisition",
            "## Phase 3 - Schema Audit",
            "## Phase 4 - Temporal Integrity Audit",
            "## Phase 5 - Exploratory Data Analysis",
            "## Phase 0-5 Boundary",
        ]
        positions = [markdown.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))

    def test_notebook_respects_phase_5_direct_eda_exception(self) -> None:
        non_phase_5_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if not cell.id.startswith("phase-5-")
        )
        tree = ast.parse(non_phase_5_code)
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
        non_phase_5_forbidden_fragments = (
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
        self.assertFalse(any(fragment in non_phase_5_code for fragment in non_phase_5_forbidden_fragments))
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
        for phase in range(6):
            self.assertIn(f"materialize_phase_{phase}", self.code)
        self.assertIn("environment_inventory(PROJECT_ROOT)", self.code)

    def test_phase_5_direct_eda_coverage_is_complete(self) -> None:
        phase_5_code = "\n\n".join(
            cell.source
            for cell in self.code_cells
            if cell.id.startswith("phase-5-")
        )
        required_fragments = (
            "df.info(",
            "df.describe(",
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
        self.assertFalse(any(fragment not in phase_5_code for fragment in required_fragments))
        for figure_index in range(1, 17):
            self.assertIn(f"EDA_{figure_index:02d}_", phase_5_code)

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
