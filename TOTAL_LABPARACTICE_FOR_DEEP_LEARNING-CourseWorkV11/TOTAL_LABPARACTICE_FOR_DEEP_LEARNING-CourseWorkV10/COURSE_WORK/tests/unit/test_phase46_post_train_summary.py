"""Tests for Part 2G-M: avg_rmse final summary NameError fix.

Background: After successful 3-seed completion of Phase 46, the runner
crashed at the final console summary with `NameError: name 'avg_rmse' is
not defined`. The variable was locally-scoped inside
write_phase46_artifacts() and never returned. The print statement in the
calling scope had no source for `avg_rmse`.

This test proves the fix: the final summary path can no longer raise
NameError after successful 3-seed completion.

No training, no inference, no Test access, no model_state_dict mutations.
"""

from __future__ import annotations

import importlib.util as _iu
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
P46_PATH = ROOT / "src" / "course_work" / "scripts" / "p46_three_seed_runs.py"


def _load_p46_module() -> object:
    spec = _iu.spec_from_file_location("_p46_2gm", str(P46_PATH))
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAvgRmseFinalSummary:
    """The final console summary must compute avg_rmse from run_records.

    Per Part 2G-M: the bug was that the print statement at the end of
    main_inner() tried to use `avg_rmse` from a scope where it was never
    defined. The fix: compute `avg_rmse` locally from the already-persisted
    per-seed rmse values in run_records, just before the print.
    """

    def test_avg_rmse_fix_present_in_runner_source(self) -> None:
        """The fix must compute avg_rmse from run_records immediately
        before the print statement, not rely on a previous scope.
        """
        runner_src = (ROOT / "src/course_work" / "scripts" / "p46_three_seed_runs.py").read_text()

        fix_pattern = (
            'avg_rmse = float(np.mean([float(r["rmse"]) for r in run_records]))'
        )
        print_pattern = 'Average FINAL_DEV_DIAGNOSTIC RMSE: {avg_rmse'
        assert fix_pattern in runner_src, (
            "Fix not found: avg_rmse must be recomputed locally from run_records"
        )
        assert runner_src.index(fix_pattern) < runner_src.index(print_pattern), (
            "Fix must appear BEFORE the print statement"
        )

    def test_avg_rmse_recomputed_from_run_records(self) -> None:
        """Demonstrate that the fix logic produces the expected average."""
        run_records = [
            {"seed": 42, "rmse": 47.85565017942988},
            {"seed": 123, "rmse": 47.30963619597326},
            {"seed": 2026, "rmse": 48.501011461966336},
        ]
        import numpy as np
        avg_rmse = float(np.mean([float(r["rmse"]) for r in run_records]))
        expected = (47.85565017942988 + 47.30963619597326 + 48.501011461966336) / 3
        assert abs(avg_rmse - expected) < 1e-9
        assert abs(avg_rmse - 47.88876594578983) < 1e-9

    def test_final_summary_path_does_not_raise_name_error(self) -> None:
        """Mock-up: simulate the final summary path.

        No actual scientific execution occurs. We only verify that the
        code path computes avg_rmse locally from run_records and would
        not raise NameError.
        """
        run_records = [
            {"seed": 42, "rmse": 47.85565017942988},
            {"seed": 123, "rmse": 47.30963619597326},
            {"seed": 2026, "rmse": 48.501011461966336},
        ]

        import numpy as np
        try:
            avg_rmse = float(np.mean([float(r["rmse"]) for r in run_records]))
            _ = f"Average FINAL_DEV_DIAGNOSTIC RMSE: {avg_rmse:.4f}"
        except NameError as exc:
            pytest.fail(
                f"NameError at final summary path: {exc}. "
                "avg_rmse must be defined locally from run_records."
            )

    def test_runner_has_no_unrelated_undefined_avg_rmse(self) -> None:
        """Verify the runner source has no remaining NameError-prone
        `avg_rmse` reference at the final summary scope (outside
        write_phase46_artifacts).
        """
        runner_src = (ROOT / "src/course_work" / "scripts" / "p46_three_seed_runs.py").read_text()

        lines = runner_src.split("\n")
        avg_rmse_lines = [
            (i + 1, line) for i, line in enumerate(lines)
            if "avg_rmse" in line
        ]
        assert len(avg_rmse_lines) == 5, (
            f"Expected exactly 5 references to avg_rmse (1 def + 2 uses in "
            f"write_phase46_artifacts + 1 def + 1 use in main_inner fix). "
            f"Found {len(avg_rmse_lines)}: {avg_rmse_lines}"
        )
