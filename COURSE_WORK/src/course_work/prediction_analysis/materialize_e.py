"""Phase 48-E — public API entry point.

Implements the finalization slice:
  - figures (20 deterministic PNGs)
  - findings (descriptive only)
  - Phase49 / Phase50 / Phase51 handoffs
  - tests / discrepancies / summary / report / README / signoff
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from course_work.prediction_analysis.contract import OUTPUT_DIR
from course_work.prediction_analysis.figures import render_all_figures
from course_work.prediction_analysis.finalize import (
    write_discrepancies,
    write_findings,
    write_phase49_handoff,
    write_phase50_handoff,
    write_phase51_handoff,
    write_readme,
    write_report,
    write_signoff,
    write_summary,
    write_tests,
)
from course_work.prediction_analysis.writers import utc_now_iso


def materialize_phase48e(
    phase48_b_passed: int = 34,
    phase48_c_passed: int = 58,
    phase48_d_passed: int = 52,
    phase48_e_passed: int = 0,
    all_gates_pass: bool = True,
) -> dict[str, Any]:
    """Execute the Phase 48-E finalization slice."""
    # 1. Figures
    fig_paths = render_all_figures()

    # 2. Findings
    write_findings()

    # 3. Handoffs
    write_phase49_handoff()
    write_phase50_handoff()
    write_phase51_handoff()

    # 4. Tests + discrepancies
    write_tests(
        phase48_b_passed=phase48_b_passed,
        phase48_c_passed=phase48_c_passed,
        phase48_d_passed=phase48_d_passed,
        phase48_e_passed=phase48_e_passed,
    )
    write_discrepancies([])

    # 5. Summary + report + README
    write_summary()
    write_report()
    write_readme()

    # 6. Signoff
    write_signoff(all_gates_pass=all_gates_pass)

    return {
        "phase": 48,
        "scope": "PHASE_48_E",
        "executed_at_utc": utc_now_iso(),
        "figures_count": len(fig_paths),
        "outputs": [
            "prediction_analysis_findings.csv",
            "phase49_residual_analysis_handoff.json",
            "phase50_error_regime_context_handoff.json",
            "phase51_worst_error_context_handoff.json",
            "prediction_analysis_tests.csv",
            "prediction_analysis_discrepancies.json",
            "prediction_analysis_summary.json",
            "prediction_analysis_report.md",
            "README_PREDICTION_ANALYSIS.md",
            "phase_48_signoff.json",
        ] + [f"figures/{p.name}" for p in fig_paths],
    }
