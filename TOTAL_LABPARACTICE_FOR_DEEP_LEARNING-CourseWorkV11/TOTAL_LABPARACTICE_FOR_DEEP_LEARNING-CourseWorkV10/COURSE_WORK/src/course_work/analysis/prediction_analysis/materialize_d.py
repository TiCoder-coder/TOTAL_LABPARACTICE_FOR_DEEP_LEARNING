"""Phase 48-D — public API entry point.

Implements pairwise seed agreement, per-target seed spread, top-20 seed
disagreement, gap-safe 144-point rolling tracking, negative-value audit,
saturation audit, and persistence baseline context.

Read-only over Phase 47 source bundles. No new inference. No model selection.
No residual/regime/worst-error analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from course_work.analysis.prediction_analysis.integrity_audits import (
    write_baseline_context,
    write_negative_audit,
    write_saturation_audit,
)
from course_work.analysis.prediction_analysis.rolling import write_rolling_tracking
from course_work.analysis.prediction_analysis.seed_agreement import (
    write_pairwise_agreement,
    write_seed_spread,
    write_top_seed_disagreement,
)
from course_work.analysis.prediction_analysis.writers import utc_now_iso


def _read_wide(path: Path) -> list[dict]:
    import csv
    with path.open() as fh:
        return list(csv.DictReader(fh))


def materialize_phase48d(
    prediction_analysis_dir: Path = Path("artifacts/prediction_analysis"),
    predictions_dir: Path = Path("artifacts/final_test/predictions"),
    ftest_dir: Path = Path("artifacts/final_test"),
) -> dict[str, Any]:
    """Execute the Phase 48-D slice."""
    wide_path = prediction_analysis_dir / "prediction_wide_table.csv"
    if not wide_path.exists():
        raise FileNotFoundError(
            f"Phase 48-B wide table missing at {wide_path}. "
            "Run materialize_phase48() first."
        )
    wide_rows = _read_wide(wide_path)

    # 1. Pairwise seed agreement
    write_pairwise_agreement(wide_rows)
    # 2. Per-target seed spread
    write_seed_spread(wide_rows)
    # 3. Top-20 seed disagreement
    write_top_seed_disagreement(wide_rows)
    # 4. Gap-safe 144-point rolling tracking
    write_rolling_tracking(wide_rows)
    # 5. Negative-value audit
    write_negative_audit(wide_rows)
    # 6. Saturation audit
    write_saturation_audit(wide_rows)
    # 7. Persistence baseline context
    write_baseline_context(wide_rows, predictions_dir=predictions_dir, ftest_dir=ftest_dir)

    return {
        "phase": 48,
        "scope": "PHASE_48_D",
        "executed_at_utc": utc_now_iso(),
        "outputs": [
            "prediction_seed_pairwise_agreement.csv",
            "prediction_seed_spread.csv",
            "prediction_top_seed_disagreement.csv",
            "prediction_rolling_tracking.csv",
            "prediction_negative_value_audit.csv",
            "prediction_saturation_audit.csv",
            "prediction_baseline_context.csv",
        ],
    }
