"""Phase 48-C — public API entry point for prediction-behavior diagnostics.

Reads the Phase 48-B canonical wide/long tables (read-only) and the frozen
Phase 47 prediction bundles via inputs.load_phase48_inputs (no new inference).

Strictly forbidden actions remain unchanged: no training, no checkpoint
loading for new predictions, no best-seed selection, no ensemble metric,
no prediction shifting / clipping, no residual / regime / worst-error /
attention analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from course_work.phase48.change_behavior import (
    write_change_summary,
    write_direction_agreement,
)
from course_work.phase48.contract import (
    ACF_REGISTERED_LAGS,
    LAG_RANGE,
    LAG_SIGN_CONVENTION,
    PEAK_TIMING_WINDOW_STEPS,
    SEED_STD_DDOF,
)
from course_work.phase48.distribution import (
    write_distribution_summary,
    write_range_compression,
)
from course_work.phase48.extrema import (
    write_local_extrema_summary,
    write_peak_timing_summary,
)
from course_work.phase48.temporal_diagnostics import (
    write_acf_diagnostics,
    write_lag_diagnostics,
)
from course_work.phase48.writers import utc_now_iso


def _read_wide(path: Path) -> list[dict]:
    import csv
    with path.open() as fh:
        return list(csv.DictReader(fh))


def materialize_phase48c(
    prediction_analysis_dir: Path = Path("artifacts/prediction_analysis"),
) -> dict[str, Any]:
    """Execute the Phase 48-C prediction-behavior diagnostic slice.

    Requires Phase 48-B outputs to already exist under
    artifacts/prediction_analysis/ (wide + long tables + contract).
    """
    wide_path = prediction_analysis_dir / "prediction_wide_table.csv"
    if not wide_path.exists():
        raise FileNotFoundError(
            f"Phase 48-B wide table missing at {wide_path}. "
            "Run materialize_phase48() from materialize.py first."
        )
    wide_rows = _read_wide(wide_path)

    # 1. Distribution summary
    write_distribution_summary(wide_rows)
    # 2. Range/compression
    write_range_compression(wide_rows)
    # 3. Change summary (gap-safe)
    change_path, change_extras = write_change_summary(wide_rows)
    # 4. Direction agreement (3-class, no epsilon)
    write_direction_agreement(wide_rows)
    # 5. Lag diagnostics
    write_lag_diagnostics(wide_rows)
    # 6. Prediction ACF
    write_acf_diagnostics(wide_rows)
    # 7. Local extrema summary
    _, extrema_extras = write_local_extrema_summary(wide_rows)
    # 8. Peak timing (±PEAK_TIMING_WINDOW_STEPS, FROZEN at 1)
    write_peak_timing_summary(wide_rows, extrema_extras)

    return {
        "phase": 48,
        "scope": "PHASE_48_C",
        "executed_at_utc": utc_now_iso(),
        "outputs": [
            "prediction_distribution_summary.csv",
            "prediction_range_compression.csv",
            "prediction_change_summary.csv",
            "prediction_direction_agreement.csv",
            "prediction_lag_diagnostics.csv",
            "prediction_acf_diagnostics.csv",
            "prediction_local_extrema_summary.csv",
            "prediction_peak_timing_summary.csv",
        ],
        "valid_transition_count": change_extras["valid_transition_count"],
        "gap_excluded_count": change_extras["gap_excluded_count"],
        "lag_range": list(LAG_RANGE),
        "lag_convention": LAG_SIGN_CONVENTION,
        "acf_lags": list(ACF_REGISTERED_LAGS),
        "peak_window_steps": PEAK_TIMING_WINDOW_STEPS,
        "seed_std_ddof": SEED_STD_DDOF,
    }
