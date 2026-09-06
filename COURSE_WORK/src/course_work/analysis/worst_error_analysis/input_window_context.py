"""Phase 51-F — exact frozen target-to-window mapping.

For every frozen Test target, the canonical input window contract is:

  lookback         = 72 steps
  horizon          = 1 step
  cadence          = 10 minutes
  feature_count    = 33 (FS2_TF1)
  feature_set      = FS2_TF1
  target_timestamp = the Test target timestamp itself
  window_end       = target_timestamp  (last input step)
  window_start     = target_timestamp − 72 × 10 min = −12 hours

Boundary semantics: WB0_CONTEXT_CARRY_OVER (canonical from
final_test_population_manifest.json).

Persist ONLY the mapping contract per target. Do NOT persist the raw 72×33
feature tensor (it is reconstructed at training/inference time and is NOT
a stored Phase 47-50 artifact). The mapping itself is sufficient for the
casebook + Phase 52 handoff to know which window goes with which target.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root


LOOKBACK = 72
HORIZON = 1
CADENCE_MINUTES = 10
FEATURE_COUNT = 33
FEATURE_SET = "FS2_TF1"
BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"


def _load_persistence(project_root: Path) -> dict[str, dict[str, str]]:
    fp = project_root / "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return {r["target_id"]: r for r in csv.DictReader(fh)}


def _ts_minus_minutes(ts_str: str, minutes: int) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    dt = datetime.strptime(ts_str, fmt)
    return (dt - timedelta(minutes=minutes)).strftime(fmt)


def _ts_plus_minutes(ts_str: str, minutes: int) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    dt = datetime.strptime(ts_str, fmt)
    return (dt + timedelta(minutes=minutes)).strftime(fmt)


def build_input_window_manifest(
    project_root: Path,
    selected_target_ids: list[str],
) -> list[dict[str, Any]]:
    """Build O51.25 input-window manifest rows for each selected target."""
    pers = _load_persistence(project_root)
    rows: list[dict[str, Any]] = []
    for tid in selected_target_ids:
        p = pers.get(tid)
        if p is None:
            raise RuntimeError(
                f"Selected target {tid} not found in frozen Persistence/Test artifacts."
            )
        ts = p["target_timestamp"]
        rows.append({
            "target_id": tid,
            "target_timestamp": ts,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "cadence_minutes": CADENCE_MINUTES,
            "feature_count": FEATURE_COUNT,
            "feature_set": FEATURE_SET,
            "boundary_protocol": BOUNDARY_PROTOCOL,
            "window_start_timestamp": _ts_minus_minutes(ts, LOOKBACK * CADENCE_MINUTES),
            "window_end_timestamp": _ts_minus_minutes(ts, 0),
            "window_target_index": LOOKBACK,  # last index of [0..71]
            "is_model_input": "PAST_ONLY",
            "future_context_used_as_model_input": False,
        })
    return rows


def verify_canonical_constants() -> dict[str, Any]:
    """Return canonical contract constants for casebook metadata."""
    return {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "cadence_minutes": CADENCE_MINUTES,
        "feature_count": FEATURE_COUNT,
        "feature_set": FEATURE_SET,
        "boundary_protocol": BOUNDARY_PROTOCOL,
    }
