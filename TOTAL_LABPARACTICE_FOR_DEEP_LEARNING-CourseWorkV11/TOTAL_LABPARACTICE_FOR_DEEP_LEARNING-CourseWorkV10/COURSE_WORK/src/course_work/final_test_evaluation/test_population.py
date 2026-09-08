"""Phase 47 — Final Test Population (FINAL_TEST_POP-v1).

Canonical producer for the FINAL_TEST_POP-v1 population fingerprint.
Uses WINDOW-level (not row-level) fingerprint: SHA256 of sorted target_sample_id strings
for L72 + TEST + WB0_valid + included_common_population windows.

This fingerprint is used for Phase47 metric population validation and is distinct from
the row-level split_manifest.test_fingerprint (which is SHA256 of row specification strings).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from course_work.utils.artifacts import get_project_root


FINAL_TEST_POP_VERSION = "FINAL_TEST_POP-v1"
EXPECTED_TEST_WINDOW_COUNT = 2961  


def compute_test_population_fingerprint(
    window_index: pd.DataFrame,
    lookback_steps: int = 72,
    boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
) -> str:
    """Compute the canonical FINAL_TEST_POP-v1 fingerprint.

    The fingerprint is SHA256 over sorted JSON-encoded list of target_sample_id strings.
    This is the WINDOW-level fingerprint used for Phase47 metric validation.

    Note: This differs from split_manifest.test_fingerprint which is row-level
    (SHA256 over "row_index|timestamp|split_id" specification strings).

    Args:
        window_index: Loaded window_index DataFrame.
        lookback_steps: Must be 72 for Phase47.
        boundary_protocol: Must be WB0_CONTEXT_CARRY_OVER for Phase47.

    Returns:
        64-character hex SHA256 fingerprint of sorted target_sample_id list.
    """
    if lookback_steps != 72:
        raise ValueError(f"FINAL_TEST_POP-v1 requires lookback=72, got {lookback_steps}")
    if boundary_protocol != "WB0_CONTEXT_CARRY_OVER":
        raise ValueError(
            f"FINAL_TEST_POP-v1 requires WB0, got {boundary_protocol}"
        )

    valid_col = "WB0_valid"
    mask = (
        (window_index["lookback_steps"] == lookback_steps)
        & (window_index["target_split_id"] == "TEST")
        & (window_index[valid_col] == True)
        & (window_index["included_common_population"] == True)
    )
    target_ids = sorted(
        str(x) for x in window_index.loc[mask, "target_sample_id"].tolist()
    )
    if len(target_ids) == 0:
        raise ValueError("FINAL_TEST_POP-v1: no valid Test windows found")
    if len(target_ids) != EXPECTED_TEST_WINDOW_COUNT:
        raise ValueError(
            f"FINAL_TEST_POP-v1: expected {EXPECTED_TEST_WINDOW_COUNT} windows, got {len(target_ids)}"
        )
    blob = json.dumps(target_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def materialize_final_test_pop_v1(
    project_root: Path | None = None,
) -> dict[str, any]:
    """Materialize FINAL_TEST_POP-v1 and return population metadata.

    Args:
        project_root: COURSE_WORK root.

    Returns:
        Dict with keys:
            - test_window_count: int (expected: 2961)
            - test_population_fingerprint: str (SHA256)
            - target_sample_ids: list[str] (sorted)
            - target_timestamps: list[str] (sorted by target_sample_id order)
            - first_target_timestamp: str
            - last_target_timestamp: str
            - lookback: int (72)
            - horizon: int (1)
            - boundary_protocol: str
            - split: str ("TEST")
            - status: str
    """
    root = project_root or get_project_root()
    window_index_path = root / "artifacts" / "windows" / "window_index.csv"

    if not window_index_path.exists():
        raise FileNotFoundError(
            f"window_index.csv not found at {window_index_path}. "
            "Ensure Phase10 windows are materialized."
        )

    df = pd.read_csv(window_index_path)
    valid_col = "WB0_valid"
    mask = (
        (df["lookback_steps"] == 72)
        & (df["target_split_id"] == "TEST")
        & (df[valid_col] == True)
        & (df["included_common_population"] == True)
    )
    sub = df.loc[mask].sort_values("target_sample_id")

    target_ids = sorted(str(x) for x in sub["target_sample_id"].tolist())
    ts_map = dict(zip(sub["target_sample_id"].astype(str), sub["target_timestamp"]))
    timestamps = [ts_map[tid] for tid in target_ids]

    fp = compute_test_population_fingerprint(df, lookback_steps=72)

    return {
        "population_id": FINAL_TEST_POP_VERSION,
        "version": FINAL_TEST_POP_VERSION,
        "test_window_count": len(target_ids),
        "test_population_fingerprint": fp,
        "target_sample_ids": target_ids,
        "target_timestamps": timestamps,
        "first_target_timestamp": timestamps[0],
        "last_target_timestamp": timestamps[-1],
        "lookback": 72,
        "horizon": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "split": "TEST",
        "status": "PASS",
    }
