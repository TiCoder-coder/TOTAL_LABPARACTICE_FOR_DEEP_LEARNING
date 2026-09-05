"""Phase 44 — ROBASE population extraction.

ROBASE-v1 = the pre-Test target population used by rolling-origin evaluation.

Population construction is a pure, deterministic, file-only operation:
it does NOT load full datasets, scale features, or build windows.
It operates on the WINDOWPOP-v1 CSV:

  artifacts/windows/common_target_population.csv

Columns:
  target_sample_id      (target_id for ROBASE; int or str like "TGT_00000144")
  target_timestamp      (ISO timestamp)
  target_split_id       (TRAIN / VALIDATION / TEST)
  continuity_segment_id
  valid_L36 / L72 / L144
  included_common_population (bool)

The population is split into:

  RTRN_IDS : sorted (target_id) where target_split_id == "TRAIN"
  RVAL_IDS : sorted (target_id) where target_split_id == "VALIDATION"

Both halves are asserted disjoint, ordered, and NOT touching TEST.

Target IDs may be strings (e.g. "TGT_00000144") or ints. They are preserved
verbatim across the pipeline; fingerprints hash the sorted string form.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROBASE_VERSION = "ROBASE-v1"
WINDOWPOP_VERSION = "WINDOWPOP-v1"

DEFAULT_WINDOWPOP_PATH = "artifacts/windows/common_target_population.csv"
ALLOWED_SPLIT_IDS = frozenset({"TRAIN", "VALIDATION"})


def _load_windowpop(windowpop_path: Path) -> pd.DataFrame:
    df = pd.read_csv(windowpop_path)
    required_cols = {"target_sample_id", "target_timestamp", "target_split_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Windowpop csv {windowpop_path} missing required columns: "
            f"{sorted(missing)}"
        )
    if "target_id" not in df.columns:
        df = df.rename(columns={"target_sample_id": "target_id"})
    # Coerce target_id to string for stability across CSV/parquet boundaries.
    df["target_id"] = df["target_id"].astype(str)
    return df


def _find_windowpop(project_root: Path) -> Path:
    candidate = project_root / DEFAULT_WINDOWPOP_PATH
    if not candidate.exists():
        raise FileNotFoundError(
            f"No windowpop csv at {candidate}. Phase 44 requires WINDOWPOP-v1."
        )
    return candidate


def extract_robase_population(project_root: Path) -> pd.DataFrame:
    """Return the ROBASE-v1 windowpop dataframe restricted to TRAIN+VALIDATION.

    Test rows are EXCLUDED (test firewall).
    """
    wp_path = _find_windowpop(project_root)
    df = _load_windowpop(wp_path)
    splits_present = set(df["target_split_id"].astype(str).unique())
    illegal_splits = splits_present - ALLOWED_SPLIT_IDS - {"TEST"}
    if illegal_splits:
        raise ValueError(
            f"WINDOWPOP contains illegal splits for ROBASE: {sorted(illegal_splits)}. "
            f"ROBASE must be a subset of TRAIN + VALIDATION. Found: {sorted(splits_present)}."
        )
    if "TEST" in splits_present:
        # Explicit TEST exclusion — this is the test firewall.
        df = df[df["target_split_id"].astype(str) != "TEST"].copy()
    df = df.sort_values("target_timestamp").reset_index(drop=True)
    return df


def extract_robase_train_ids(project_root: Path) -> list[str]:
    df = extract_robase_population(project_root)
    return (
        df.loc[df["target_split_id"].astype(str) == "TRAIN", "target_id"]
        .astype(str)
        .tolist()
    )


def extract_robase_val_ids(project_root: Path) -> list[str]:
    df = extract_robase_population(project_root)
    return (
        df.loc[df["target_split_id"].astype(str) == "VALIDATION", "target_id"]
        .astype(str)
        .tolist()
    )


def compute_population_fingerprint(target_ids: Iterable) -> str:
    """Deterministic sha256 over an ordered target_id list.

    IDs are normalized to strings so the fingerprint is stable regardless of
    whether the upstream CSV used ints or strings.
    """
    ordered = sorted(str(t) for t in target_ids)
    blob = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_raw_row_count(target_ids: Iterable, windowpop_df: pd.DataFrame) -> int:
    """Count raw windowpop rows (not target_ids) for a given subset."""
    return int(windowpop_df["target_id"].isin([str(t) for t in target_ids]).sum())