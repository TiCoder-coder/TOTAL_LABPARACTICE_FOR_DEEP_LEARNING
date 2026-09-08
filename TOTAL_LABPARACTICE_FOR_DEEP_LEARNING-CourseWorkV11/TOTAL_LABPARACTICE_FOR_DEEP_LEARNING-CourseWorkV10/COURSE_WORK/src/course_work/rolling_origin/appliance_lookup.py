"""Phase 44 — Join windowpop target_ids with actual Appliances values.

The WINDOWPOP-v1 CSV is metadata only (no Appliances column). To obtain
the actual Appliances for a target we join through:

  WINDOWPOP.target_id (TGT_NNNNNNNN) → numeric suffix = timeline_target →
  raw_row_index in
  data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv
  → row.Appliances
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_NUMERIC_SUFFIX = re.compile(r"(\d+)$")


def _numeric_suffix(tid) -> int:
    s = str(tid)
    m = _NUMERIC_SUFFIX.search(s)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return -1
    return -1


def load_appliances_lookup(project_root: Path) -> dict[int, float]:
    """Return a dict mapping raw_row_index → observed Appliances Wh.

    Reads the validated, sha256-checked FEATURES-v1 derived file.
    """
    feature_path = (
        project_root / "data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv"
    )
    if not feature_path.exists():
        return {}
    df = pd.read_csv(feature_path)
    return {int(r): float(a) for r, a in zip(df["raw_row_index"], df["Appliances"])}


def build_windowpop_with_appliances(
    project_root: Path,
) -> pd.DataFrame:
    """Return the WINDOWPOP-v1 frame with an Appliances column.

    Returns an empty dataframe if the lookup fails (caller decides whether
    to fall back to dataset-level runtime lookup).
    """
    windowpop_path = project_root / "artifacts/windows/common_target_population.csv"
    if not windowpop_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(windowpop_path)
    if "target_id" not in df.columns:
        df = df.rename(columns={"target_sample_id": "target_id"})
    df["target_id"] = df["target_id"].astype(str)
    lookup = load_appliances_lookup(project_root)
    if not lookup:
        df["Appliances"] = float("nan")
    else:
        df["Appliances"] = df["target_id"].map(
            lambda tid: lookup.get(_numeric_suffix(tid), float("nan"))
        )
    return df
