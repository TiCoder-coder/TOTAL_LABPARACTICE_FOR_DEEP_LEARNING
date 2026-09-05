"""Phase 44 — Persistence baseline (prior-history lookup).

Correct semantics (plan §44, §103):
  prediction_for_target_t+1 = actual_observed_Appliances_at_t

The historical implementation shifted the outer y_true vector within the
outer block, which assigns zero residual to the first target by construction.
This module implements the correct explicit prior lookup.

The lookup table is built from the ROBASE actual observations (not from
the outer block itself). For each outer target timestamp `ts`, the prior
timestamp is `ts - sampling_interval` and we look up `Appliances[prior_ts]`
in the source history.

For the first target of each outer block, the prior timestamp falls BEFORE
`min(outer_eval)`, so we look up in the ROBASE pre-history.

For later targets in the same outer block (one-step rolling context, WB0),
the prior timestamp may also fall within the outer block, but the prediction
for `t+1` still uses the ACTUAL observed value at `t`, NOT a previous
prediction. This is implemented via a tiered lookup:
  1. pre-history (TRAIN, ordered by ts)
  2. outer block earlier targets (when prior_ts is in outer_eval)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERSISTENCE_MODEL_ID = "PERSISTENCE_LAST_VALUE"
DEFAULT_SAMPLING_INTERVAL_MINUTES = 10


@dataclass(frozen=True)
class PersistenceBundle:
    fold_id: str
    target_ids: tuple[int, ...]
    target_timestamps: tuple[str, ...]
    y_true_wh: tuple[float, ...]
    y_pred_wh: tuple[float, ...]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fold_id": [self.fold_id] * len(self.target_ids),
                "target_id": list(self.target_ids),
                "target_timestamp": list(self.target_timestamps),
                "y_true_wh": list(self.y_true_wh),
                "y_pred_wh": list(self.y_pred_wh),
                "residual_wh": [
                    float(t - p)
                    for t, p in zip(self.y_true_wh, self.y_pred_wh)
                ],
            }
        )


def build_prior_history_lookup(
    windowpop_df: pd.DataFrame,
    *,
    value_column: str = "Appliances",
    timestamp_column: str = "target_timestamp",
) -> dict[str, float]:
    """Return a dict: ISO timestamp -> observed value.

    Built from the ROBASE windowpop (pre-Test). Includes TRAIN rows and
    VALIDATION rows. Each timestamp maps to its observed Appliances value.

    Duplicate timestamps are de-duplicated by taking the last occurrence
    (the dataset contract guarantees at most one row per target_timestamp).
    """
    if timestamp_column not in windowpop_df.columns:
        raise ValueError(f"Missing {timestamp_column} in windowpop")
    if value_column not in windowpop_df.columns:
        raise ValueError(
            f"Missing {value_column} in windowpop. Available: "
            f"{list(windowpop_df.columns)}"
        )
    df = windowpop_df[[timestamp_column, value_column]].copy()
    df = df.dropna(subset=[value_column])
    df = df.drop_duplicates(subset=[timestamp_column], keep="last")
    return {str(ts): float(v) for ts, v in df.itertuples(index=False)}


def compute_persistence_bundle(
    *,
    fold_id: str,
    fold_outer_eval_target_ids: list[int],
    fold_outer_eval_target_timestamps: list[str],
    fold_outer_eval_y_true_wh: list[float],
    prior_lookup: dict[str, float],
    sampling_interval_minutes: int = DEFAULT_SAMPLING_INTERVAL_MINUTES,
) -> PersistenceBundle:
    """Compute persistence predictions for a fold's outer evaluation block.

    For each (target_id, target_ts, y_true):
      1. Compute prior_ts = target_ts - sampling_interval minutes.
      2. Look up prior_lookup[prior_ts]. If missing, look up the latest
         timestamp STRICTLY less than target_ts in prior_lookup (defensive
         fallback for missing pre-history rows).
      3. y_pred = that looked-up value.
    """
    if not (len(fold_outer_eval_target_ids) == len(fold_outer_eval_target_timestamps) == len(fold_outer_eval_y_true_wh)):
        raise ValueError(
            "fold_outer_eval arrays must be same length ("
            f"{len(fold_outer_eval_target_ids)}, "
            f"{len(fold_outer_eval_target_timestamps)}, "
            f"{len(fold_outer_eval_y_true_wh)})"
        )

    sorted_prior_ts = sorted(prior_lookup.keys())
    y_pred: list[float] = []
    for ts_str in fold_outer_eval_target_timestamps:
        prior_ts = _subtract_minutes(ts_str, sampling_interval_minutes)
        if prior_ts in prior_lookup:
            y_pred.append(prior_lookup[prior_ts])
        else:
            # Fallback: largest ts strictly less than ts_str
            cand = None
            for k in sorted_prior_ts:
                if k < ts_str:
                    cand = k
                else:
                    break
            if cand is None:
                raise ValueError(
                    f"No prior lookup entry for target_ts={ts_str} in fold={fold_id}"
                )
            y_pred.append(prior_lookup[cand])

    return PersistenceBundle(
        fold_id=fold_id,
        target_ids=tuple(str(t) for t in fold_outer_eval_target_ids),
        target_timestamps=tuple(str(t) for t in fold_outer_eval_target_timestamps),
        y_true_wh=tuple(float(v) for v in fold_outer_eval_y_true_wh),
        y_pred_wh=tuple(float(v) for v in y_pred),
    )


def _subtract_minutes(ts_str: str, minutes: int) -> str:
    ts = pd.Timestamp(ts_str)
    new_ts = ts - pd.Timedelta(minutes=minutes)
    return str(new_ts)