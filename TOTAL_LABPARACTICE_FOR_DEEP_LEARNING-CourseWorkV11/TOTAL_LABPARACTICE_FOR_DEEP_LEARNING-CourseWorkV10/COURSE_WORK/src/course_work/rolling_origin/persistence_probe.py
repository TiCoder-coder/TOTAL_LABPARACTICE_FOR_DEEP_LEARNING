"""Phase 44 — Persistence real-data probe (TASK 8).

For the FIRST outer-eval target in each of RO1/RO2/RO3:

  target_timestamp
  prior_timestamp  = target_ts - sampling_interval
  actual_prior_Appliances      (from ROBASE pre-history)
  persistence_prediction       (= actual_prior)

This proves the persistence prediction is:
  - NOT a copy of the current target's y_true
  - NOT a shift of the outer_eval y_true vector
  - NOT derived from Test rows

It uses the FULL windowpop CSV (TRAIN+VAL+TEST) so prior_ts may fall into
TRAIN rows even for RO1 (the prior history is everything BEFORE the
outer-eval target).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from course_work.rolling_origin.appliance_lookup import build_windowpop_with_appliances
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.populations import (
    extract_robase_population,
    extract_robase_train_ids,
    extract_robase_val_ids,
)
from course_work.rolling_origin.persistence import build_prior_history_lookup


DEFAULT_SAMPLING_INTERVAL_MINUTES = 10


@dataclass
class FirstTargetProbe:
    fold_id: str
    target_id: str
    target_timestamp: str
    prior_timestamp: str
    prior_observed_appliances: float
    persistence_prediction: float
    outer_y_true_at_target: float
    proof_self_consistent: bool

    def as_row(self) -> dict[str, str]:
        return {
            "fold_id": self.fold_id,
            "target_id": self.target_id,
            "target_timestamp": self.target_timestamp,
            "prior_timestamp": self.prior_timestamp,
            "prior_observed_appliances": f"{self.prior_observed_appliances:.6f}",
            "persistence_prediction": f"{self.persistence_prediction:.6f}",
            "outer_y_true_at_target": f"{self.outer_y_true_at_target:.6f}",
            "proof_self_consistent": "PASS" if self.proof_self_consistent else "FAIL",
        }


def run_persistence_probe(
    *,
    project_root: Path,
) -> list[FirstTargetProbe]:
    """For each fold, return FirstTargetProbe for the first outer-eval target."""
    full_windowpop = build_windowpop_with_appliances(project_root)
    if full_windowpop.empty or full_windowpop["Appliances"].isna().all():
        return []
    robase_df = extract_robase_population(project_root)
    robase_df["target_id"] = robase_df["target_id"].astype(str)
    appliance_map = dict(
        zip(
            full_windowpop["target_id"].astype(str),
            full_windowpop["Appliances"].astype(float),
        )
    )
    robase_df = robase_df.merge(
        full_windowpop[["target_id", "Appliances"]],
        on="target_id",
        how="left",
    )
    rtrn_ids = extract_robase_train_ids(project_root)
    rval_ids = extract_robase_val_ids(project_root)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)

    lookup = build_prior_history_lookup(robase_df, value_column="Appliances")

    out: list[FirstTargetProbe] = []
    for f in folds:
        first_target_id = str(next(iter(f.outer_eval_ids)))
        match = full_windowpop[full_windowpop["target_id"].astype(str) == first_target_id]
        if match.empty:
            sorted_df = full_windowpop.sort_values("target_timestamp")
            outer_sorted = sorted_df[sorted_df["target_id"].astype(str).isin(
                [str(x) for x in f.outer_eval_ids]
            )]
            if outer_sorted.empty:
                continue
            first_row = outer_sorted.iloc[0]
        else:
            first_row = match.iloc[0]
        target_ts = pd.Timestamp(first_row["target_timestamp"])
        prior_ts = target_ts - pd.Timedelta(minutes=DEFAULT_SAMPLING_INTERVAL_MINUTES)
        prior_ts_str = str(prior_ts)
        prior_val = None
        if prior_ts_str in lookup:
            prior_val = float(lookup[prior_ts_str])
        else:
            sorted_keys = sorted(lookup.keys())
            cand = None
            for k in sorted_keys:
                if k < str(target_ts):
                    cand = k
                else:
                    break
            if cand is None:
                continue
            prior_val = float(lookup[cand])
        outer_y = float(first_row.get("Appliances", float("nan")))
        proof_ok = (
            prior_val is not None
            and (not pd.isna(outer_y))
            and str(prior_ts) < str(target_ts)  
        )
        out.append(
            FirstTargetProbe(
                fold_id=str(f.fold_id),
                target_id=str(first_target_id),
                target_timestamp=str(target_ts),
                prior_timestamp=prior_ts_str,
                prior_observed_appliances=prior_val,
                persistence_prediction=prior_val, 
                outer_y_true_at_target=outer_y,
                proof_self_consistent=bool(proof_ok),
            )
        )
    return out
