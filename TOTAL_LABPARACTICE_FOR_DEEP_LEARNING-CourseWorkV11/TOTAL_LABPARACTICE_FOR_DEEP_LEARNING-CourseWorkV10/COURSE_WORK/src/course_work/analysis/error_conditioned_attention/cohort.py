"""Phase 56 - cohort assignment (frozen before attention join).

Builds:
- Seed-specific rank-based 20/60/20 error cohorts (LOW/MID/HIGH)
- Exact rank-based 10 error deciles (DECILE_1..DECILE_10)
- Shared-hardness cohorts (derived from mean abs error across 3 seeds)
- Residual sign groups (UNDER/OVER/ZERO)
- Assignment SHA256 (frozen before attention join)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .sources import (
    FrozenSources56,
    SEEDS,
    N_TEST,
    COHORT_RULE,
    COHORT_EDGE_FRACTION,
    DECILE_COUNT,
)


@dataclass
class CohortAssignmentResult:
    """Result of cohort assignment computation."""
    assignment_df: pd.DataFrame  
    shared_hardness_df: pd.DataFrame  
    per_seed_counts: dict[int, dict[str, int]]
    shared_counts: dict[str, int]
    assignment_sha256: str
    shared_sha256: str


def _stable_rank(df: pd.DataFrame, primary: str, tie_break: list[str]) -> pd.DataFrame:
    """Return df with column `rank_asc` (zero-based, deterministic)."""
    return df.sort_values([primary] + tie_break, kind="mergesort").reset_index(drop=True).assign(rank_asc=lambda d: range(len(d)))


def assign_seed_cohorts(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply rank-based 20/60/20 + deciles + sign group to one seed's targets."""
    n = len(seed_df)
    n_edge = max(1, int((COHORT_EDGE_FRACTION * n) // 1))
    sorted_df = seed_df.sort_values(
        ["absolute_error_wh", "target_timestamp", "target_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    sorted_df["seed_error_rank_asc"] = range(n)

    cohort = []
    for r in range(n):
        if r < n_edge:
            cohort.append("LOW_ERROR")
        elif r >= n - n_edge:
            cohort.append("HIGH_ERROR")
        else:
            cohort.append("MID_ERROR")
    sorted_df["seed_error_cohort"] = cohort

    deciles = []
    for r in range(n):
        d = 1 + (10 * r) // n
        if d > 10:
            d = 10
        deciles.append(d)
    sorted_df["seed_error_decile"] = deciles

    sign_group = []
    high_err_sign_group = []
    for _, row in sorted_df.iterrows():
        e = float(row["residual_wh"])
        if e > 0:
            sign_group.append("UNDER")
            if row["seed_error_cohort"] == "HIGH_ERROR":
                high_err_sign_group.append("HIGH_UNDER")
            else:
                high_err_sign_group.append("")
        elif e < 0:
            sign_group.append("OVER")
            if row["seed_error_cohort"] == "HIGH_ERROR":
                high_err_sign_group.append("HIGH_OVER")
            else:
                high_err_sign_group.append("")
        else:
            sign_group.append("ZERO")
            high_err_sign_group.append("")
    sorted_df["residual_sign_group"] = sign_group
    sorted_df["high_error_sign_group"] = high_err_sign_group
    return sorted_df


def build_cohort_assignment(sources: FrozenSources56) -> CohortAssignmentResult:
    """Build the full cohort assignment for all seeds.

    Steps:
    1. Read Phase 49 residuals (residual_long_table.csv has columns:
       target_id, target_timestamp, seed, y_true_wh, y_pred_wh, residual_wh,
       absolute_error_wh, residual_sign, source_prediction_sha256).
    2. Derive shared_hardness per target_id = mean of |residual_wh| across 3 seeds.
    3. Apply seed-specific rank-based 20/60/20 + deciles + sign group.
    4. Build shared cohort assignment based on shared_hardness (same rules).
    5. Compute assignment SHA256 (frozen before attention join).
    """
    res_df = pd.DataFrame(sources.residual_long)
    res_df = res_df[res_df["seed"].astype(str).isin([str(s) for s in SEEDS])].copy()
    res_df["residual_wh"] = res_df["residual_wh"].astype(float)
    res_df["absolute_error_wh"] = res_df["absolute_error_wh"].astype(float)
    res_df["seed"] = res_df["seed"].astype(int)
    res_df["target_id"] = res_df["target_id"].astype(str)
    res_df["target_timestamp"] = res_df["target_timestamp"].astype(str)

    shared = res_df.groupby("target_id")["absolute_error_wh"].mean().reset_index()
    shared.columns = ["target_id", "shared_hardness"]

    assigned_parts: list[pd.DataFrame] = []
    per_seed_counts: dict[int, dict[str, int]] = {}
    for seed in SEEDS:
        seed_df = res_df[res_df["seed"] == seed].copy()
        n = len(seed_df)
        assert n == N_TEST, f"seed {seed} has {n} targets, expected {N_TEST}"
        assigned = assign_seed_cohorts(seed_df)
        # Counts
        c_low = int((assigned["seed_error_cohort"] == "LOW_ERROR").sum())
        c_mid = int((assigned["seed_error_cohort"] == "MID_ERROR").sum())
        c_high = int((assigned["seed_error_cohort"] == "HIGH_ERROR").sum())
        per_seed_counts[seed] = {"LOW_ERROR": c_low, "MID_ERROR": c_mid, "HIGH_ERROR": c_high}
        assigned_parts.append(assigned)

    assignment_long = pd.concat(assigned_parts, ignore_index=True)

    shared_sorted = shared.sort_values(
        ["shared_hardness", "target_id"],
        kind="mergesort",
    ).reset_index(drop=True).assign(shared_error_rank_asc=lambda d: range(len(d)))

    n = len(shared_sorted)
    n_edge = max(1, int((COHORT_EDGE_FRACTION * n) // 1))
    s_cohort = []
    for r in range(n):
        if r < n_edge:
            s_cohort.append("SHARED_LOW_ERROR")
        elif r >= n - n_edge:
            s_cohort.append("SHARED_HIGH_ERROR")
        else:
            s_cohort.append("SHARED_MID_ERROR")
    shared_sorted["shared_error_cohort"] = s_cohort
    s_deciles = []
    for r in range(n):
        d = 1 + (10 * r) // n
        if d > 10:
            d = 10
        s_deciles.append(d)
    shared_sorted["shared_error_decile"] = s_deciles

    shared_counts = {
        "SHARED_LOW_ERROR": s_cohort.count("SHARED_LOW_ERROR"),
        "SHARED_MID_ERROR": s_cohort.count("SHARED_MID_ERROR"),
        "SHARED_HIGH_ERROR": s_cohort.count("SHARED_HIGH_ERROR"),
    }

    shared_to_merge = shared_sorted[[
        "target_id", "shared_hardness", "shared_error_rank_asc",
        "shared_error_cohort", "shared_error_decile",
    ]]
    assignment_long = assignment_long.merge(shared_to_merge, on="target_id", how="left")

    assignment_long["status"] = "OK"

    sha_input = assignment_long.sort_values(["seed", "target_id"]).to_csv(index=False).encode("utf-8")
    assignment_sha = hashlib.sha256(sha_input).hexdigest()

    shared_sha = hashlib.sha256(shared_sorted.sort_values("target_id").to_csv(index=False).encode("utf-8")).hexdigest()

    return CohortAssignmentResult(
        assignment_df=assignment_long,
        shared_hardness_df=shared_sorted,
        per_seed_counts=per_seed_counts,
        shared_counts=shared_counts,
        assignment_sha256=assignment_sha,
        shared_sha256=shared_sha,
    )
