"""Phase 56 - join error cohort assignment with Phase 54 attention metrics.

Key: (seed, target_id)
Join rule: 3 seeds * 2961 targets * 2 layers * 4 heads = 71,064 rows expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .sources import FrozenSources56, SEEDS, NUM_LAYERS, NUM_HEADS, N_TEST


@dataclass
class JoinResult:
    """Result of join."""
    joined_df: pd.DataFrame  # rows = 3*2961*2*4 = 71,064
    join_audit: list[dict[str, Any]]
    expected_rows: int


def join_attention_with_cohorts(sources: FrozenSources56, assignment_df: pd.DataFrame) -> JoinResult:
    """Join attention metrics with cohort assignment.

    Phase 54 last_query_metrics_long.csv has columns:
    seed, target_id, target_timestamp, layer_idx0, head_idx0, ...,
    normalized_entropy, expected_lag_minutes, recent_1h_mass,
    recent_6h_mass, top5_mass, lag80_minutes, ...
    """
    metrics = pd.DataFrame(sources.last_query_metrics_long)
    metrics["seed"] = metrics["seed"].astype(int)
    metrics["layer_idx0"] = metrics["layer_idx0"].astype(int)
    metrics["head_idx0"] = metrics["head_idx0"].astype(int)
    metrics["target_id"] = metrics["target_id"].astype(str)

    # Cast core metrics to float
    for col in [
        "normalized_entropy", "expected_lag_minutes", "recent_1h_mass",
        "recent_6h_mass", "top5_mass", "lag80_minutes",
    ]:
        metrics[col] = metrics[col].astype(float)

    expected = len(SEEDS) * N_TEST * NUM_LAYERS * NUM_HEADS

    # Assignment df has columns: seed, target_id, target_timestamp, ..., cohort info
    assignment = assignment_df.copy()
    assignment["seed"] = assignment["seed"].astype(int)
    assignment["target_id"] = assignment["target_id"].astype(str)

    # Cohort join columns
    cohort_cols = [
        "seed", "target_id",
        "residual_wh", "absolute_error_wh",
        "seed_error_rank_asc", "seed_error_cohort", "seed_error_decile",
        "shared_hardness", "shared_error_rank_asc",
        "shared_error_cohort", "shared_error_decile",
        "residual_sign_group", "high_error_sign_group",
    ]
    assignment_subset = assignment[cohort_cols].copy()

    joined = metrics.merge(assignment_subset, on=["seed", "target_id"], how="inner")

    audit: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_metrics = metrics[metrics["seed"] == seed]
        seed_assignment = assignment[assignment["seed"] == seed]
        n_metrics_targets = int(seed_metrics["target_id"].nunique())
        n_assign_targets = int(seed_assignment["target_id"].nunique())
        seed_joined = joined[joined["seed"] == seed]
        observed_rows = len(seed_joined)
        expected_rows = N_TEST * NUM_LAYERS * NUM_HEADS
        ids_metrics = set(seed_metrics["target_id"].astype(str))
        ids_assign = set(seed_assignment["target_id"].astype(str))
        # Unmatched in attention: present in metrics but not in assignment
        unmatched_attn = len(ids_metrics - ids_assign)
        # Unmatched in error: present in assignment but not in metrics
        unmatched_err = len(ids_assign - ids_metrics)
        n_dup_err = int(seed_assignment["target_id"].duplicated().sum())
        # Missing head rows = expected - observed
        missing_head_rows = expected_rows - observed_rows
        audit.append({
            "seed": seed,
            "error_target_count": n_assign_targets,
            "attention_target_count": n_metrics_targets,
            "layer_count": NUM_LAYERS,
            "head_count": NUM_HEADS,
            "expected_join_rows": expected_rows,
            "observed_join_rows": observed_rows,
            "unmatched_error_targets": unmatched_err,
            "unmatched_attention_targets": unmatched_attn,
            "duplicate_error_rows": n_dup_err,
            "missing_head_rows": missing_head_rows,
            "status": "PASS" if (
                n_metrics_targets == N_TEST and
                n_assign_targets == N_TEST and
                observed_rows == expected_rows and
                unmatched_err == 0 and unmatched_attn == 0 and
                n_dup_err == 0 and missing_head_rows == 0
            ) else "FAIL",
        })

    return JoinResult(
        joined_df=joined,
        join_audit=audit,
        expected_rows=expected,
    )
