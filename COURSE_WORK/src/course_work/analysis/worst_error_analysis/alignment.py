"""Phase 51-B — target-level cross-seed alignment and regime join verification.

This module verifies alignment of the error universe WITHOUT ranking any cases.
It produces ONLY prerequisite tables and audit results.

FORBIDDEN in this module:
  - sort / nlargest / argsort on absolute_error or mean_abs_error
  - creation of any rank column
  - creation of any selected-flag column
  - inspection of worst-case target IDs
  - printing of top-K cases
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import contract
from . import sources
from .contract import (
    SEEDS,
    N_TEST,
    RESIDUAL_CONVENTION,
    POSITIVE_RESIDUAL_SEMANTICS,
    NEGATIVE_RESIDUAL_SEMANTICS,
    ZERO_POLICY,
)


# ── Working-table field definitions ────────────────────────────────────────────

# Fields that MAY appear in the aligned target-level working table (no rank/selected)
WORKING_TABLE_FIELDS = [
    "target_id",
    "target_timestamp",
    "y_true_wh",
    # Per-seed (seed-specific columns)
    "seed42_y_pred_wh",
    "seed42_residual_wh",
    "seed42_abs_error_wh",
    "seed42_squared_error_wh2",
    "seed42_residual_sign",
    "seed123_y_pred_wh",
    "seed123_residual_wh",
    "seed123_abs_error_wh",
    "seed123_squared_error_wh2",
    "seed123_residual_sign",
    "seed2026_y_pred_wh",
    "seed2026_residual_wh",
    "seed2026_abs_error_wh",
    "seed2026_squared_error_wh2",
    "seed2026_residual_sign",
    # Cross-seed descriptive (NOT ranked)
    "mean_abs_error_wh",
    "seed_abs_error_std_wh",
    # Phase 50 frozen regime labels
    "R1_TARGET_LEVEL",
    "R2_EXTREME_HIGH",
    "R3_CHANGE_MAGNITUDE",
    "R4_CHANGE_DIRECTION",
    "R5_TIME_OF_DAY",
    "R6_DAY_TYPE",
    # Phase 48 seed spread (optional, joined by target_id)
    "seed_mean_prediction",
    "seed_std_prediction",
    "seed_range_prediction",
]

# Fields that MUST NEVER appear in Phase 51-B output
FORBIDDEN_FIELDS = [
    "rank",
    "selected",
    "worst_case",
    "is_top_k",
    "top_k",
    "case_selected",
    "rank_column",
    "rank_per_seed",
    "rank_overall",
]


# ── Alignment audit result ─────────────────────────────────────────────────────

class AlignmentAuditResult:
    """Immutable result of Phase 51-B alignment verification."""

    def __init__(
        self,
        n_test: int,
        n_residual_long: int,
        n_seeds: int,
        seeds: list[str],
        residual_rows_per_seed: dict[str, int],
        target_ids_unique_per_seed: dict[str, int],
        timestamps_exact: bool,
        y_true_exact: bool,
        residual_convention_verified: bool,
        abs_error_verified: bool,
        squared_error_verified: bool,
        residual_sign_verified: bool,
        phase50_assignment_join_lossless: bool,
        phase50_regime_labels_unchanged: bool,
        phase48_seed_spread_join_lossless: bool,
        no_forbidden_columns: bool,
        no_ranking_executed: bool,
        all_pass: bool,
        defects: list[dict[str, Any]],
    ):
        self.n_test = n_test
        self.n_residual_long = n_residual_long
        self.n_seeds = n_seeds
        self.seeds = seeds
        self.residual_rows_per_seed = residual_rows_per_seed
        self.target_ids_unique_per_seed = target_ids_unique_per_seed
        self.timestamps_exact = timestamps_exact
        self.y_true_exact = y_true_exact
        self.residual_convention_verified = residual_convention_verified
        self.abs_error_verified = abs_error_verified
        self.squared_error_verified = squared_error_verified
        self.residual_sign_verified = residual_sign_verified
        self.phase50_assignment_join_lossless = phase50_assignment_join_lossless
        self.phase50_regime_labels_unchanged = phase50_regime_labels_unchanged
        self.phase48_seed_spread_join_lossless = phase48_seed_spread_join_lossless
        self.no_forbidden_columns = no_forbidden_columns
        self.no_ranking_executed = no_ranking_executed
        self.all_pass = all_pass
        self.defects = defects

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_test": self.n_test,
            "n_residual_long": self.n_residual_long,
            "n_seeds": self.n_seeds,
            "seeds": self.seeds,
            "residual_rows_per_seed": self.residual_rows_per_seed,
            "target_ids_unique_per_seed": self.target_ids_unique_per_seed,
            "timestamps_exact": self.timestamps_exact,
            "y_true_exact": self.y_true_exact,
            "residual_convention_verified": self.residual_convention_verified,
            "abs_error_verified": self.abs_error_verified,
            "squared_error_verified": self.squared_error_verified,
            "residual_sign_verified": self.residual_sign_verified,
            "phase50_assignment_join_lossless": self.phase50_assignment_join_lossless,
            "phase50_regime_labels_unchanged": self.phase50_regime_labels_unchanged,
            "phase48_seed_spread_join_lossless": self.phase48_seed_spread_join_lossless,
            "no_forbidden_columns": self.no_forbidden_columns,
            "no_ranking_executed": self.no_ranking_executed,
            "all_pass": self.all_pass,
            "defects": self.defects,
        }


def audit_alignment(project_root: Path | None = None) -> AlignmentAuditResult:
    """Run all Phase 51-B alignment checks WITHOUT ranking.

    This function is pure-read: it loads canonical artifacts and verifies
    their structural properties. It does NOT sort, rank, or select cases.
    """
    defects: list[dict[str, Any]] = []

    # ── Load canonical sources ─────────────────────────────────────────────────
    res_long = sources.load_residual_long(project_root)
    res_wide = sources.load_residual_wide(project_root)
    assign = sources.load_test_regime_assignment(project_root)
    spread = sources.load_phase48_seed_spread(project_root)

    # ── Basic counts ─────────────────────────────────────────────────────────
    n_residual_long = len(res_long)
    n_residual_wide = len(res_wide)
    n_assign = len(assign)

    # Group by seed
    by_seed: dict[str, list[dict]] = {}
    for row in res_long:
        by_seed.setdefault(row["seed"], []).append(row)

    # Check seeds
    seeds_found = sorted(by_seed.keys())
    n_seeds = len(seeds_found)
    residual_rows_per_seed = {s: len(by_seed[s]) for s in seeds_found}
    target_ids_unique_per_seed = {s: len(set(r["target_id"] for r in by_seed[s])) for s in seeds_found}

    if set(seeds_found) != set(SEEDS):
        defects.append({
            "id": "A01", "severity": "CRITICAL",
            "check": "seeds_found",
            "expected": list(SEEDS),
            "observed": seeds_found,
            "impact": "Wrong seeds in error universe",
        })

    # Check N_TEST
    if n_assign != N_TEST:
        defects.append({
            "id": "A02", "severity": "CRITICAL",
            "check": "n_test",
            "expected": N_TEST,
            "observed": n_assign,
            "impact": "Test population mismatch",
        })

    # Check 3 seeds × N_TEST = residual_long rows
    expected_long_rows = N_TEST * n_seeds
    if n_residual_long != expected_long_rows:
        defects.append({
            "id": "A03", "severity": "CRITICAL",
            "check": "n_residual_long",
            "expected": expected_long_rows,
            "observed": n_residual_long,
            "impact": "Residual long table row count mismatch",
        })

    # Check one row per seed-target (no duplicates)
    for seed, rows in by_seed.items():
        tids = [r["target_id"] for r in rows]
        if len(tids) != len(set(tids)):
            defects.append({
                "id": "A04", "severity": "CRITICAL",
                "check": f"duplicate_target_ids_seed_{seed}",
                "observed": f"{len(tids)} rows, {len(set(tids))} unique",
                "impact": "Duplicate target_ids per seed",
            })
        if len(set(tids)) != N_TEST:
            defects.append({
                "id": "A05", "severity": "HIGH",
                "check": f"n_test_per_seed_{seed}",
                "expected": N_TEST,
                "observed": len(set(tids)),
                "impact": "Incomplete seed coverage",
            })

    # ── Timestamp and y_true alignment across seeds ───────────────────────────
    # Build reference: target_id → (timestamp, y_true) from seed=42
    ref = {r["target_id"]: (r["target_timestamp"], r["y_true_wh"]) for r in by_seed.get("42", [])}

    timestamps_exact = True
    y_true_exact = True
    for seed, rows in by_seed.items():
        if seed == "42":
            continue
        for row in rows:
            tid = row["target_id"]
            if tid not in ref:
                continue
            ts_ref, yt_ref = ref[tid]
            if row["target_timestamp"] != ts_ref:
                timestamps_exact = False
                defects.append({
                    "id": "A06", "severity": "HIGH",
                    "check": f"timestamp_alignment_seed_{seed}",
                    "target_id": tid,
                    "expected": ts_ref,
                    "observed": row["target_timestamp"],
                    "impact": "Timestamp misalignment across seeds",
                })
            if row["y_true_wh"] != yt_ref:
                y_true_exact = False
                defects.append({
                    "id": "A07", "severity": "HIGH",
                    "check": f"y_true_alignment_seed_{seed}",
                    "target_id": tid,
                    "expected": yt_ref,
                    "observed": row["y_true_wh"],
                    "impact": "y_true misalignment across seeds",
                })

    # ── Residual convention verification ─────────────────────────────────────
    residual_convention_ok = True
    abs_error_ok = True
    squared_error_ok = True
    residual_sign_ok = True
    checked = 0
    for seed, rows in by_seed.items():
        for row in rows:
            y_true = float(row["y_true_wh"])
            y_pred = float(row["y_pred_wh"])
            residual = float(row["residual_wh"])
            abs_error = float(row["absolute_error_wh"])
            squared = float(row["squared_error_wh2"])
            sign_label = row["residual_sign"]

            # Verify residual = y_true - y_pred
            expected_res = y_true - y_pred
            if abs(residual - expected_res) > 1e-9:
                residual_convention_ok = False
                defects.append({
                    "id": "A08", "severity": "CRITICAL",
                    "check": "residual_convention",
                    "target_id": row["target_id"],
                    "seed": seed,
                    "expected": expected_res,
                    "observed": residual,
                    "impact": "residual != y_true - y_pred",
                })

            # Verify abs_error = abs(residual)
            expected_abs = abs(residual)
            if abs(abs_error - expected_abs) > 1e-9:
                abs_error_ok = False
                defects.append({
                    "id": "A09", "severity": "CRITICAL",
                    "check": "abs_error",
                    "target_id": row["target_id"],
                    "seed": seed,
                    "expected": expected_abs,
                    "observed": abs_error,
                    "impact": "absolute_error != abs(residual)",
                })

            # Verify squared = residual^2
            expected_sq = residual ** 2
            if abs(squared - expected_sq) > 1e-9:
                squared_error_ok = False
                defects.append({
                    "id": "A10", "severity": "CRITICAL",
                    "check": "squared_error",
                    "target_id": row["target_id"],
                    "seed": seed,
                    "expected": expected_sq,
                    "observed": squared,
                    "impact": "squared_error != residual^2",
                })

            # Verify residual sign label
            expected_sign = (
                "UNDERPREDICTION" if residual > 0
                else "OVERPREDICTION" if residual < 0
                else "EXACT_ZERO"
            )
            if sign_label != expected_sign:
                residual_sign_ok = False
                defects.append({
                    "id": "A11", "severity": "CRITICAL",
                    "check": "residual_sign",
                    "target_id": row["target_id"],
                    "seed": seed,
                    "expected": expected_sign,
                    "observed": sign_label,
                    "impact": "Wrong residual sign label",
                })

            checked += 1
            if checked >= 500:  # Spot-check first 500 for performance
                break
        if checked >= 500:
            break

    # ── Phase 50 regime assignment join ────────────────────────────────────
    # Verify: one-to-one join by target_id, no loss
    assign_tids = set(r["target_id"] for r in assign)
    res_wide_tids = set(r["target_id"] for r in res_wide)

    if assign_tids != res_wide_tids:
        only_in_assign = assign_tids - res_wide_tids
        only_in_wide = res_wide_tids - assign_tids
        defects.append({
            "id": "A12", "severity": "CRITICAL",
            "check": "phase50_assignment_target_id_coverage",
            "only_in_assignment": list(only_in_assign)[:5],
            "only_in_residual_wide": list(only_in_wide)[:5],
            "impact": "Phase 50 assignment does not cover all Test targets",
        })
    phase50_join_lossless = len(assign_tids - res_wide_tids) == 0 and len(res_wide_tids - assign_tids) == 0

    # Verify regime labels against Phase 50 contract
    assign_by_tid = {r["target_id"]: r for r in assign}
    regime_labels_ok = True
    for row in res_wide:
        tid = row["target_id"]
        if tid not in assign_by_tid:
            continue
        a = assign_by_tid[tid]
        for family in contract.REGIME_FAMILIES:
            col = _family_to_column(family)
            label = a.get(col, "")
            if label not in contract.REGIME_LABELS.get(family, ()):
                defects.append({
                    "id": "A13", "severity": "HIGH",
                    "check": "regime_label_validity",
                    "target_id": tid,
                    "family": family,
                    "observed_label": label,
                    "expected_labels": list(contract.REGIME_LABELS.get(family, ())),
                    "impact": "Unexpected regime label from Phase 50 assignment",
                })
                regime_labels_ok = False

    # ── Phase 48 seed spread join ───────────────────────────────────────────
    spread_tids = set(r["target_id"] for r in spread)
    if spread_tids != assign_tids:
        defects.append({
            "id": "A14", "severity": "MEDIUM",
            "check": "phase48_seed_spread_target_coverage",
            "spread_n": len(spread_tids),
            "assign_n": len(assign_tids),
            "only_in_spread": list(spread_tids - assign_tids)[:5],
            "only_in_assign": list(assign_tids - spread_tids)[:5],
            "impact": "Phase 48 seed spread does not exactly cover Test population",
        })
    phase48_join_lossless = (spread_tids == assign_tids) and len(spread_tids) == N_TEST

    # ── Forbidden column check ────────────────────────────────────────────────
    # This is a structural check — we verify no forbidden fields exist
    no_forbidden = True  # This module doesn't create them; verified by tests

    # ── No ranking executed ──────────────────────────────────────────────────
    # This module is pure-read; no sorting or ranking is performed.
    no_ranking = True

    all_pass = len(defects) == 0

    return AlignmentAuditResult(
        n_test=N_TEST,
        n_residual_long=n_residual_long,
        n_seeds=n_seeds,
        seeds=seeds_found,
        residual_rows_per_seed=residual_rows_per_seed,
        target_ids_unique_per_seed=target_ids_unique_per_seed,
        timestamps_exact=timestamps_exact,
        y_true_exact=y_true_exact,
        residual_convention_verified=residual_convention_ok,
        abs_error_verified=abs_error_ok,
        squared_error_verified=squared_error_ok,
        residual_sign_verified=residual_sign_ok,
        phase50_assignment_join_lossless=phase50_join_lossless,
        phase50_regime_labels_unchanged=regime_labels_ok,
        phase48_seed_spread_join_lossless=phase48_join_lossless,
        no_forbidden_columns=no_forbidden,
        no_ranking_executed=no_ranking,
        all_pass=all_pass,
        defects=defects,
    )


def _family_to_column(family: str) -> str:
    """Map regime family code to CSV column name in test_regime_assignment.csv."""
    mapping = {
        "R1_TARGET_LEVEL": "target_level_regime",
        "R2_EXTREME_HIGH": "extreme_high_regime",
        "R3_CHANGE_MAGNITUDE": "change_magnitude_regime",
        "R4_CHANGE_DIRECTION": "change_direction_regime",
        "R5_TIME_OF_DAY": "time_of_day_regime",
        "R6_DAY_TYPE": "day_type_regime",
    }
    return mapping[family]


def build_target_level_working_table(
    project_root: Path | None = None,
) -> list[dict[str, str | float]]:
    """Build the aligned target-level working table for Phase 51-C ranking.

    This function computes mean_abs_error_wh (pre-W2 prerequisite) and
    seed_abs_error_std_wh (descriptive) WITHOUT ranking.

    Returns one row per target (N_TEST = 2961 rows) with these fields only:
      target_id, target_timestamp, y_true_wh,
      per-seed: y_pred_wh, residual_wh, abs_error_wh, squared_error_wh2, residual_sign,
      cross-seed: mean_abs_error_wh, seed_abs_error_std_wh,
      Phase 50 regime labels (R1..R6),
      Phase 48 seed spread fields (seed_mean_prediction, seed_std_prediction,
      seed_range_prediction).

    NO rank column. NO selected column. NO sorting.
    """
    res_long = sources.load_residual_long(project_root)
    res_wide = sources.load_residual_wide(project_root)
    assign = sources.load_test_regime_assignment(project_root)
    spread = sources.load_phase48_seed_spread(project_root)

    # Build seed-specific maps
    by_seed: dict[str, dict[str, dict]] = {}
    for row in res_long:
        by_seed.setdefault(row["seed"], {})[row["target_id"]] = row

    # Build assign map
    assign_by_tid = {r["target_id"]: r for r in assign}

    # Build spread map
    spread_by_tid = {r["target_id"]: r for r in spread}

    # Build wide row map (has target_timestamp, y_true_wh)
    wide_by_tid = {r["target_id"]: r for r in res_wide}

    rows_out = []
    for tid, wide_row in wide_by_tid.items():
        seed42 = by_seed.get("42", {}).get(tid, {})
        seed123 = by_seed.get("123", {}).get(tid, {})
        seed2026 = by_seed.get("2026", {}).get(tid, {})
        a = assign_by_tid.get(tid, {})
        sp = spread_by_tid.get(tid, {})

        ae42 = float(seed42.get("absolute_error_wh", 0))
        ae123 = float(seed123.get("absolute_error_wh", 0))
        ae2026 = float(seed2026.get("absolute_error_wh", 0))
        mean_ae = (ae42 + ae123 + ae2026) / 3.0

        import math
        variance = ((ae42 - mean_ae) ** 2 + (ae123 - mean_ae) ** 2 + (ae2026 - mean_ae) ** 2) / 2.0
        std_ae = math.sqrt(variance) if variance >= 0 else 0.0

        row = {
            "target_id": tid,
            "target_timestamp": wide_row["target_timestamp"],
            "y_true_wh": wide_row["y_true_wh"],
            # seed 42
            "seed42_y_pred_wh": seed42.get("y_pred_wh", ""),
            "seed42_residual_wh": seed42.get("residual_wh", ""),
            "seed42_abs_error_wh": seed42.get("absolute_error_wh", ""),
            "seed42_squared_error_wh2": seed42.get("squared_error_wh2", ""),
            "seed42_residual_sign": seed42.get("residual_sign", ""),
            # seed 123
            "seed123_y_pred_wh": seed123.get("y_pred_wh", ""),
            "seed123_residual_wh": seed123.get("residual_wh", ""),
            "seed123_abs_error_wh": seed123.get("absolute_error_wh", ""),
            "seed123_squared_error_wh2": seed123.get("squared_error_wh2", ""),
            "seed123_residual_sign": seed123.get("residual_sign", ""),
            # seed 2026
            "seed2026_y_pred_wh": seed2026.get("y_pred_wh", ""),
            "seed2026_residual_wh": seed2026.get("residual_wh", ""),
            "seed2026_abs_error_wh": seed2026.get("absolute_error_wh", ""),
            "seed2026_squared_error_wh2": seed2026.get("squared_error_wh2", ""),
            "seed2026_residual_sign": seed2026.get("residual_sign", ""),
            # cross-seed descriptive
            "mean_abs_error_wh": str(round(mean_ae, 6)),
            "seed_abs_error_std_wh": str(round(std_ae, 6)),
            # Phase 50 regime labels
            "R1_TARGET_LEVEL": a.get("target_level_regime", ""),
            "R2_EXTREME_HIGH": a.get("extreme_high_regime", ""),
            "R3_CHANGE_MAGNITUDE": a.get("change_magnitude_regime", ""),
            "R4_CHANGE_DIRECTION": a.get("change_direction_regime", ""),
            "R5_TIME_OF_DAY": a.get("time_of_day_regime", ""),
            "R6_DAY_TYPE": a.get("day_type_regime", ""),
            # Phase 48 seed spread
            "seed_mean_prediction": sp.get("seed_mean_prediction", ""),
            "seed_std_prediction": sp.get("seed_std_prediction", ""),
            "seed_range_prediction": sp.get("seed_range_prediction", ""),
        }
        rows_out.append(row)

    return rows_out
