"""Phase 51-F — deterministic casebook construction.

Casebook derives case_ids from frozen Phase 51-C ranking artifacts ONLY.
Each selected case row is keyed by a deterministic case_id of the form:

  CASE_<selection_family>_<seed|ALL>_rank<NN>_<target_id>

No manual selection. No random UUID. Rerun produces identical IDs.

Cross-family membership is preserved as a separate bridge: each case
appears once per selection-family membership. A master unique-case list
is also emitted (one row per unique target_id).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import contract
from ..utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _load_csv(project_root: Path, rel: str) -> list[dict[str, str]]:
    fp = project_root / PHASE51_DIR_REL / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _family_short(family: str) -> str:
    return {
        "W1_PER_SEED_WORST": "W1",
        "W2_SHARED_WORST": "W2",
        "W3_UNDERPREDICTION_WORST": "W3",
        "W4_OVERPREDICTION_WORST": "W4",
        "W3_SH_SHARED_ALL_UNDER": "W3SH",
        "W4_SH_SHARED_ALL_OVER": "W4SH",
    }.get(family, family)


def build_case_id(
    selection_family: str,
    seed: str,
    rank: str | int,
    target_id: str,
) -> str:
    """Deterministic case_id per (family, seed, rank, target_id)."""
    return f"CASE_{_family_short(selection_family)}_{seed}_rank{int(rank):03d}_{target_id}"


def load_frozen_cases(project_root: Path) -> list[dict[str, Any]]:
    """Load all frozen Phase 51-C selected cases and tag with case_id.

    Returns one record per (selection_family, seed, rank, target_id).
    """
    wt_fp = project_root / PHASE51_DIR_REL / "phase51_target_level_working_table.csv"
    wt = {r["target_id"]: r for r in csv.DictReader(
        wt_fp.open("r", encoding="utf-8", newline="")
    )}

    cases: list[dict[str, Any]] = []

    ranking_files: list[tuple[str, str, str]] = [
        # (selection_family, csv_rel, error_field)
        ("W1_PER_SEED_WORST", "worst_per_seed_top20.csv", "absolute_error_wh"),
        ("W2_SHARED_WORST", "worst_shared_top20.csv", "mean_abs_error_wh"),
        ("W3_UNDERPREDICTION_WORST", "worst_underprediction_top10.csv", "absolute_error_wh"),
        ("W4_OVERPREDICTION_WORST", "worst_overprediction_top10.csv", "absolute_error_wh"),
        ("W3_SH_SHARED_ALL_UNDER", "shared_all_under_top10.csv", "mean_abs_error_wh"),
        ("W4_SH_SHARED_ALL_OVER", "shared_all_over_top10.csv", "mean_abs_error_wh"),
    ]

    for family, rel, err_field in ranking_files:
        rows = _load_csv(project_root, rel)
        for r in rows:
            tid = r["target_id"]
            seed = r.get("seed", "ALL")
            rank = r["rank"]
            w = wt[tid]
            case_id = build_case_id(family, seed, rank, tid)
            case = {
                "case_id": case_id,
                "selection_family": family,
                "seed": seed,
                "rank": rank,
                "target_id": tid,
                "target_timestamp": r.get("target_timestamp", w["target_timestamp"]),
                "y_true_wh": r.get("y_true_wh", w["y_true_wh"]),
                "y_pred_wh": (
                    r.get(err_field, w.get(f"seed{seed}_y_pred_wh"))
                    if family in ("W1_PER_SEED_WORST", "W3_UNDERPREDICTION_WORST",
                                  "W4_OVERPREDICTION_WORST")
                    else r.get(err_field, "")
                ),
                "residual_wh": r.get("residual_wh", ""),
                "absolute_error_wh": r.get("absolute_error_wh", "") if family.startswith("W1") or family.startswith("W3") or family.startswith("W4") else r.get("mean_abs_error_wh", ""),
                "R1_TARGET_LEVEL": w["R1_TARGET_LEVEL"],
                "R2_EXTREME_HIGH": w["R2_EXTREME_HIGH"],
                "R3_CHANGE_MAGNITUDE": w["R3_CHANGE_MAGNITUDE"],
                "R4_CHANGE_DIRECTION": w["R4_CHANGE_DIRECTION"],
                "R5_TIME_OF_DAY": w["R5_TIME_OF_DAY"],
                "R6_DAY_TYPE": w["R6_DAY_TYPE"],
                "seed42_y_pred_wh": w["seed42_y_pred_wh"],
                "seed42_residual_wh": w["seed42_residual_wh"],
                "seed42_abs_error_wh": w["seed42_abs_error_wh"],
                "seed123_y_pred_wh": w["seed123_y_pred_wh"],
                "seed123_residual_wh": w["seed123_residual_wh"],
                "seed123_abs_error_wh": w["seed123_abs_error_wh"],
                "seed2026_y_pred_wh": w["seed2026_y_pred_wh"],
                "seed2026_residual_wh": w["seed2026_residual_wh"],
                "seed2026_abs_error_wh": w["seed2026_abs_error_wh"],
                "mean_abs_error_wh": r.get("mean_abs_error_wh", w["mean_abs_error_wh"]),
                "cross_seed_consensus_class": _consensus(w),
                "context_radius": 6,
                "max_context_rows": 13,
                "input_window_lookback": 72,
                "input_feature_count": 33,
                "input_feature_set": "FS2_TF1",
                "input_window_reference_id": f"WIN_{tid}_L72_F33_FS2TF1",
                "casebook_role": "FROZEN_PHASE51_C_SELECTION",
                "casebook_ranking_source": rel,
            }
            cases.append(case)

    return cases


def _consensus(w: dict[str, str]) -> str:
    """Compute Phase 49 cross-seed sign consensus class per target."""
    def sign(x: float) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)
    s = sorted([
        sign(float(w["seed42_residual_wh"])),
        sign(float(w["seed123_residual_wh"])),
        sign(float(w["seed2026_residual_wh"])),
    ])
    return {
        (1, 1, 1): "ALL_UNDER",
        (-1, -1, -1): "ALL_OVER",
        (0, 0, 0): "ALL_EXACT",
        (-1, 1, 1): "TWO_UNDER_ONE_OVER",
        (-1, -1, 1): "TWO_OVER_ONE_UNDER",
    }.get(tuple(s), "MIXED")


def build_casebook_membership_bridge(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (case_id, selection_family) preserving cross-family
    membership. Case_id is preserved as the row key; multiple rows with
    different selection_family share the same case_id when a target is
    selected by multiple Phase 51-C families.
    """
    out: list[dict[str, Any]] = []
    for c in cases:
        out.append({
            "case_id": c["case_id"],
            "target_id": c["target_id"],
            "selection_family": c["selection_family"],
            "seed": c["seed"],
            "rank": c["rank"],
            "casebook_role": c["casebook_role"],
        })
    return out


def build_unique_case_master(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per unique target_id, aggregating cross-family membership."""
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_target[c["target_id"]].append(c)
    out: list[dict[str, Any]] = []
    for tid in sorted(by_target.keys()):
        members = by_target[tid]
        first = members[0]
        families = sorted({m["selection_family"] for m in members})
        ranks_by_family = {
            m["selection_family"]: m["rank"] for m in members
        }
        out.append({
            "target_id": tid,
            "n_case_memberships": len(members),
            "selection_families": ",".join(families),
            "ranks_by_family": ";".join(
                f"{f}:{ranks_by_family[f]}" for f in families
            ),
            "primary_case_id": first["case_id"],
            "y_true_wh": first["y_true_wh"],
            "R1_TARGET_LEVEL": first["R1_TARGET_LEVEL"],
            "R2_EXTREME_HIGH": first["R2_EXTREME_HIGH"],
            "R3_CHANGE_MAGNITUDE": first["R3_CHANGE_MAGNITUDE"],
            "R4_CHANGE_DIRECTION": first["R4_CHANGE_DIRECTION"],
            "R5_TIME_OF_DAY": first["R5_TIME_OF_DAY"],
            "R6_DAY_TYPE": first["R6_DAY_TYPE"],
            "cross_seed_consensus_class": first["cross_seed_consensus_class"],
        })
    return out
