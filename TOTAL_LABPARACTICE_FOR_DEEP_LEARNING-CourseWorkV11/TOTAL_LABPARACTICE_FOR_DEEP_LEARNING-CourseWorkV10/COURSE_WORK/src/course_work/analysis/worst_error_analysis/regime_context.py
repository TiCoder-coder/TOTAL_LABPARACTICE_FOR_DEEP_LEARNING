"""Phase 51-E — Regime Context (O51.18) + Regime Enrichment (O51.19).

Joins frozen Phase 50 regime labels onto Phase 51-C selected targets and
computes per-regime-family prevalence / enrichment diagnostics.

Phase 50 labels are loaded from ``phase51_target_level_working_table.csv``,
which was already constructed by Phase 51-B and carries R1_TARGET_LEVEL,
R2_EXTREME_HIGH, R3_CHANGE_MAGNITUDE, R4_CHANGE_DIRECTION, R5_TIME_OF_DAY,
R6_DAY_TYPE for every one of the 2961 targets.

No regime recomputation. No threshold tuning. No Cartesian regime mining.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from . import contract
from course_work.utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"

# Canonical Phase 50 regime families × labels (from frozen contract)
REGIME_FAMILIES: dict[str, list[str]] = {
    "R1_TARGET_LEVEL": ["TL_LOW", "TL_MID", "TL_HIGH"],
    "R2_EXTREME_HIGH": ["EXTREME_HIGH", "NON_EXTREME"],
    "R3_CHANGE_MAGNITUDE": [
        "CHANGE_NORMAL", "CHANGE_RAPID", "CHANGE_UNCLASSIFIED",
    ],
    "R4_CHANGE_DIRECTION": [
        "DIR_DOWN", "DIR_FLAT", "DIR_UP", "DIR_UNCLASSIFIED",
    ],
    "R5_TIME_OF_DAY": [
        "TOD_NIGHT", "TOD_MORNING", "TOD_AFTERNOON", "TOD_EVENING",
    ],
    "R6_DAY_TYPE": ["DAY_WEEKDAY", "DAY_WEEKEND"],
}

PHASE50_ASSIGNMENT_SHA = (
    "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
)


def _load_working_table(project_root: Path) -> list[dict[str, str]]:
    fp = project_root / PHASE51_DIR_REL / "phase51_target_level_working_table.csv"
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_ranking(project_root: Path, rel: str) -> list[dict[str, str]]:
    fp = project_root / PHASE51_DIR_REL / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _seed_targets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """For flattened per-seed ranking files (W1, W3, W4)."""
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["seed"], set()).add(r["target_id"])
    return out


def attach_regime_labels(
    project_root: Path,
    targets: list[str],
) -> list[dict[str, Any]]:
    """Attach all 6 Phase 50 regime labels to each target_id.

    Returns rows keyed by target_id, carrying the regime labels.
    Many-to-one join via target_id; verifies uniqueness (Phase 50 must
    give exactly one label per target per family).
    """
    wt = _load_working_table(project_root)
    wt_by_id = {r["target_id"]: r for r in wt}

    out: list[dict[str, Any]] = []
    for tid in targets:
        w = wt_by_id.get(tid)
        if w is None:
            raise RuntimeError(
                f"Target {tid} not found in Phase 50 working-table — "
                f"alignment audit failed."
            )
        out.append({
            "target_id": tid,
            "R1_TARGET_LEVEL": w["R1_TARGET_LEVEL"],
            "R2_EXTREME_HIGH": w["R2_EXTREME_HIGH"],
            "R3_CHANGE_MAGNITUDE": w["R3_CHANGE_MAGNITUDE"],
            "R4_CHANGE_DIRECTION": w["R4_CHANGE_DIRECTION"],
            "R5_TIME_OF_DAY": w["R5_TIME_OF_DAY"],
            "R6_DAY_TYPE": w["R6_DAY_TYPE"],
        })
    return out


def verify_join_uniqueness(project_root: Path) -> dict[str, int]:
    """Return counts: must be 2961 rows × exactly one label per family."""
    wt = _load_working_table(project_root)
    assert len(wt) == 2961
    families = list(REGIME_FAMILIES.keys())
    counts: dict[str, int] = {"rows": len(wt)}
    for fam in families:
        # No duplicate (target_id, family_label) pair
        pairs = {(r["target_id"], r[fam]) for r in wt}
        counts[fam] = len(pairs)
    return counts


def compute_global_prevalence(project_root: Path) -> dict[str, dict[str, Any]]:
    """For each regime family, count global Test prevalence over 2961."""
    wt = _load_working_table(project_root)
    n = len(wt)
    out: dict[str, dict[str, Any]] = {}
    for fam, labels in REGIME_FAMILIES.items():
        cnt = Counter(r[fam] for r in wt)
        out[fam] = {
            "denominator": n,
            "counts": {lbl: cnt.get(lbl, 0) for lbl in labels},
            "total_classified": sum(cnt.values()),
        }
    return out


def compute_selected_prevalence(
    project_root: Path,
    selection_set: list[str],
) -> dict[str, dict[str, Any]]:
    """For each regime family, count prevalence in a selected target set."""
    labeled = attach_regime_labels(project_root, selection_set)
    n = len(labeled)
    out: dict[str, dict[str, Any]] = {}
    for fam, labels in REGIME_FAMILIES.items():
        cnt = Counter(r[fam] for r in labeled)
        out[fam] = {
            "denominator": n,
            "counts": {lbl: cnt.get(lbl, 0) for lbl in labels},
            "total_classified": sum(cnt.values()),
        }
    return out


def build_regime_context_table(
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Attach regime labels to each Phase 51-C ranking family.

    Returns a dict mapping selection_family → list of rows.
    """
    wt = _load_working_table(project_root)

    families: dict[str, tuple[str, str]] = {  # (ranking_csv, kind)
        "W1_PER_SEED_WORST": ("worst_per_seed_top20.csv", "per_seed"),
        "W3_UNDERPREDICTION_WORST": ("worst_underprediction_top10.csv", "per_seed"),
        "W4_OVERPREDICTION_WORST": ("worst_overprediction_top10.csv", "per_seed"),
        "W3_SH_SHARED_ALL_UNDER": ("shared_all_under_top10.csv", "shared"),
        "W4_SH_SHARED_ALL_OVER": ("shared_all_over_top10.csv", "shared"),
    }

    out: dict[str, list[dict[str, Any]]] = {}

    for family, (rel, kind) in families.items():
        rows = _load_ranking(project_root, rel)
        wt_by_id = {r["target_id"]: r for r in wt}
        annotated: list[dict[str, Any]] = []
        for r in rows:
            w = wt_by_id[r["target_id"]]
            annotated.append({
                "selection_family": family,
                "seed": r.get("seed", "ALL"),
                "rank": r["rank"],
                "target_id": r["target_id"],
                "target_timestamp": r.get("target_timestamp", w["target_timestamp"]),
                "y_true_wh": r.get("y_true_wh", w["y_true_wh"]),
                "R1_TARGET_LEVEL": w["R1_TARGET_LEVEL"],
                "R2_EXTREME_HIGH": w["R2_EXTREME_HIGH"],
                "R3_CHANGE_MAGNITUDE": w["R3_CHANGE_MAGNITUDE"],
                "R4_CHANGE_DIRECTION": w["R4_CHANGE_DIRECTION"],
                "R5_TIME_OF_DAY": w["R5_TIME_OF_DAY"],
                "R6_DAY_TYPE": w["R6_DAY_TYPE"],
            })
        out[family] = annotated

    # W2 (Shared Worst)
    w2_rows = _load_ranking(project_root, "worst_shared_top20.csv")
    wt_by_id = {r["target_id"]: r for r in wt}
    out["W2_SHARED_WORST"] = [
        {
            "selection_family": "W2_SHARED_WORST",
            "seed": "ALL",
            "rank": r["rank"],
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": r["y_true_wh"],
            "mean_abs_error_wh": r["mean_abs_error_wh"],
            "R1_TARGET_LEVEL": wt_by_id[r["target_id"]]["R1_TARGET_LEVEL"],
            "R2_EXTREME_HIGH": wt_by_id[r["target_id"]]["R2_EXTREME_HIGH"],
            "R3_CHANGE_MAGNITUDE": wt_by_id[r["target_id"]]["R3_CHANGE_MAGNITUDE"],
            "R4_CHANGE_DIRECTION": wt_by_id[r["target_id"]]["R4_CHANGE_DIRECTION"],
            "R5_TIME_OF_DAY": wt_by_id[r["target_id"]]["R5_TIME_OF_DAY"],
            "R6_DAY_TYPE": wt_by_id[r["target_id"]]["R6_DAY_TYPE"],
        }
        for r in w2_rows
    ]

    return out


def build_enrichment_table(
    project_root: Path,
) -> list[dict[str, Any]]:
    """Build O51.19 regime-enrichment rows.

    Selection families evaluated:
      W1 per seed (42 / 123 / 2026)            → K=20 each
      W2 SHARED_WORST                            → K=20
      W3 per seed (residual > 0)                 → K=10 each
      W4 per seed (residual < 0)                 → K=10 each
      W3_SH SHARED_ALL_UNDER                     → K≤10
      W4_SH SHARED_ALL_OVER                      → K≤10

    For each (selection_family, seed, regime_family, regime_label):
      selected_count
      selected_prevalence = selected_count / |S|
      global_count
      global_prevalence = global_count / 2961
      enrichment_ratio = selected_prevalence / global_prevalence   (if global>0)
      prevalence_difference = selected_prevalence − global_prevalence
    """
    wt = _load_working_table(project_root)
    n_test = len(wt)
    assert n_test == 2961, f"Expected 2961, got {n_test}"

    # Selection sets
    def _flatten_per_seed(rel: str) -> dict[str, list[str]]:
        rows = _load_ranking(project_root, rel)
        out_d: dict[str, list[str]] = {}
        for r in rows:
            out_d.setdefault(r["seed"], []).append(r["target_id"])
        return out_d

    w1 = _flatten_per_seed("worst_per_seed_top20.csv")
    w3 = _flatten_per_seed("worst_underprediction_top10.csv")
    w4 = _flatten_per_seed("worst_overprediction_top10.csv")
    w2_targets = [r["target_id"] for r in _load_ranking(project_root, "worst_shared_top20.csv")]
    w3sh_targets = [r["target_id"] for r in _load_ranking(project_root, "shared_all_under_top10.csv")]
    w4sh_targets = [r["target_id"] for r in _load_ranking(project_root, "shared_all_over_top10.csv")]

    # Global prevalence
    global_cnt: dict[str, Counter] = {}
    for fam in REGIME_FAMILIES:
        global_cnt[fam] = Counter(r[fam] for r in wt)

    # Selection-family → seed → target_id list
    selections: list[tuple[str, str, list[str]]] = []
    for seed in contract.SEEDS:
        selections.append(("W1_PER_SEED_WORST", seed, w1[seed]))
    for seed in contract.SEEDS:
        selections.append(("W3_UNDERPREDICTION_WORST", seed, w3[seed]))
    for seed in contract.SEEDS:
        selections.append(("W4_OVERPREDICTION_WORST", seed, w4[seed]))
    selections.append(("W2_SHARED_WORST", "ALL", w2_targets))
    selections.append(("W3_SH_SHARED_ALL_UNDER", "ALL", w3sh_targets))
    selections.append(("W4_SH_SHARED_ALL_OVER", "ALL", w4sh_targets))

    wt_by_id = {r["target_id"]: r for r in wt}

    rows: list[dict[str, Any]] = []
    for family, seed, sel_targets in selections:
        k = len(sel_targets)
        if k == 0:
            continue
        # Selected prevalence counter
        for fam in REGIME_FAMILIES:
            sel_counter = Counter(wt_by_id[tid][fam] for tid in sel_targets)
            for lbl in REGIME_FAMILIES[fam]:
                sel_c = sel_counter.get(lbl, 0)
                gbl_c = global_cnt[fam].get(lbl, 0)
                sel_prev = sel_c / k
                gbl_prev = gbl_c / n_test
                if gbl_prev > 0:
                    enrich = sel_prev / gbl_prev
                    status = "OK"
                else:
                    enrich = None
                    status = "DENOMINATOR_ZERO"
                rows.append({
                    "selection_family": family,
                    "seed": seed,
                    "selection_k": k,
                    "regime_family": fam,
                    "regime_label": lbl,
                    "selected_count": sel_c,
                    "selected_prevalence": round(sel_prev, 9),
                    "global_count": gbl_c,
                    "global_prevalence": round(gbl_prev, 9),
                    "enrichment_ratio": (
                        round(enrich, 9) if enrich is not None else ""
                    ),
                    "prevalence_difference": round(sel_prev - gbl_prev, 9),
                    "status": status,
                    "phase50_assignment_sha256": PHASE50_ASSIGNMENT_SHA,
                    "selection_contract_sha256": (
                        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
                    ),
                })

    return rows
