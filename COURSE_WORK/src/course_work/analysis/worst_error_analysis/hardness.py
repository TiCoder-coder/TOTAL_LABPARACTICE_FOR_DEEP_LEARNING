"""Phase 51-D — Hardness vs Cross-Seed Disagreement + Seed-Spread Context (O51.20).

For each Test target, build the deterministic target-level hardness
diagnostics from the EXISTING three-seed absolute errors already in the
working table:

  mean_abs_error_wh  (already on working table; re-verified)
  median_abs_error_wh (cross-seed median)
  min_abs_error_wh   (cross-seed min)
  max_abs_error_wh   (cross-seed max)
  sample_sd_abs_error_wh   (sample SD with ddof=1)

Plus the Phase-48 seed-spread fields that are already embedded in the
working table (seed_mean_prediction, seed_std_prediction, seed_range_prediction).

Outputs a single CSV `hardness_vs_seed_disagreement.csv` covering:
  - W2 (Shared top20)
  - W1 per-seed top20 (across union)
  - Shared three-seed worst subset (intersection of W1 across all 3 seeds)

Then computes aggregate summaries per grouping without inventing
thresholds or regimes.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from . import contract
from course_work.utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _load_csv(project_root: Path, rel: str) -> list[dict[str, str]]:
    fp = (project_root / PHASE51_DIR_REL) / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)


def _abs_errors_for_target(r: dict[str, str]) -> list[float]:
    return [
        _to_float(r["seed42_abs_error_wh"]),
        _to_float(r["seed123_abs_error_wh"]),
        _to_float(r["seed2026_abs_error_wh"]),
    ]


def per_target_hardness(r: dict[str, str]) -> dict[str, Any]:
    """Append deterministic diagnostic fields to a working-table row."""
    aes = _abs_errors_for_target(r)
    return {
        **r,
        "computed_mean_abs_error_wh": round(statistics.fmean(aes), 9),
        "computed_median_abs_error_wh": round(statistics.median(aes), 9),
        "computed_min_abs_error_wh": round(min(aes), 9),
        "computed_max_abs_error_wh": round(max(aes), 9),
        "computed_sample_sd_abs_error_wh": round(
            statistics.stdev(aes), 9  # ddof=1
        ),
        # Phase-48 spread fields are already in the row.
        "phase48_seed_mean_prediction": r.get("seed_mean_prediction", ""),
        "phase48_seed_std_prediction": r.get("seed_std_prediction", ""),
        "phase48_seed_range_prediction": r.get("seed_range_prediction", ""),
    }


def _group_summary(
    rows: list[dict[str, Any]],
    group_label: str,
    source: str,
) -> dict[str, Any]:
    """Aggregate descriptive summary for a set of target rows."""
    if not rows:
        return {
            "group_label": group_label,
            "source": source,
            "n_targets": 0,
            "mean_of_mean_abs_error_wh": 0.0,
            "median_of_mean_abs_error_wh": 0.0,
            "mean_of_sample_sd_abs_error_wh": 0.0,
            "mean_of_seed_std_prediction": 0.0,
            "mean_of_seed_range_prediction": 0.0,
        }
    maes = [_to_float(r["computed_mean_abs_error_wh"]) for r in rows]
    sds = [_to_float(r["computed_sample_sd_abs_error_wh"]) for r in rows]
    std_preds = [_to_float(r["phase48_seed_std_prediction"]) for r in rows
                 if r.get("phase48_seed_std_prediction") not in ("", None)]
    rng_preds = [_to_float(r["phase48_seed_range_prediction"]) for r in rows
                 if r.get("phase48_seed_range_prediction") not in ("", None)]
    return {
        "group_label": group_label,
        "source": source,
        "n_targets": len(rows),
        "mean_of_mean_abs_error_wh": round(statistics.fmean(maes), 9),
        "median_of_mean_abs_error_wh": round(statistics.median(maes), 9),
        "mean_of_sample_sd_abs_error_wh": round(
            statistics.fmean(sds) if sds else 0.0, 9
        ),
        "mean_of_seed_std_prediction": round(
            statistics.fmean(std_preds) if std_preds else 0.0, 9
        ),
        "mean_of_seed_range_prediction": round(
            statistics.fmean(rng_preds) if rng_preds else 0.0, 9
        ),
    }


def build_hardness_table(
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build O51.20 rows + group summaries."""
    if project_root is None:
        project_root = get_project_root()

    wt_rows = _load_csv(project_root, "phase51_target_level_working_table.csv")
    wt_by_id = {r["target_id"]: r for r in wt_rows}

    # Membership sets
    w1_rows = _load_csv(project_root, "worst_per_seed_top20.csv")
    w2_rows = _load_csv(project_root, "worst_shared_top20.csv")
    w1_by_seed: dict[str, set[str]] = {}
    for r in w1_rows:
        w1_by_seed.setdefault(r["seed"], set()).add(r["target_id"])
    w2_set = {r["target_id"] for r in w2_rows}
    shared_three = (
        w1_by_seed["42"] & w1_by_seed["123"] & w1_by_seed["2026"]
    )

    # Group definitions
    groups: list[tuple[str, str, list[str]]] = [
        ("W2_SHARED_WORST_TOP20", "worst_shared_top20.csv",
         sorted(w2_set)),
        ("W1_PER_SEED_WORST_TOP20_UNION", "worst_per_seed_top20.csv",
         sorted(set().union(*w1_by_seed.values()))),
        ("SHARED_THREE_SEED_WORST_INTERSECTION",
         "worst_per_seed_top20.csv", sorted(shared_three)),
        ("ALL_TEST_POPULATION_BASELINE",
         "phase51_target_level_working_table.csv",
         sorted(wt_by_id.keys())),
    ]

    all_target_ids = sorted(set().union(
        w2_set, *[w1_by_seed[s] for s in w1_by_seed], shared_three,
    ))

    # Per-target rows (only for groupings; union set covers everything)
    per_target_rows: list[dict[str, Any]] = []
    for tid in all_target_ids:
        r = wt_by_id[tid]
        per_target_rows.append(per_target_hardness(r))

    # Group summary rows
    summaries: list[dict[str, Any]] = []
    for label, source, ids in groups:
        subset = [per_target_hardness(wt_by_id[tid]) for tid in ids]
        summaries.append(_group_summary(subset, label, source))

    # Annotate per-target rows with membership / group labels
    annotated: list[dict[str, Any]] = []
    for r in per_target_rows:
        tid = r["target_id"]
        out = dict(r)
        out["in_W1_seed42"] = 1 if tid in w1_by_seed.get("42", set()) else 0
        out["in_W1_seed123"] = 1 if tid in w1_by_seed.get("123", set()) else 0
        out["in_W1_seed2026"] = 1 if tid in w1_by_seed.get("2026", set()) else 0
        out["in_W2"] = 1 if tid in w2_set else 0
        out["in_shared_three_seed_worst"] = 1 if tid in shared_three else 0
        out["group_label"] = (
            "SHARED_THREE_SEED_WORST_INTERSECTION" if tid in shared_three else
            "W2_SHARED_WORST_TOP20" if tid in w2_set else
            "W1_PER_SEED_WORST_TOP20_UNION"
        )
        annotated.append(out)

    return annotated, summaries
