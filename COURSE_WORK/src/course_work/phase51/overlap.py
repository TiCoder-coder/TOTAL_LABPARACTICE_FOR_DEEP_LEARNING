"""Phase 51-D — Cross-seed Top-K Overlap (O51.15) + Membership Matrix (O51.16).

Inputs (read-only):
  - worst_per_seed_top20.csv         (O51.8, W1)
  - worst_shared_top20.csv           (O51.10, W2)
  - worst_underprediction_top10.csv  (O51.11, W3)
  - worst_overprediction_top10.csv   (O51.12, W4)
  - shared_all_under_top10.csv       (O51.13, W3_SH)
  - shared_all_over_top10.csv        (O51.14, W4_SH)

Produced:
  - seed_overlap_table.csv           (O51.15): pairwise + 3-way overlap / Jaccard
  - worst_case_membership_matrix.csv (O51.16): binary membership per list

Determinism:
  - Target IDs are loaded from canonical Phase 51-C corrected CSVs.
  - No sorting, no ranking, no case selection by hand.
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any

from . import contract
from ..utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _artifact_dir(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    return root / PHASE51_DIR_REL


def _read_seed_sets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Return {seed: set(target_ids)} from a flattened W1/W3/W4 ranking CSV."""
    out: dict[str, set[str]] = {}
    for r in rows:
        seed = r["seed"]
        out.setdefault(seed, set()).add(r["target_id"])
    return out


def _load_csv(project_root: Path, rel: str) -> list[dict[str, str]]:
    fp = _artifact_dir(project_root) / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_w1_seed_sets(project_root: Path) -> dict[str, set[str]]:
    rows = _load_csv(project_root, "worst_per_seed_top20.csv")
    return _read_seed_sets(rows)


def _load_w3_seed_sets(project_root: Path) -> dict[str, set[str]]:
    rows = _load_csv(project_root, "worst_underprediction_top10.csv")
    return _read_seed_sets(rows)


def _load_w4_seed_sets(project_root: Path) -> dict[str, set[str]]:
    rows = _load_csv(project_root, "worst_overprediction_top10.csv")
    return _read_seed_sets(rows)


def _load_w2_set(project_root: Path) -> set[str]:
    rows = _load_csv(project_root, "worst_shared_top20.csv")
    return {r["target_id"] for r in rows}


def _load_w3_shared_set(project_root: Path) -> set[str]:
    rows = _load_csv(project_root, "shared_all_under_top10.csv")
    return {r["target_id"] for r in rows}


def _load_w4_shared_set(project_root: Path) -> set[str]:
    rows = _load_csv(project_root, "shared_all_over_top10.csv")
    return {r["target_id"] for r in rows}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _compute_pairwise(
    sets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    out = []
    for s1, s2 in combinations(sorted(sets.keys()), 2):
        a, b = sets[s1], sets[s2]
        union = a | b
        out.append({
            "family_a": s1,
            "family_b": s2,
            "intersection_count": len(a & b),
            "union_count": len(union),
            "jaccard": _jaccard(a, b),
        })
    return out


def _compute_three_way(sets: dict[str, set[str]]) -> dict[str, Any]:
    keys = sorted(sets.keys())
    inter = sets[keys[0]] & sets[keys[1]] & sets[keys[2]]
    union = sets[keys[0]] | sets[keys[1]] | sets[keys[2]]
    return {
        "intersection_count": len(inter),
        "intersection_targets": sorted(inter),
        "union_count": len(union),
        "k": len(next(iter(sets.values()))),  # e.g. 20
        "fraction_of_k": (
            len(inter) / len(next(iter(sets.values()))) if next(iter(sets.values())) else 0.0
        ),
    }


def build_seed_overlap_table(project_root: Path | None = None) -> list[dict[str, Any]]:
    """Build O51.15 seed-overlap rows."""
    # callers pass the project root already; no path wrapping needed.

    rows: list[dict[str, Any]] = []

    # A. W1
    w1_sets = _load_w1_seed_sets(project_root)
    w1_pair = _compute_pairwise(w1_sets)
    for r in w1_pair:
        rows.append({
            "section": "W1_PAIRWISE",
            "group_label": f"{r['family_a']}-{r['family_b']}",
            "set_a": r["family_a"],
            "set_b": r["family_b"],
            "intersection_count": r["intersection_count"],
            "union_count": r["union_count"],
            "jaccard": round(r["jaccard"], 6),
            "extra": "",
        })
    w1_three = _compute_three_way(w1_sets)
    rows.append({
        "section": "W1_3WAY",
        "group_label": "42-123-2026",
        "set_a": "42",
        "set_b": "123-2026",
        "intersection_count": w1_three["intersection_count"],
        "union_count": w1_three["union_count"],
        "jaccard": "",  # 3-way Jaccard not standard (use union_count)
        "extra": f"intersection={','.join(w1_three['intersection_targets'])}",
    })

    # B. W2 vs W1_seed
    w2_set = _load_w2_set(project_root)
    for seed in contract.SEEDS:
        s = w1_sets[seed]
        union = s | w2_set
        rows.append({
            "section": "W2_VS_W1",
            "group_label": f"W2_vs_seed{seed}",
            "set_a": "W2",
            "set_b": f"seed{seed}",
            "intersection_count": len(s & w2_set),
            "union_count": len(union),
            "jaccard": round(_jaccard(s, w2_set), 6),
            "extra": "",
        })

    # C. W3
    w3_sets = _load_w3_seed_sets(project_root)
    for r in _compute_pairwise(w3_sets):
        rows.append({
            "section": "W3_PAIRWISE",
            "group_label": f"{r['family_a']}-{r['family_b']}",
            "set_a": r["family_a"],
            "set_b": r["family_b"],
            "intersection_count": r["intersection_count"],
            "union_count": r["union_count"],
            "jaccard": round(r["jaccard"], 6),
            "extra": "",
        })
    w3_three = _compute_three_way(w3_sets)
    rows.append({
        "section": "W3_3WAY",
        "group_label": "42-123-2026",
        "set_a": "42",
        "set_b": "123-2026",
        "intersection_count": w3_three["intersection_count"],
        "union_count": w3_three["union_count"],
        "jaccard": "",
        "extra": f"intersection={','.join(w3_three['intersection_targets'])}",
    })

    # D. W4
    w4_sets = _load_w4_seed_sets(project_root)
    for r in _compute_pairwise(w4_sets):
        rows.append({
            "section": "W4_PAIRWISE",
            "group_label": f"{r['family_a']}-{r['family_b']}",
            "set_a": r["family_a"],
            "set_b": r["family_b"],
            "intersection_count": r["intersection_count"],
            "union_count": r["union_count"],
            "jaccard": round(r["jaccard"], 6),
            "extra": "",
        })
    w4_three = _compute_three_way(w4_sets)
    rows.append({
        "section": "W4_3WAY",
        "group_label": "42-123-2026",
        "set_a": "42",
        "set_b": "123-2026",
        "intersection_count": w4_three["intersection_count"],
        "union_count": w4_three["union_count"],
        "jaccard": "",
        "extra": f"intersection={','.join(w4_three['intersection_targets'])}",
    })

    # E. W3_SH vs W3_seed
    w3_sh = _load_w3_shared_set(project_root)
    for seed in contract.SEEDS:
        s = w3_sets[seed]
        rows.append({
            "section": "W3_SH_VS_W3",
            "group_label": f"W3_SH_vs_seed{seed}",
            "set_a": "W3_SH",
            "set_b": f"seed{seed}",
            "intersection_count": len(s & w3_sh),
            "union_count": len(s | w3_sh),
            "jaccard": round(_jaccard(s, w3_sh), 6),
            "extra": "",
        })

    # F. W4_SH vs W4_seed
    w4_sh = _load_w4_shared_set(project_root)
    for seed in contract.SEEDS:
        s = w4_sets[seed]
        rows.append({
            "section": "W4_SH_VS_W4",
            "group_label": f"W4_SH_vs_seed{seed}",
            "set_a": "W4_SH",
            "set_b": f"seed{seed}",
            "intersection_count": len(s & w4_sh),
            "union_count": len(s | w4_sh),
            "jaccard": round(_jaccard(s, w4_sh), 6),
            "extra": "",
        })

    return rows


def build_membership_matrix(
    project_root: Path | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Build O51.16 binary membership matrix rows."""
    # project root is the absolute project root; _load_csv uses absolute paths.

    w1_sets = _load_w1_seed_sets(project_root)
    w3_sets = _load_w3_seed_sets(project_root)
    w4_sets = _load_w4_seed_sets(project_root)
    w2 = _load_w2_set(project_root)
    w3sh = _load_w3_shared_set(project_root)
    w4sh = _load_w4_shared_set(project_root)

    all_targets: set[str] = set()
    for s in (w1_sets.values(), w3_sets.values(), w4_sets.values()):
        for s_set in s:
            all_targets |= s_set
    all_targets |= w2
    all_targets |= w3sh
    all_targets |= w4sh

    header = [
        "target_id",
        "W1_seed42", "W1_seed123", "W1_seed2026", "W2",
        "W3_seed42", "W3_seed123", "W3_seed2026", "W3_SH",
        "W4_seed42", "W4_seed123", "W4_seed2026", "W4_SH",
    ]
    rows = []
    for tid in sorted(all_targets):
        rows.append([
            tid,
            1 if tid in w1_sets.get("42", set()) else 0,
            1 if tid in w1_sets.get("123", set()) else 0,
            1 if tid in w1_sets.get("2026", set()) else 0,
            1 if tid in w2 else 0,
            1 if tid in w3_sets.get("42", set()) else 0,
            1 if tid in w3_sets.get("123", set()) else 0,
            1 if tid in w3_sets.get("2026", set()) else 0,
            1 if tid in w3sh else 0,
            1 if tid in w4_sets.get("42", set()) else 0,
            1 if tid in w4_sets.get("123", set()) else 0,
            1 if tid in w4_sets.get("2026", set()) else 0,
            1 if tid in w4sh else 0,
        ])
    return header, rows
