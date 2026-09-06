"""Phase 51-E — Phase 49 cross-seed residual-sign consensus context.

Loads the canonical Phase 49 residual long table (target-level residual
already in working table) and computes the per-target cross-seed sign
consensus class for every selected target. The Phase 49 taxonomy is:

  ALL_UNDER     — all three seeds have residual > 0
  ALL_OVER      — all three seeds have residual < 0
  ALL_EXACT     — all three seeds have residual == 0
  TWO_UNDER_ONE_OVER
  TWO_OVER_ONE_UNDER
  MIXED         — degenerate; included for completeness

No taxonomy reduction. No silent collapse.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _load_working_table(project_root: Path) -> list[dict[str, str]]:
    fp = project_root / PHASE51_DIR_REL / "phase51_target_level_working_table.csv"
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)


def _consensus_class(r: dict[str, str]) -> str:
    u = _to_float(r["seed42_residual_wh"])
    o = _to_float(r["seed123_residual_wh"])
    t = _to_float(r["seed2026_residual_wh"])
    sgn = lambda x: 0 if x == 0 else (1 if x > 0 else -1)
    sigs = sorted([sgn(u), sgn(o), sgn(t)])
    if sigs == [1, 1, 1]:
        return "ALL_UNDER"
    if sigs == [-1, -1, -1]:
        return "ALL_OVER"
    if sigs == [0, 0, 0]:
        return "ALL_EXACT"
    if sigs == [-1, 1, 1]:
        return "TWO_UNDER_ONE_OVER"
    if sigs == [-1, -1, 1]:
        return "TWO_OVER_ONE_UNDER"
    return "MIXED"


def build_sign_consensus_context(
    project_root: Path,
    selection_target_ids: list[str],
) -> list[dict[str, Any]]:
    wt = _load_working_table(project_root)
    wt_by_id = {r["target_id"]: r for r in wt}
    out: list[dict[str, Any]] = []
    for tid in selection_target_ids:
        r = wt_by_id[tid]
        out.append({
            "target_id": tid,
            "seed42_residual_sign": "UNDERPREDICTION" if _to_float(r["seed42_residual_wh"]) > 0
                                     else ("OVERPREDICTION" if _to_float(r["seed42_residual_wh"]) < 0 else "EXACT_ZERO"),
            "seed123_residual_sign": "UNDERPREDICTION" if _to_float(r["seed123_residual_wh"]) > 0
                                      else ("OVERPREDICTION" if _to_float(r["seed123_residual_wh"]) < 0 else "EXACT_ZERO"),
            "seed2026_residual_sign": "UNDERPREDICTION" if _to_float(r["seed2026_residual_wh"]) > 0
                                       else ("OVERPREDICTION" if _to_float(r["seed2026_residual_wh"]) < 0 else "EXACT_ZERO"),
            "cross_seed_consensus_class": _consensus_class(r),
        })
    return out
