"""Phase 51-D — Error Concentration Analysis (O51.17).

For each seed compute global SAE / SSE over the FULL 2961-target Test
population. Then compute Top-K (K in {1, 5, 10, 20}) cumulative
contribution of the canonical W1 ranking per seed.

No 3N iid pooling — each seed is evaluated independently against its own
2961 targets and its own canonical ranking.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from . import contract
from ..utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _load_csv(project_root: Path, rel: str) -> list[dict[str, str]]:
    fp = (project_root / PHASE51_DIR_REL) / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)


def build_error_concentration_table(
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build O51.17 error-concentration rows.

    Schema per row:
      seed | k | sae_top_k | sse_top_k | sae_share | sse_share
    """
    if project_root is None:
        project_root = get_project_root()

    wt_rows = _load_csv(project_root, "phase51_target_level_working_table.csv")
    w1_rows = _load_csv(project_root, "worst_per_seed_top20.csv")

    # Global SAE/SSE per seed across FULL 2961
    global_sae: dict[str, float] = {}
    global_sse: dict[str, float] = {}
    for seed in contract.SEEDS:
        abs_field = f"seed{seed}_abs_error_wh"
        sq_field = f"seed{seed}_squared_error_wh2"
        global_sae[seed] = sum(_to_float(r[abs_field]) for r in wt_rows)
        global_sse[seed] = sum(_to_float(r[sq_field]) for r in wt_rows)

    # W1 seed → rank-ordered list (sorted by rank)
    w1_by_seed: dict[str, list[dict[str, str]]] = {}
    for r in w1_rows:
        w1_by_seed.setdefault(r["seed"], []).append(r)
    for seed in w1_by_seed:
        w1_by_seed[seed] = sorted(w1_by_seed[seed], key=lambda r: int(r["rank"]))

    rows: list[dict[str, Any]] = []
    for seed in contract.SEEDS:
        ranked = w1_by_seed[seed]
        sae_running = 0.0
        sse_running = 0.0
        for k in (1, 5, 10, 20):
            topk = ranked[:k]
            sae_k = sum(_to_float(r["absolute_error_wh"]) for r in topk)
            sse_k = sum(_to_float(r["squared_error_wh2"]) for r in topk)
            sae_share = sae_k / global_sae[seed] if global_sae[seed] else 0.0
            sse_share = sse_k / global_sse[seed] if global_sse[seed] else 0.0
            rows.append({
                "seed": seed,
                "k": k,
                "sae_top_k": round(sae_k, 6),
                "sse_top_k": round(sse_k, 6),
                "sae_share": round(sae_share, 9),
                "sse_share": round(sse_share, 9),
                "global_sae_seed": round(global_sae[seed], 6),
                "global_sse_seed": round(global_sse[seed], 6),
                "n_population": len(wt_rows),
                "top_k_source": "worst_per_seed_top20.csv",
            })

    return rows
