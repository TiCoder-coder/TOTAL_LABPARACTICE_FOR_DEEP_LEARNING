"""Phase 44 — Transformer ranking with exact tie-break.

Primary key : pooled_outer_rmse_wh (ascending)
Tie-break 1 : worst_fold_rmse_wh  (ascending)
Tie-break 2 : fold_rmse_sd_wh     (ascending)
Tie-break 3 : shortlist_position  (ascending — earlier first)

LSTM / Persistence do NOT enter Transformer candidate tie-break
(plan §78). They are reported as family-level comparisons in O44.29.
"""
from __future__ import annotations

from dataclasses import dataclass

from course_work.rolling_origin.pooling import PooledMetrics, MacroRobustnessMetrics


@dataclass(frozen=True)
class TransformerRankingEntry:
    rank: int
    candidate_id: str
    pooled_rmse_wh: float
    worst_fold_rmse_wh: float
    fold_rmse_sd_wh: float
    shortlist_position: int
    pooled_metrics: PooledMetrics
    macro_metrics: MacroRobustnessMetrics

    def as_row(self) -> dict:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "pooled_rmse_wh": self.pooled_rmse_wh,
            "worst_fold_rmse_wh": self.worst_fold_rmse_wh,
            "fold_rmse_sd_wh": self.fold_rmse_sd_wh,
            "shortlist_position": self.shortlist_position,
        }


def rank_transformers(
    *,
    candidate_pooled: dict[str, PooledMetrics],
    candidate_macro: dict[str, MacroRobustnessMetrics],
    shortlist_position: dict[str, int],
    candidate_families: dict[str, str] | None = None,
) -> list[TransformerRankingEntry]:
    """Rank ONLY Transformer candidates (skip LSTM and Persistence).

    Returns a list of TransformerRankingEntry sorted by:
      (pooled_rmse_wh, worst_fold_rmse_wh, fold_rmse_sd_wh, shortlist_position)
    all ascending.
    """
    candidate_families = candidate_families or {}
    tr_pooled = {
        cid: pm
        for cid, pm in candidate_pooled.items()
        if candidate_families.get(cid, "TRANSFORMER_ENCODER") == "TRANSFORMER_ENCODER"
    }
    rows: list[TransformerRankingEntry] = []
    keys = sorted(
        tr_pooled.keys(),
        key=lambda cid: (
            tr_pooled[cid].pooled_rmse_wh,
            candidate_macro[cid].worst_fold_rmse_wh,
            candidate_macro[cid].fold_rmse_sd_wh,
            shortlist_position.get(cid, 10**6),
        ),
    )
    for rank, cid in enumerate(keys, start=1):
        rows.append(
            TransformerRankingEntry(
                rank=rank,
                candidate_id=cid,
                pooled_rmse_wh=tr_pooled[cid].pooled_rmse_wh,
                worst_fold_rmse_wh=candidate_macro[cid].worst_fold_rmse_wh,
                fold_rmse_sd_wh=candidate_macro[cid].fold_rmse_sd_wh,
                shortlist_position=shortlist_position.get(cid, 10**6),
                pooled_metrics=tr_pooled[cid],
                macro_metrics=candidate_macro[cid],
            )
        )
    return rows